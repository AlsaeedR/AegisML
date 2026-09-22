from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = PROJECT_ROOT / ".aegisml_runtime"
DATABASE_PATH = MEMORY_DIR / "audit_memory.db"

_LOCK = Lock()


class StaleArtifactError(RuntimeError):
    """Raised when target pipeline code, model, or dataset has changed since audit initialization."""
    pass


def _init_tables(connection: sqlite3.Connection) -> None:
    """Initializes all required SQLite schema tables with atomic transactions."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_sessions (
            audit_id TEXT PRIMARY KEY,
            session_blob BLOB NOT NULL,
            updated_at TEXT
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_artifacts (
            audit_id TEXT PRIMARY KEY,
            code_hash TEXT,
            model_hash TEXT,
            dataset_hash TEXT,
            registered_at TEXT
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_steps (
            audit_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_seconds REAL,
            step_blob BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (audit_id, step_name)
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_subtests (
            audit_id TEXT NOT NULL,
            test_id TEXT NOT NULL,
            status TEXT,
            severity TEXT,
            evidence_blob BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (audit_id, test_id)
        )
        """
    )
    connection.commit()


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
        check_same_thread=False,
    )
    try:
        # Enable WAL mode for high concurrency and resilience
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")

        # Auto-initialize schema if database is newly created or missing tables
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_artifacts' LIMIT 1;"
        ).fetchone()
        if row is None:
            _init_tables(connection)

        yield connection
    finally:
        connection.close()


def initialize_memory() -> None:
    """Initializes all required SQLite schema tables with atomic transactions."""
    with _LOCK:
        with _connect() as connection:
            _init_tables(connection)


# =====================================================================
# Cryptographic Artifact Fingerprinting
# =====================================================================

def compute_code_hash(code_str: Optional[str]) -> str:
    """Computes a canonical SHA-256 hash for target Python code string."""
    if not code_str:
        return ""
    normalized = "\n".join(line.rstrip() for line in code_str.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: Optional[str]) -> str:
    """Computes SHA-256 for a file on disk (model weights or dataset)."""
    if not file_path or not os.path.exists(file_path):
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def register_audit_artifacts(
    audit_id: str,
    code: Optional[str] = None,
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Registers the cryptographic fingerprints for an audit run.
    If already registered, preserves original baseline.
    """
    code_h = compute_code_hash(code)
    model_h = compute_file_hash(model_path)
    dataset_h = compute_file_hash(dataset_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    with _LOCK:
        with _connect() as connection:
            existing = connection.execute(
                "SELECT code_hash, model_hash, dataset_hash FROM audit_artifacts WHERE audit_id = ?",
                (audit_id,),
            ).fetchone()

            if existing:
                return {
                    "code_hash": existing[0] or "",
                    "model_hash": existing[1] or "",
                    "dataset_hash": existing[2] or "",
                }

            connection.execute(
                """
                INSERT INTO audit_artifacts (audit_id, code_hash, model_hash, dataset_hash, registered_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (audit_id, code_h, model_h, dataset_h, now_iso),
            )
            connection.commit()

    return {
        "code_hash": code_h,
        "model_hash": model_h,
        "dataset_hash": dataset_h,
    }


def find_audit_by_artifacts(
    code: Optional[str] = None,
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Optional[str]:
    """
    Finds the most recent completed audit that evaluated the exact same
    target code, model weights, and dataset (verified via SHA-256).
    """
    code_h = compute_code_hash(code)
    model_h = compute_file_hash(model_path)
    dataset_h = compute_file_hash(dataset_path)

    if not code_h:
        return None

    with _LOCK:
        with _connect() as connection:
            # Look for an audit where code matches and model/dataset matches if provided
            query = """
                SELECT a.audit_id 
                FROM audit_artifacts a
                JOIN audit_steps s ON a.audit_id = s.audit_id
                WHERE a.code_hash = ?
            """
            params: List[Any] = [code_h]

            if model_h:
                query += " AND (a.model_hash = ? OR a.model_hash = '')"
                params.append(model_h)
            if dataset_h:
                query += " AND (a.dataset_hash = ? OR a.dataset_hash = '')"
                params.append(dataset_h)

            query += " GROUP BY a.audit_id ORDER BY a.registered_at DESC LIMIT 1"
            row = connection.execute(query, tuple(params)).fetchone()

    return row[0] if row else None


def verify_artifact_integrity(
    audit_id: str,
    code: Optional[str] = None,
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Verifies that the target pipeline code, model weights, and dataset have not changed
    since the audit baseline was established.

    Raises StaleArtifactError if any fingerprint mismatches and force=False.
    """
    with _LOCK:
        with _connect() as connection:
            row = connection.execute(
                "SELECT code_hash, model_hash, dataset_hash FROM audit_artifacts WHERE audit_id = ?",
                (audit_id,),
            ).fetchone()

    # If not registered yet, register now and succeed
    if not row:
        register_audit_artifacts(audit_id, code, model_path, dataset_path)
        return True

    expected_code, expected_model, expected_dataset = row
    current_code = compute_code_hash(code)
    current_model = compute_file_hash(model_path)
    current_dataset = compute_file_hash(dataset_path)

    mismatches = []
    if expected_code and current_code and expected_code != current_code:
        mismatches.append(f"Pipeline source code was modified (baseline {expected_code[:8]}... vs current {current_code[:8]}...)")
    if expected_model and current_model and expected_model != current_model:
        mismatches.append(f"Model artifact at '{model_path}' was replaced (baseline {expected_model[:8]}... vs current {current_model[:8]}...)")
    if expected_dataset and current_dataset and expected_dataset != current_dataset:
        mismatches.append(f"Evaluation dataset at '{dataset_path}' was modified (baseline {expected_dataset[:8]}... vs current {current_dataset[:8]}...)")

    if mismatches:
        err_msg = (
            f"Audit '{audit_id}' integrity violation: " + "; ".join(mismatches) + ". "
            "Resuming from this checkpoint would produce inconsistent security findings. "
            "Start a fresh audit or pass force=True to override."
        )
        if not force:
            raise StaleArtifactError(err_msg)

    return True


# =====================================================================
# Step-Level Checkpointing
# =====================================================================

def save_step_checkpoint(
    audit_id: str,
    agent_name: str,
    step_name: str,
    step_data: Dict[str, Any],
    duration_seconds: Optional[float] = None,
    status: str = "completed",
) -> None:
    """Persists a step's output dictionary atomically."""
    if not audit_id:
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    blob = pickle.dumps(step_data, protocol=pickle.HIGHEST_PROTOCOL)

    with _LOCK:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_steps (audit_id, step_name, agent_name, status, duration_seconds, step_blob, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(audit_id, step_name) DO UPDATE SET
                    agent_name = excluded.agent_name,
                    status = excluded.status,
                    duration_seconds = excluded.duration_seconds,
                    step_blob = excluded.step_blob,
                    created_at = excluded.created_at
                """,
                (audit_id, step_name, agent_name, status, duration_seconds, blob, now_iso),
            )
            connection.commit()


def get_step_checkpoint(
    audit_id: Optional[str],
    step_name: str,
) -> Optional[Dict[str, Any]]:
    """Retrieves cached step output dictionary, or None if not checkpointed."""
    if not audit_id:
        return None

    with _LOCK:
        with _connect() as connection:
            row = connection.execute(
                "SELECT step_blob FROM audit_steps WHERE audit_id = ? AND step_name = ? AND status = 'completed'",
                (audit_id, step_name),
            ).fetchone()

            # Artifact-matching fallback if direct audit_id step not found
            if not row:
                art_row = connection.execute(
                    "SELECT code_hash, model_hash, dataset_hash FROM audit_artifacts WHERE audit_id = ?",
                    (audit_id,),
                ).fetchone()
                if art_row and art_row[0]:
                    code_h, model_h, dataset_h = art_row[0], art_row[1] or "", art_row[2] or ""
                    query = """
                        SELECT s.step_blob, s.agent_name, s.duration_seconds
                        FROM audit_steps s
                        JOIN audit_artifacts a ON s.audit_id = a.audit_id
                        WHERE a.code_hash = ?
                          AND (? = '' OR a.model_hash = ? OR a.model_hash = '')
                          AND (? = '' OR a.dataset_hash = ? OR a.dataset_hash = '')
                          AND s.step_name = ?
                          AND s.status = 'completed'
                        ORDER BY s.created_at DESC
                        LIMIT 1
                    """
                    row = connection.execute(
                        query,
                        (code_h, model_h, model_h, dataset_h, dataset_h, step_name),
                    ).fetchone()
                    if row:
                        try:
                            connection.execute(
                                """
                                INSERT INTO audit_steps (
                                    audit_id, step_name, agent_name, status, duration_seconds, step_blob, created_at
                                )
                                VALUES (?, ?, ?, 'completed', ?, ?, ?)
                                ON CONFLICT(audit_id, step_name) DO UPDATE SET
                                    agent_name = excluded.agent_name,
                                    status = excluded.status,
                                    duration_seconds = excluded.duration_seconds,
                                    step_blob = excluded.step_blob
                                """,
                                (
                                    audit_id,
                                    step_name,
                                    row[1],
                                    row[2],
                                    row[0],
                                    datetime.now(timezone.utc).isoformat(),
                                ),
                            )
                            connection.commit()
                        except Exception:
                            pass

    if not row:
        return None

    try:
        data = pickle.loads(row[0])
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def has_step_completed(
    audit_id: Optional[str],
    step_name: str,
) -> bool:
    """Quick boolean query whether a step is already checkpointed."""
    if not audit_id:
        return False
    with _LOCK:
        with _connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM audit_steps WHERE audit_id = ? AND step_name = ? AND status = 'completed'",
                (audit_id, step_name),
            ).fetchone()
    return row is not None


# =====================================================================
# Sub-Test Checkpointing (V1, V2, V3, V4 Dynamic Penetration Attacks)
# =====================================================================

def save_subtest_checkpoint(
    audit_id: str,
    test_id: str,
    evidence: Dict[str, Any],
) -> None:
    """Persists an individual dynamic attack evidence dictionary."""
    if not audit_id or not test_id or not isinstance(evidence, dict):
        return

    status = str(evidence.get("status", "completed")).lower()
    # Never persist skipped, unverified, or error placeholders as valid dynamic subtests
    if status in ("skipped", "unverified", "skipped_zero_trust", "error"):
        return
    if str(evidence.get("evidence", {}).get("status", "")).lower() == "skipped":
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    severity = evidence.get("severity")
    blob = pickle.dumps(evidence, protocol=pickle.HIGHEST_PROTOCOL)

    with _LOCK:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_subtests (audit_id, test_id, status, severity, evidence_blob, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(audit_id, test_id) DO UPDATE SET
                    status = excluded.status,
                    severity = excluded.severity,
                    evidence_blob = excluded.evidence_blob,
                    created_at = excluded.created_at
                """,
                (audit_id, test_id, status, severity, blob, now_iso),
            )
            connection.commit()


def get_subtest_checkpoint(
    audit_id: Optional[str],
    test_id: str,
) -> Optional[Dict[str, Any]]:
    """Retrieves an individual dynamic test evidence dictionary."""
    if not audit_id:
        return None

    with _LOCK:
        with _connect() as connection:
            row = connection.execute(
                "SELECT evidence_blob FROM audit_subtests WHERE audit_id = ? AND test_id = ?",
                (audit_id, test_id),
            ).fetchone()

    if not row:
        return None

    try:
        data = pickle.loads(row[0])
        if isinstance(data, dict):
            st = str(data.get("status", "")).lower()
            ev_st = str(data.get("evidence", {}).get("status", "")).lower()
            if st not in ("skipped", "unverified", "skipped_zero_trust", "error") and ev_st != "skipped":
                return data
    except Exception:
        pass
    return None


def get_all_subtests(
    audit_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Retrieves all completed valid sub-tests for an audit."""
    if not audit_id:
        return {}

    with _LOCK:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT test_id, evidence_blob FROM audit_subtests WHERE audit_id = ?",
                (audit_id,),
            ).fetchall()

    result = {}
    for tid, blob in rows:
        try:
            ev = pickle.loads(blob)
            if isinstance(ev, dict):
                st = str(ev.get("status", "")).lower()
                ev_st = str(ev.get("evidence", {}).get("status", "")).lower()
                if st not in ("skipped", "unverified", "skipped_zero_trust", "error") and ev_st != "skipped":
                    result[tid] = ev
        except Exception:
            continue
    return result


def get_subtests_by_artifacts(
    code: Optional[str] = None,
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    audit_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Finds and aggregates all completed dynamic sub-tests across all previous
    audits that evaluated the exact same code, model, and dataset.
    """
    code_h = compute_code_hash(code)
    model_h = compute_file_hash(model_path)
    dataset_h = compute_file_hash(dataset_path)

    if not code_h and audit_id:
        with _LOCK:
            with _connect() as connection:
                row = connection.execute(
                    "SELECT code_hash, model_hash, dataset_hash FROM audit_artifacts WHERE audit_id = ?",
                    (audit_id,),
                ).fetchone()
                if row and row[0]:
                    code_h = row[0]
                    if not model_h:
                        model_h = row[1] or ""
                    if not dataset_h:
                        dataset_h = row[2] or ""

    if not code_h:
        return {}

    with _LOCK:
        with _connect() as connection:
            query = """
                SELECT DISTINCT a.audit_id
                FROM audit_artifacts a
                WHERE a.code_hash = ?
            """
            params: List[Any] = [code_h]
            if model_h:
                query += " AND (a.model_hash = ? OR a.model_hash = '')"
                params.append(model_h)
            if dataset_h:
                query += " AND (a.dataset_hash = ? OR a.dataset_hash = '')"
                params.append(dataset_h)

            matching_audits = [r[0] for r in connection.execute(query, tuple(params)).fetchall()]

    aggregated: Dict[str, Dict[str, Any]] = {}
    for aid in matching_audits:
        subtests = get_all_subtests(aid)
        for tid, ev in subtests.items():
            if tid not in aggregated:
                aggregated[tid] = ev
    return aggregated


def sync_subtests_for_audit(
    audit_id: str,
    code: Optional[str] = None,
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Ensures that any previously completed sub-tests for the given artifacts
    are copied into the current audit's subtests in memory.
    """
    if not audit_id:
        return {}
    existing = get_all_subtests(audit_id)
    artifact_subtests = get_subtests_by_artifacts(code, model_path, dataset_path, audit_id=audit_id)
    for tid, ev in artifact_subtests.items():
        if tid not in existing:
            save_subtest_checkpoint(audit_id, tid, ev)
            existing[tid] = ev
    return existing


def is_post_gate1_fully_cached(
    audit_id: str,
    planned_tests: Optional[List[str]] = None,
) -> bool:
    """
    Checks if all planned sub-tests and subsequent post-Gate 1 analysis
    (forensic diagnosis and report synthesis) are already completely cached
    and valid in audit memory.
    """
    if not audit_id:
        return False

    tests = planned_tests or ["V1_poisoning", "V2_preprocessing", "V3_validation", "V4_adversarial"]
    subtests = get_all_subtests(audit_id)
    for tid in tests:
        if tid not in subtests:
            return False

    forensic = get_step_checkpoint(audit_id, "forensic_diagnosis")
    if not forensic:
        return False

    report = get_step_checkpoint(audit_id, "synthesize_audit_report") or get_step_checkpoint(audit_id, "synthesize_report")
    if not report:
        return False

    return True


def invalidate_post_gate1_steps(
    audit_id: str,
) -> None:
    """
    Removes cached post-Gate 1 step checkpoints (execute_sandbox, forensic_diagnosis,
    aggregate_results, correlate_findings, synthesize_audit_report, validate_audit_report,
    finalize_report) so that new or merged dynamic test results can be re-correlated
    and synthesized into a fresh report without serving stale checkpoints.
    Preserves Agent 1 static analysis, Agent 2 strategy planning, and audit_subtests.
    """
    if not audit_id:
        return

    post_gate1_steps = (
        "execute_sandbox",
        "forensic_diagnosis",
        "aggregate_results",
        "correlate_findings",
        "synthesize_audit_report",
        "synthesize_report",
        "validate_audit_report",
        "finalize_report",
    )

    placeholders = ",".join("?" for _ in post_gate1_steps)
    with _LOCK:
        with _connect() as connection:
            connection.execute(
                f"DELETE FROM audit_steps WHERE audit_id = ? AND step_name IN ({placeholders})",
                (audit_id, *post_gate1_steps),
            )
            connection.commit()


# =====================================================================
# Audit Ledger & Inspection
# =====================================================================

def get_audit_ledger(audit_id: str) -> List[Dict[str, Any]]:
    """Returns chronological trace of all completed steps and subtests."""
    if not audit_id:
        return []

    with _LOCK:
        with _connect() as connection:
            rows = connection.execute(
                """
                SELECT step_name, agent_name, status, duration_seconds, created_at
                FROM audit_steps
                WHERE audit_id = ?
                ORDER BY created_at ASC
                """,
                (audit_id,),
            ).fetchall()

    return [
        {
            "step_name": r[0],
            "agent_name": r[1],
            "status": r[2],
            "duration_seconds": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def get_audit_summary(audit_id: str) -> Dict[str, Any]:
    """Returns high-level summary of audit checkpoints, artifacts, and execution metrics."""
    if not audit_id:
        return {
            "audit_id": "",
            "is_checkpoint_resumed": False,
            "steps_count": 0,
            "cached_steps": [],
            "completed_subtests": [],
            "artifacts": {},
            "total_duration": 0.0,
            "ledger": [],
        }

    with _LOCK:
        with _connect() as connection:
            art_row = connection.execute(
                "SELECT code_hash, model_hash, dataset_hash, registered_at FROM audit_artifacts WHERE audit_id = ?",
                (audit_id,),
            ).fetchone()

    artifacts = {}
    if art_row:
        artifacts = {
            "code_hash": art_row[0] or "",
            "model_hash": art_row[1] or "",
            "dataset_hash": art_row[2] or "",
            "registered_at": art_row[3] or "",
        }

    ledger = get_audit_ledger(audit_id)
    subtests = get_all_subtests(audit_id)
    total_dur = sum(s.get("duration_seconds") or 0.0 for s in ledger)

    return {
        "audit_id": audit_id,
        "is_checkpoint_resumed": len(ledger) > 0,
        "steps_count": len(ledger),
        "cached_steps": [s["step_name"] for s in ledger],
        "completed_subtests": list(subtests.keys()),
        "artifacts": artifacts,
        "total_duration": round(total_dur, 3),
        "ledger": ledger,
    }



# =====================================================================
# Session-Level Persistence (Backward Compatibility)
# =====================================================================

def save_session(
    audit_id: str,
    session: Dict[str, Any],
) -> None:
    """Persist the complete audit state for backward compatibility."""
    now_iso = datetime.now(timezone.utc).isoformat()
    session_blob = pickle.dumps(
        session,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    with _LOCK:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_sessions (
                    audit_id,
                    session_blob,
                    updated_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(audit_id)
                DO UPDATE SET
                    session_blob = excluded.session_blob,
                    updated_at = excluded.updated_at
                """,
                (
                    audit_id,
                    session_blob,
                    now_iso,
                ),
            )
            connection.commit()


def load_session(
    audit_id: str,
) -> Optional[Dict[str, Any]]:
    """Restore a previous audit session."""
    with _LOCK:
        with _connect() as connection:
            row = connection.execute(
                """
                SELECT session_blob
                FROM audit_sessions
                WHERE audit_id = ?
                """,
                (audit_id,),
            ).fetchone()

    if row is None:
        return None

    try:
        session = pickle.loads(row[0])
        if isinstance(session, dict):
            return session
    except Exception:
        pass
    return None


def delete_session(
    audit_id: str,
) -> None:
    """Deletes all session, artifact, step, and subtest data for an audit."""
    with _LOCK:
        with _connect() as connection:
            connection.execute("DELETE FROM audit_sessions WHERE audit_id = ?", (audit_id,))
            connection.execute("DELETE FROM audit_artifacts WHERE audit_id = ?", (audit_id,))
            connection.execute("DELETE FROM audit_steps WHERE audit_id = ?", (audit_id,))
            connection.execute("DELETE FROM audit_subtests WHERE audit_id = ?", (audit_id,))
            connection.commit()


initialize_memory()