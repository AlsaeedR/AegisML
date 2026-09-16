from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = PROJECT_ROOT / ".aegisml_runtime"
DATABASE_PATH = MEMORY_DIR / "audit_memory.db"

_LOCK = Lock()


def _connect() -> sqlite3.Connection:
    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
        check_same_thread=False,
    )

    return connection


def initialize_memory() -> None:
    with _LOCK:
        with _connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_sessions (
                    audit_id TEXT PRIMARY KEY,
                    session_blob BLOB NOT NULL
                )
                """
            )

            connection.commit()


def save_session(
    audit_id: str,
    session: Dict[str, Any],
) -> None:
    """
    Persist the complete audit state.

    Pickle is used because the internal LangGraph / NetworkX state
    is not guaranteed to be JSON serializable.
    """

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
                    session_blob
                )
                VALUES (?, ?)
                ON CONFLICT(audit_id)
                DO UPDATE SET
                    session_blob = excluded.session_blob
                """,
                (
                    audit_id,
                    session_blob,
                ),
            )

            connection.commit()


def load_session(
    audit_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Restore a previous audit session.
    """

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

    session = pickle.loads(
        row[0]
    )

    if not isinstance(
        session,
        dict,
    ):
        return None

    return session


def delete_session(
    audit_id: str,
) -> None:
    with _LOCK:
        with _connect() as connection:
            connection.execute(
                """
                DELETE FROM audit_sessions
                WHERE audit_id = ?
                """,
                (audit_id,),
            )

            connection.commit()


initialize_memory()