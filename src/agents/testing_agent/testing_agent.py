from __future__ import annotations

import functools
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from src.core.llm import get_llm, is_llm_available
from .schemas import (
    TestingAgentState,
    AttackStrategyPlan,
    AdversarialAttackConfig,
    ForensicAnalysisReport,
    ForensicFinding,
    TEST_ORDER,
    MVP_VULNERABILITIES,
)
from .tools import (
    inspect_dataset_profile,
    calculate_perturbation_budget,
    make_cognitive_planning_tools,
    make_forensic_diagnostic_tools,
)
from src.core.audit_memory import (
    get_step_checkpoint,
    save_step_checkpoint,
    has_step_completed,
    get_subtest_checkpoint,
    save_subtest_checkpoint,
    get_all_subtests,
    verify_artifact_integrity,
)


# =====================================================================
# Telemetry Streaming Bus
# =====================================================================

_bus_lock = threading.Lock()
_streams: Dict[str, queue.Queue] = {}
DONE_EVENT = {"event": "done"}


def create_stream(audit_id: str) -> queue.Queue:
    with _bus_lock:
        q: queue.Queue = queue.Queue()
        _streams[audit_id] = q
        return q


def get_stream(audit_id: str) -> Optional[queue.Queue]:
    with _bus_lock:
        return _streams.get(audit_id)


def publish(audit_id: Optional[str], event: Dict[str, Any]) -> None:
    if not audit_id:
        return
    with _bus_lock:
        q = _streams.get(audit_id)
    if q is not None:
        q.put(event)


def close_stream(audit_id: Optional[str]) -> None:
    if not audit_id:
        return
    publish(audit_id, dict(DONE_EVENT))
    with _bus_lock:
        _streams.pop(audit_id, None)


# =====================================================================
# OpenTelemetry & LangSmith Tracing
# =====================================================================

_TRACER_PROVIDER_INITIALIZED = False


def _summarize_state_for_span(state: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if "dataset_path" in state:
        summary["aegisml.dataset_path"] = str(state.get("dataset_path"))
    if "model_path" in state:
        summary["aegisml.model_path"] = str(state.get("model_path"))
    if "planned_tests" in state and state.get("planned_tests"):
        summary["aegisml.planned_tests"] = ",".join(state.get("planned_tests") or [])
    if "sandbox_status" in state and state.get("sandbox_status"):
        summary["aegisml.sandbox_status"] = str(state.get("sandbox_status"))
    if "status" in state and state.get("status"):
        summary["aegisml.status"] = str(state.get("status"))
    return summary


def _init_tracer_provider() -> None:
    global _TRACER_PROVIDER_INITIALIZED
    if _TRACER_PROVIDER_INITIALIZED:
        return
    resource = Resource.create({"service.name": "aegisml-testing-agent"})
    provider = TracerProvider(resource=resource)
    exporter_kind = os.getenv("AEGISML_OTEL_EXPORTER", "console").lower()
    if exporter_kind == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            endpoint = os.getenv("AEGISML_OTEL_ENDPOINT", "localhost:4317")
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except ImportError:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER_INITIALIZED = True


def _langsmith_available() -> bool:
    return bool(
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and os.getenv("LANGCHAIN_API_KEY")
    )


def traced_node(node_name: str) -> Callable:
    _init_tracer_provider()
    tracer = trace.get_tracer("aegisml.testing_agent")

    def decorator(node_fn: Callable) -> Callable:
        @functools.wraps(node_fn)
        def otel_wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            start = time.time()
            with tracer.start_as_current_span(f"agent2.node.{node_name}") as span:
                span.set_attribute("aegisml.node_name", node_name)
                for key, value in _summarize_state_for_span(state).items():
                    span.set_attribute(key, value)
                try:
                    result = node_fn(state)
                    span.set_attribute(
                        "aegisml.duration_seconds", round(time.time() - start, 4)
                    )
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise

        if not _langsmith_available():
            return otel_wrapped

        from langsmith import traceable

        @functools.wraps(node_fn)
        @traceable(name=f"agent2.{node_name}", run_type="chain")
        def fully_wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            return otel_wrapped(state)

        return fully_wrapped

    return decorator


# =====================================================================
# LangGraph Workflow Nodes
# =====================================================================

def node_prepare_metadata(state: TestingAgentState, audit_id: Optional[str] = None) -> Dict[str, Any]:
    audit_id = audit_id or state.get("audit_id")
    if audit_id:
        cached = get_step_checkpoint(audit_id, "prepare_metadata")
        if cached is not None:
            return cached

    t0 = time.time()
    dataset_path = state.get("dataset_path", "")
    text_column = state.get("text_column")
    label_column = state.get("label_column")
    profile = inspect_dataset_profile.invoke({
        "dataset_path": dataset_path,
        "text_column": text_column,
        "label_column": label_column,
    })
    log = list(state.get("execution_plan_log") or [])
    log.append(f"Host-side safe metadata inspection completed for: {os.path.basename(dataset_path)}")
    res = {"dataset_profile": profile, "execution_plan_log": log}
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 2", "prepare_metadata", res, duration_seconds=round(time.time() - t0, 3))
    return res


def node_reason_strategy(state: TestingAgentState, audit_id: Optional[str] = None) -> Dict[str, Any]:
    audit_id = audit_id or state.get("audit_id")
    if audit_id:
        cached = get_step_checkpoint(audit_id, "reason_strategy")
        if cached is not None:
            publish(audit_id, {
                "event": "step_resumed_from_checkpoint",
                "agent": "Agent 2",
                "step": "Attack strategy formulation",
                "message": "Strategy plan loaded from checkpoint.",
            })
            return cached

    t0 = time.time()
    log = list(state.get("execution_plan_log") or [])
    agent_1 = state.get("agent_1_results") or {}
    dataset_profile = state.get("dataset_profile") or {}
    dataset_path = state.get("dataset_path", "data/dataset.csv")
    text_column = state.get("text_column", "text")
    label_column = state.get("label_column", "label")
    explicit_targets = state.get("test_targets")
    strict_mode = os.getenv("AEGISML_STRICT_AGENT", "false").lower() in ("true", "1")

    def get_fallback(reason: str) -> AttackStrategyPlan:
        if strict_mode:
            raise RuntimeError(f"AEGISML_STRICT_AGENT enforcement failure: {reason}")
        plan = _get_default_strategy_plan(explicit_targets, agent_1, dataset_profile)
        plan.strategy_provenance = "static_baseline"
        plan.planning_rationale = f"Static baseline applied: {reason}"
        return plan

    if not is_llm_available():
        fallback = get_fallback("LLM provider credentials unavailable.")
        log.append("LLM credentials unavailable. Applied deterministic attack strategy plan (provenance: static_baseline).")
        return {
            "attack_strategy_plan": fallback.model_dump(),
            "planned_tests": fallback.selected_tests,
            "execution_plan_log": log,
        }

    try:
        threat_model = agent_1.get("threat_model", {})
        vuln_findings = agent_1.get("vulnerability_findings", {}).get("vulnerabilities", [])

        system_msg = SystemMessage(
            content=(
                "You are the AegisML Lead Penetration Testing Strategist (Agent 2).\n"
                "You have cognitive planning tools to formulate an empirical testing campaign:\n\n"
                "- bound_inspect_dataset_profile: queries dataset row counts, class balances, and text statistics.\n"
                "- calculate_perturbation_budget: computes mathematically sound epsilon bounds, max iterations, and sample sizes for HopSkipJump evasion attacks.\n"
                "- bound_resolve_threat_surface: maps Agent 1's static findings to relevant dynamic tests.\n"
                "- bound_inspect_agent1_hypotheses: inspects specific vulnerability claims and affected components.\n\n"
                "Calibrate all attack parameters specifically for the target pipeline's feature representation. "
                "Call tools in whatever sequence needed to gather quantitative evidence. When you have enough information, stop calling tools."
            )
        )
        human_msg = HumanMessage(
            content=(
                f"Dataset path: {dataset_path}\n"
                f"Text column: {text_column}\n"
                f"Label column: {label_column}\n\n"
                f"Initial dataset profile:\n{dataset_profile}\n\n"
                f"Agent 1 Threat Model:\n{threat_model}\n\n"
                f"Agent 1 Vulnerability Findings Count: {len(vuln_findings)}\n\n"
                f"Explicit test targets override (if any): {explicit_targets}\n\n"
                "Use your planning tools to gather quantitative calibration data, then synthesize the strategy plan."
            )
        )

        planning_tools = make_cognitive_planning_tools(
            agent_1_results=agent_1,
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column,
        )
        tool_map: Dict[str, Any] = {t.name: t for t in planning_tools}
        llm_with_tools = get_llm(temperature=0.0).bind_tools(planning_tools)

        messages: List[Any] = [system_msg, human_msg]
        MAX_TOOL_ROUNDS = 3
        for round_idx in range(MAX_TOOL_ROUNDS):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                log.append(f"Cognitive strategist completed tool investigation after {round_idx} round(s). Synthesizing AttackStrategyPlan.")
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                if tool_name in tool_map:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                    log.append(f"Cognitive strategist called tool '{tool_name}' with args: {tool_args}.")
                else:
                    tool_result = {"error": f"Unknown tool requested: {tool_name}"}
                    log.append(f"Cognitive strategist attempted unknown tool '{tool_name}' - rejected.")
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

        messages.append(
            HumanMessage(
                content=(
                    "You have collected all necessary quantitative data. Produce the final AttackStrategyPlan as a structured object.\n\n"
                    "Constraints:\n"
                    "- selected_tests must only contain valid IDs from: [V1_poisoning, V4_adversarial, V2_preprocessing, V3_validation].\n"
                    "- Calibrate adversarial_config with your computed perturbation budget and max_iter.\n"
                    "- Calibrate poisoning_config poison_fractions according to class balance.\n"
                    "- Include a thorough planning_rationale citing the tool observations that informed your decisions."
                )
            )
        )

        structured_llm = get_llm(temperature=0.0).with_structured_output(AttackStrategyPlan)
        try:
            strategy_plan: AttackStrategyPlan = structured_llm.invoke(messages)
        except Exception as schema_err:
            log.append(f"Strategy extraction encountered error ({str(schema_err)}). Attempting self-repair reflection.")
            messages.append(
                HumanMessage(
                    content=f"Your previous output failed schema validation with error: {str(schema_err)}. Please regenerate the AttackStrategyPlan strictly matching schema types."
                )
            )
            strategy_plan = structured_llm.invoke(messages)

        if isinstance(strategy_plan, dict):
            strategy_plan = AttackStrategyPlan.model_validate(strategy_plan)

        strategy_plan.strategy_provenance = "autonomous_cognitive"
        not_applicable_tests = {
            f"{str(finding.get('vulnerability_id', '')).upper()}_"
            + {
                "V1": "poisoning",
                "V2": "preprocessing",
                "V3": "validation",
                "V4": "adversarial",
            }.get(str(finding.get("vulnerability_id", "")).upper(), "")
            for finding in vuln_findings
            if str(finding.get("status", "")).lower() == "not_applicable"
        }
        ordered_tests = [t for t in TEST_ORDER if t in strategy_plan.selected_tests]
        strategy_plan.selected_tests = [
            test for test in ordered_tests if test not in not_applicable_tests
        ]
        if not strategy_plan.selected_tests:
            strategy_plan.selected_tests = [
                test for test in TEST_ORDER if test not in not_applicable_tests
            ]

        log.append(f"Cognitive strategy formulated (provenance: {strategy_plan.strategy_provenance}). Planned tests: {strategy_plan.selected_tests}")
        log.append(f"Strategy rationale: {strategy_plan.planning_rationale}")

        res = {
            "attack_strategy_plan": strategy_plan.model_dump(),
            "planned_tests": strategy_plan.selected_tests,
            "execution_plan_log": log,
        }
        if audit_id:
            save_step_checkpoint(audit_id, "Agent 2", "reason_strategy", res, duration_seconds=round(time.time() - t0, 3))
        return res

    except Exception as e:
        fallback = get_fallback(f"Reasoning loop encountered exception: {str(e)}")
        log.append(f"LLM strategy formulation error ({str(e)}). Applied baseline strategy (provenance: {fallback.strategy_provenance}).")
        res = {
            "attack_strategy_plan": fallback.model_dump(),
            "planned_tests": fallback.selected_tests,
            "execution_plan_log": log,
        }
        if audit_id:
            save_step_checkpoint(audit_id, "Agent 2", "reason_strategy", res, duration_seconds=round(time.time() - t0, 3))
        return res


def node_execute_sandbox(state: TestingAgentState) -> Dict[str, Any]:
    from .sandbox_runner import dispatch_sandbox
    from src.core.audit_memory import (
        get_all_subtests,
        save_subtest_checkpoint,
        sync_subtests_for_audit,
        invalidate_post_gate1_steps,
    )

    log = list(state.get("execution_plan_log") or [])
    planned_tests = state.get("planned_tests", TEST_ORDER)
    strategy_config = state.get("attack_strategy_plan", {})
    audit_id = state.get("audit_id")
    t0 = time.time()

    # Sync and retrieve all previously completed dynamic sub-tests for this audit / pipeline
    cached_subtests: Dict[str, Dict[str, Any]] = {}
    if audit_id:
        try:
            synced = sync_subtests_for_audit(
                audit_id,
                code=state.get("pipeline_source"),
                model_path=state.get("model_path"),
                dataset_path=state.get("dataset_path"),
            )
            cached_subtests = dict(synced)
        except Exception:
            cached_subtests = get_all_subtests(audit_id)
    
    # Filter to only genuine completed dynamic tests (never skipped/unverified)
    valid_cached = {}
    for tid, ev in cached_subtests.items():
        if isinstance(ev, dict):
            st = str(ev.get("status", "")).lower()
            ev_st = str(ev.get("evidence", {}).get("status", "")).lower()
            if st not in ("skipped", "unverified", "skipped_zero_trust", "error") and ev_st != "skipped":
                valid_cached[tid] = ev
    cached_subtests = valid_cached

    subtest_key_map = {
        "V1_poisoning": "poisoning_evidence",
        "V4_adversarial": "adversarial_evidence",
        "V2_preprocessing": "preprocessing_evidence",
        "V3_validation": "validation_evidence",
    }

    # Identify tests needing execution
    needed_tests = [t for t in planned_tests if t not in cached_subtests]

    if audit_id and not needed_tests:
        log.append(f"All planned dynamic tests ({planned_tests}) restored from sub-test checkpoints. Skipping sandbox re-execution.")
        publish(audit_id, {
            "event": "agent_step_finished",
            "agent": "Agent 2",
            "step": "Dynamic security testing",
            "status": "executed",
            "message": f"Dynamic tests restored from memory: {', '.join(planned_tests)}",
        })
        res = {
            "sandbox_status": "executed",
            "sandbox_telemetry": {"duration_seconds": 0.0, "cached": True},
            "poisoning_evidence": cached_subtests.get("V1_poisoning", {}),
            "adversarial_evidence": cached_subtests.get("V4_adversarial", {}),
            "preprocessing_evidence": cached_subtests.get("V2_preprocessing", {}),
            "validation_evidence": cached_subtests.get("V3_validation", {}),
            "execution_plan_log": log,
        }
        if not has_step_completed(audit_id, "execute_sandbox"):
            save_step_checkpoint(audit_id, "Agent 2", "execute_sandbox", res, duration_seconds=0.0)
        return res

    publish(audit_id, {
        "event": "agent_step_started",
        "agent": "Agent 2",
        "step": "Dynamic security testing",
        "message": f"Running approved tests: {', '.join(needed_tests)}" + (f" (retained {len(cached_subtests)} from memory)" if cached_subtests else ""),
    })
    log.append(f"Dispatching dynamic tests to sandbox orchestrator: {needed_tests}" + (f" (retaining {list(cached_subtests.keys())} from memory)" if cached_subtests else ""))

    resolved_text_col = (
        state.get("text_column")
        or (state.get("dataset_profile") or {}).get("text_column")
        or "text"
    )
    resolved_label_col = (
        state.get("label_column")
        or (state.get("dataset_profile") or {}).get("label_column")
        or "label"
    )

    sandbox_result = dispatch_sandbox(
        model_path=state.get("model_path", "data/model.pkl"),
        dataset_path=state.get("dataset_path", "data/dataset.csv"),
        pipeline_path=state.get("pipeline_path") or "data/pipeline.py",
        vectorizer_path=state.get("vectorizer_path"),
        text_column=resolved_text_col,
        label_column=resolved_label_col,
        planned_tests=needed_tests,
        strategy_config=strategy_config,
        agent_1_results=state.get("agent_1_results"),
        audit_id=audit_id,
    )

    log.extend(sandbox_result.get("execution_log", []))

    # Persist newly executed subtests
    if audit_id and sandbox_result.get("sandbox_status") == "executed":
        for test_id, ev_key in subtest_key_map.items():
            ev_data = sandbox_result.get(ev_key)
            if test_id in needed_tests and ev_data:
                save_subtest_checkpoint(audit_id, test_id, ev_data)

    if audit_id:
        invalidate_post_gate1_steps(audit_id)

    # Merge cached and freshly executed evidence
    final_poisoning = sandbox_result.get("poisoning_evidence") or cached_subtests.get("V1_poisoning", {})
    final_adversarial = sandbox_result.get("adversarial_evidence") or cached_subtests.get("V4_adversarial", {})
    final_preprocessing = sandbox_result.get("preprocessing_evidence") or cached_subtests.get("V2_preprocessing", {})
    final_validation = sandbox_result.get("validation_evidence") or cached_subtests.get("V3_validation", {})

    publish(audit_id, {
        "event": "agent_step_finished",
        "agent": "Agent 2",
        "step": "Dynamic security testing",
        "status": sandbox_result.get("sandbox_status", "unknown"),
        "message": "Dynamic tests completed." if sandbox_result.get("sandbox_status") == "executed" else "Dynamic tests finished with a guarded status.",
    })
    res = {
        "sandbox_status": sandbox_result.get("sandbox_status", "unknown"),
        "sandbox_telemetry": sandbox_result.get("telemetry", {}),
        "poisoning_evidence": final_poisoning,
        "adversarial_evidence": final_adversarial,
        "preprocessing_evidence": final_preprocessing,
        "validation_evidence": final_validation,
        "execution_plan_log": log,
    }
    if audit_id and sandbox_result.get("sandbox_status") == "executed":
        save_step_checkpoint(audit_id, "Agent 2", "execute_sandbox", res, duration_seconds=round(time.time() - t0, 3))
    return res


def node_forensic_diagnosis(state: TestingAgentState) -> Dict[str, Any]:
    audit_id = state.get("audit_id")
    if audit_id:
        cached = get_step_checkpoint(audit_id, "forensic_diagnosis")
        if cached is not None:
            publish(audit_id, {
                "event": "step_resumed_from_checkpoint",
                "agent": "Agent 2",
                "step": "Forensic diagnosis",
                "message": "Forensic diagnosis loaded from checkpoint.",
            })
            return cached

    t0 = time.time()
    log = list(state.get("execution_plan_log") or [])
    sandbox_status = state.get("sandbox_status", "")
    agent_1 = state.get("agent_1_results") or {}
    publish(state.get("audit_id"), {
        "event": "agent_step_started",
        "agent": "Agent 2",
        "step": "Forensic diagnosis",
        "message": "Correlating empirical evidence with Agent 1 hypotheses.",
    })

    if sandbox_status == "skipped_zero_trust":
        log.append("Dynamic tests skipped under Zero-Trust policy. Forensic analysis deferred.")
        result = {
            "forensic_analysis": {
                "overall_forensic_summary": (
                    "Empirical penetration testing was halted on the host because the Docker sandbox was offline. "
                    "Zero-Trust policy prevented untrusted artifact execution. All findings remain based on Agent 1's static threat model."
                ),
                "findings": [],
            },
            "execution_plan_log": log,
        }
        publish(state.get("audit_id"), {
            "event": "agent_step_finished",
            "agent": "Agent 2",
            "step": "Forensic diagnosis",
            "status": "skipped_zero_trust",
            "message": "Forensic review recorded the guarded skip.",
        })
        return result

    evidence_bundle = {
        "V1_poisoning": state.get("poisoning_evidence"),
        "V4_adversarial": state.get("adversarial_evidence"),
        "V2_preprocessing": state.get("preprocessing_evidence"),
        "V3_validation": state.get("validation_evidence"),
    }

    if not is_llm_available():
        fallback_forensics = _get_default_forensic_report(evidence_bundle)
        result = {
            "forensic_analysis": fallback_forensics.model_dump(),
            "execution_plan_log": log,
        }
        publish(state.get("audit_id"), {
            "event": "agent_step_finished",
            "agent": "Agent 2",
            "step": "Forensic diagnosis",
            "status": "completed",
            "message": "Deterministic forensic diagnosis completed.",
        })
        return result

    try:
        diag_tools = make_forensic_diagnostic_tools(
            evidence_bundle=evidence_bundle,
            agent_1_results=agent_1,
        )
        tool_map = {t.name: t for t in diag_tools}
        llm_with_tools = get_llm(temperature=0.0).bind_tools(diag_tools)

        system_prompt = (
            "You are the AegisML Lead ML Forensic Security Diagnostician (Agent 2).\n"
            "You have diagnostic tools to investigate empirical container telemetry:\n"
            "- bound_query_attack_telemetry(test_id): retrieves detailed measurements and degradation curves.\n"
            "- bound_evaluate_hypothesis_correlation(vulnerability_id): mathematically correlates Agent 1's claim with empirical telemetry to determine 'Confirmed Risk', 'False Positive (Mitigated)', 'Hidden Risk', or 'Unverified'.\n\n"
            "Use these tools to investigate anomalous telemetry, then produce your final ForensicAnalysisReport."
        )
        user_prompt = (
            f"Empirical Test Telemetry Summary:\n{evidence_bundle}\n\n"
            f"Agent 1 Static Findings:\n{agent_1.get('vulnerability_findings')}\n\n"
            "Investigate the empirical telemetry using your diagnostic tools, then formulate the forensic report."
        )

        messages: List[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        for _ in range(2):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                t_name = tc["name"]
                t_args = tc.get("args", {})
                t_id = tc["id"]
                if t_name in tool_map:
                    t_result = tool_map[t_name].invoke(t_args)
                    log.append(f"Forensic diagnostician queried '{t_name}' with args {t_args}.")
                else:
                    t_result = {"error": f"Unknown tool: {t_name}"}
                messages.append(ToolMessage(content=str(t_result), tool_call_id=t_id))

        messages.append(
            HumanMessage(
                content=(
                    "Based on your empirical investigation and tool findings, produce the final ForensicAnalysisReport as a structured object. "
                    "Document mathematical root causes and explicit hypothesis confirmations for each tested vulnerability."
                )
            )
        )

        structured_llm = get_llm(temperature=0.1).with_structured_output(ForensicAnalysisReport)
        report = structured_llm.invoke(messages)
        if isinstance(report, dict):
            report = ForensicAnalysisReport.model_validate(report)

        log.append("Cognitive forensic diagnosis completed successfully via tool-empowered investigation.")
        publish(state.get("audit_id"), {
            "event": "agent_step_finished",
            "agent": "Agent 2",
            "step": "Forensic diagnosis",
            "status": "completed",
            "message": "Forensic diagnosis completed.",
        })
        res = {"forensic_analysis": report.model_dump(), "execution_plan_log": log}
        if audit_id:
            save_step_checkpoint(audit_id, "Agent 2", "forensic_diagnosis", res, duration_seconds=round(time.time() - t0, 3))
        return res

    except Exception as e:
        log.append(f"LLM forensic diagnosis encountered error ({str(e)}). Used deterministic fallback.")
        fallback = _get_default_forensic_report(evidence_bundle)
        publish(state.get("audit_id"), {
            "event": "agent_step_finished",
            "agent": "Agent 2",
            "step": "Forensic diagnosis",
            "status": "fallback",
            "message": "Forensic diagnosis used a deterministic fallback.",
        })
        res = {"forensic_analysis": fallback.model_dump(), "execution_plan_log": log}
        if audit_id:
            save_step_checkpoint(audit_id, "Agent 2", "forensic_diagnosis", res, duration_seconds=round(time.time() - t0, 3))
        return res


def node_aggregate_results(state: TestingAgentState) -> Dict[str, Any]:
    audit_id = state.get("audit_id")
    if audit_id:
        cached = get_step_checkpoint(audit_id, "aggregate_results")
        if cached is not None:
            return cached

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []
    agent_1 = state.get("agent_1_results")

    p_ev = state.get("poisoning_evidence")
    if p_ev:
        results.append(p_ev)
        if agent_1:
            drop = (
                p_ev.get("evidence", {}).get("accuracy_drop")
                or p_ev.get("evidence", {}).get("generic_test", {}).get("max_accuracy_drop")
            )
            verifications.append(
                _verification_from_status("V1", "Data Poisoning", p_ev, {"accuracy_drop": drop})
            )

    prep_ev = state.get("preprocessing_evidence")
    if prep_ev:
        results.append(prep_ev)
        if agent_1:
            verifications.append(
                _verification_from_status("V2", "Preprocessing Attack Surface", prep_ev, prep_ev.get("evidence", {}))
            )

    val_ev = state.get("validation_evidence")
    if val_ev:
        results.append(val_ev)
        if agent_1:
            verifications.append(
                _verification_from_status(
                    "V3",
                    "Data Validation Weaknesses",
                    val_ev,
                    {"accuracy_drop": val_ev.get("evidence", {}).get("accuracy_drop")},
                )
            )

    adv_ev = state.get("adversarial_evidence")
    if adv_ev:
        results.append(adv_ev)
        if agent_1:
            asr = adv_ev.get("evidence", {}).get("attack_success_rate_within_budget")
            verifications.append(
                _verification_from_status("V4", "Adversarial Robustness", adv_ev, {"attack_success_rate": asr})
            )

    present_ids = {r.get("vulnerability_id") for r in results}
    for vid, vname in MVP_VULNERABILITIES:
        if vid not in present_ids:
            results.append({
                "vulnerability_id": vid,
                "vulnerability_name": vname,
                "status": "not_tested",
                "severity": None,
                "evidence": {"reason": "Test was not scheduled in strategy."},
            })

    result_order = {t.split("_")[0]: i for i, t in enumerate(TEST_ORDER)}
    results.sort(key=lambda r: result_order.get(r.get("vulnerability_id", ""), 99))

    res = {
        "structured_test_results": {
            "results": results,
            "hypothesis_verifications": verifications,
            "attack_strategy_plan": state.get("attack_strategy_plan"),
            "forensic_analysis": state.get("forensic_analysis"),
            "sandbox_status": state.get("sandbox_status"),
            "sandbox_telemetry": state.get("sandbox_telemetry"),
            "execution_log": state.get("execution_plan_log") or [],
        },
        "hypothesis_verifications": verifications,
        "status": "completed",
    }
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 2", "aggregate_results", res, duration_seconds=round(time.time() - t0, 3))
    return res


def _verification_from_status(
    vulnerability_id: str,
    vulnerability_name: str,
    evidence_dict: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    test_status = evidence_dict.get("status", "unverified")
    dynamic_severity = evidence_dict.get("severity")
    if test_status == "vulnerable":
        correlation = "Confirmed Risk"
        rationale = f"Empirical testing confirmed susceptibility ({dynamic_severity or 'detected'})."
    elif test_status == "not_vulnerable":
        correlation = "False Positive (Mitigated)"
        rationale = "Empirical tests showed the model resisted attack conditions."
    elif test_status == "not_applicable":
        correlation = "Not Applicable"
        rationale = "Test was not applicable to the target pipeline structure."
    else:
        correlation = "Unverified"
        rationale = "Empirical testing was inconclusive or skipped under Zero-Trust policy."
    return {
        "vulnerability_id": vulnerability_id,
        "category": vulnerability_name,
        "correlation_status": correlation,
        "test_status": test_status,
        "dynamic_severity": dynamic_severity,
        "correlation_rationale": rationale,
        "metrics": metrics,
    }


def _get_default_strategy_plan(
    explicit_targets: Optional[List[str]],
    agent_1_results: Dict[str, Any],
    dataset_profile: Dict[str, Any],
) -> AttackStrategyPlan:
    avg_words = dataset_profile.get("text_stats", {}).get("avg_word_count", 100)
    budget_info = calculate_perturbation_budget.invoke({"avg_word_count": avg_words, "high_sparsity": True})
    adv_cfg = AdversarialAttackConfig(
        sample_size=budget_info.get("recommended_sample_size", 50),
        max_relative_perturbation_budget=budget_info.get("recommended_max_relative_budget", 0.4),
        max_iter=budget_info.get("recommended_max_iter", 45),
        rationale=budget_info.get("rationale", "Standard budget for text classification."),
    )
    planned = list(TEST_ORDER)
    if explicit_targets:
        planned = [t for t in TEST_ORDER if any(x.lower() in t.lower() for x in explicit_targets)]
        if not planned:
            planned = list(TEST_ORDER)
    return AttackStrategyPlan(
        selected_tests=planned,
        adversarial_config=adv_cfg,
        planning_rationale="Deterministic rule-based baseline strategy.",
        strategy_provenance="static_baseline",
    )


def _get_default_forensic_report(evidence_bundle: Dict[str, Any]) -> ForensicAnalysisReport:
    findings = []
    for test_key, ev in evidence_bundle.items():
        if not ev:
            continue
        status = ev.get("status", "unverified")
        vid = ev.get("vulnerability_id", test_key[:2])
        vname = ev.get("vulnerability_name", test_key)
        findings.append(
            ForensicFinding(
                vulnerability_id=vid,
                category=vname,
                hypothesis_confirmation=(
                    "Confirmed Risk"
                    if status == "vulnerable"
                    else "False Positive (Mitigated)"
                    if status == "not_vulnerable"
                    else "Unverified"
                ),
                root_cause_diagnosis=ev.get("summary", "Automated empirical metric diagnosis."),
                empirical_metric_summary=str(ev.get("evidence", {})),
                recommended_focus_area="Remediation guidance deferred to Agent 3.",
            )
        )
    return ForensicAnalysisReport(
        findings=findings,
        overall_forensic_summary="Empirical test results evaluated across dynamic testing modules.",
    )


def build_testing_agent_graph():
    graph = StateGraph(TestingAgentState)
    graph.add_node("prepare_metadata", traced_node("prepare_metadata")(node_prepare_metadata))
    graph.add_node("reason_strategy", traced_node("reason_strategy")(node_reason_strategy))
    graph.add_node("execute_sandbox", traced_node("execute_sandbox")(node_execute_sandbox))
    graph.add_node("forensic_diagnosis", traced_node("forensic_diagnosis")(node_forensic_diagnosis))
    graph.add_node("aggregate_results", traced_node("aggregate_results")(node_aggregate_results))

    graph.set_entry_point("prepare_metadata")
    graph.add_edge("prepare_metadata", "reason_strategy")
    graph.add_edge("reason_strategy", "execute_sandbox")
    graph.add_edge("execute_sandbox", "forensic_diagnosis")
    graph.add_edge("forensic_diagnosis", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


# =====================================================================
# Public Execution Entrypoint
# =====================================================================

def run_testing_agent(
    agent_1_results: Optional[Dict[str, Any]] = None,
    model_path: str = "data/model.pkl",
    dataset_path: str = "data/dataset.csv",
    text_column: str = "text",
    label_column: str = "label",
    vectorizer_path: Optional[str] = None,
    test_targets: Optional[List[str]] = None,
    pipeline_path: Optional[str] = None,
    audit_id: Optional[str] = None,
    force_resume: bool = False,
) -> Dict[str, Any]:
    """
    Executes Agent 2 (Vulnerability Testing Agent).

    Ingests Agent 1 findings (pipeline graph, threat model, and vulnerability findings),
    autonomously plans and dispatches empirical security tests, and cross-verifies
    empirical measurements against upstream hypotheses.
    """
    if audit_id:
        verify_artifact_integrity(
            audit_id,
            model_path=model_path,
            dataset_path=dataset_path,
            force=force_resume,
        )

    app = build_testing_agent_graph()

    initial_state: TestingAgentState = {
        "agent_1_results": agent_1_results,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "text_column": text_column,
        "label_column": label_column,
        "vectorizer_path": vectorizer_path,
        "test_targets": test_targets,
        "pipeline_path": pipeline_path,
        "audit_id": audit_id,
        "status": "initialized",
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    result = run_testing_agent(
        model_path="data/model.pkl",
        dataset_path="data/dataset.csv",
        text_column="text",
        label_column="label",
    )
    print(result.get("structured_test_results"))
