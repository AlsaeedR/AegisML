import os
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from src.core.llm import get_llm, is_llm_available
from .state import TestingAgentState
from .schemas import AttackStrategyPlan, AdversarialAttackConfig, ForensicAnalysisReport, ForensicFinding
from .tools import inspect_dataset_profile, calculate_perturbation_budget, make_cognitive_planning_tools, make_forensic_diagnostic_tools
from .sandbox_runner import dispatch_sandbox
from .constants import TEST_ORDER, MVP_VULNERABILITIES
from .tracing import traced_node

def node_prepare_metadata(state: TestingAgentState) -> Dict[str, Any]:
    dataset_path = state.get('dataset_path', '')
    text_column = state.get('text_column')
    label_column = state.get('label_column')
    profile = inspect_dataset_profile.invoke({'dataset_path': dataset_path, 'text_column': text_column, 'label_column': label_column})
    log = list(state.get('execution_plan_log') or [])
    log.append(f'Host-side safe metadata inspection completed for: {os.path.basename(dataset_path)}')
    return {'dataset_profile': profile, 'execution_plan_log': log}

def node_reason_strategy(state: TestingAgentState) -> Dict[str, Any]:
    log = list(state.get('execution_plan_log') or [])
    agent_1 = state.get('agent_1_results') or {}
    dataset_profile = state.get('dataset_profile') or {}
    dataset_path = state.get('dataset_path', 'data/dataset.csv')
    text_column = state.get('text_column', 'text')
    label_column = state.get('label_column', 'label')
    explicit_targets = state.get('test_targets')
    strict_mode = os.getenv('AEGISML_STRICT_AGENT', 'false').lower() in ('true', '1')

    def get_fallback(reason: str) -> AttackStrategyPlan:
        if strict_mode:
            raise RuntimeError(f'AEGISML_STRICT_AGENT enforcement failure: {reason}')
        plan = _get_default_strategy_plan(explicit_targets, agent_1, dataset_profile)
        plan.strategy_provenance = 'static_baseline'
        plan.planning_rationale = f'Static baseline applied: {reason}'
        return plan
    if not is_llm_available():
        fallback = get_fallback('LLM provider credentials unavailable.')
        log.append('LLM credentials unavailable. Applied deterministic attack strategy plan (provenance: static_baseline).')
        return {'attack_strategy_plan': fallback.model_dump(), 'planned_tests': fallback.selected_tests, 'execution_plan_log': log}
    try:
        threat_model = agent_1.get('threat_model', {})
        vuln_findings = agent_1.get('vulnerability_findings', {}).get('vulnerabilities', [])
        system_msg = SystemMessage(content="You are the AegisML Lead Penetration Testing Strategist (Agent 2).\nYou have cognitive planning tools to formulate an empirical testing campaign:\n\n- bound_inspect_dataset_profile: queries dataset row counts, class balances, and text statistics.\n- calculate_perturbation_budget: computes mathematically sound epsilon bounds, max iterations, and sample sizes for HopSkipJump evasion attacks.\n- bound_resolve_threat_surface: maps Agent 1's static findings to relevant dynamic tests.\n- bound_inspect_agent1_hypotheses: inspects specific vulnerability claims and affected components.\n\nCalibrate all attack parameters specifically for the target pipeline's feature representation. Call tools in whatever sequence needed to gather quantitative evidence. When you have enough information, stop calling tools.")
        human_msg = HumanMessage(content=f'Dataset path: {dataset_path}\nText column: {text_column}\nLabel column: {label_column}\n\nInitial dataset profile:\n{dataset_profile}\n\nAgent 1 Threat Model:\n{threat_model}\n\nAgent 1 Vulnerability Findings Count: {len(vuln_findings)}\n\nExplicit test targets override (if any): {explicit_targets}\n\nUse your planning tools to gather quantitative calibration data, then synthesize the strategy plan.')
        planning_tools = make_cognitive_planning_tools(agent_1_results=agent_1, dataset_path=dataset_path, text_column=text_column, label_column=label_column)
        tool_map: Dict[str, Any] = {t.name: t for t in planning_tools}
        llm_with_tools = get_llm(temperature=0.0).bind_tools(planning_tools)
        messages: List[Any] = [system_msg, human_msg]
        MAX_TOOL_ROUNDS = 3
        for round_idx in range(MAX_TOOL_ROUNDS):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                log.append(f'Cognitive strategist completed tool investigation after {round_idx} round(s). Synthesizing AttackStrategyPlan.')
                break
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                if tool_name in tool_map:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                    log.append(f"Cognitive strategist called tool '{tool_name}' with args: {tool_args}.")
                else:
                    tool_result = {'error': f'Unknown tool requested: {tool_name}'}
                    log.append(f"Cognitive strategist attempted unknown tool '{tool_name}' — rejected.")
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
        messages.append(HumanMessage(content='You have collected all necessary quantitative data. Produce the final AttackStrategyPlan as a structured object.\n\nConstraints:\n- selected_tests must only contain valid IDs from: [V1_poisoning, V4_adversarial, V2_preprocessing, V3_validation].\n- Calibrate adversarial_config with your computed perturbation budget and max_iter.\n- Calibrate poisoning_config poison_fractions according to class balance.\n- Include a thorough planning_rationale citing the tool observations that informed your decisions.'))
        structured_llm = get_llm(temperature=0.0).with_structured_output(AttackStrategyPlan)
        try:
            strategy_plan: AttackStrategyPlan = structured_llm.invoke(messages)
        except Exception as schema_err:
            log.append(f'Strategy extraction encountered error ({str(schema_err)}). Attempting self-repair reflection.')
            messages.append(HumanMessage(content=f'Your previous output failed schema validation with error: {str(schema_err)}. Please regenerate the AttackStrategyPlan strictly matching schema types.'))
            strategy_plan = structured_llm.invoke(messages)
        if isinstance(strategy_plan, dict):
            strategy_plan = AttackStrategyPlan.model_validate(strategy_plan)
        strategy_plan.strategy_provenance = 'autonomous_cognitive'
        ordered_tests = [t for t in TEST_ORDER if t in strategy_plan.selected_tests]
        strategy_plan.selected_tests = ordered_tests or list(TEST_ORDER)
        log.append(f'Cognitive strategy formulated (provenance: {strategy_plan.strategy_provenance}). Planned tests: {strategy_plan.selected_tests}')
        log.append(f'Strategy rationale: {strategy_plan.planning_rationale}')
        return {'attack_strategy_plan': strategy_plan.model_dump(), 'planned_tests': strategy_plan.selected_tests, 'execution_plan_log': log}
    except Exception as e:
        fallback = get_fallback(f'Reasoning loop encountered exception: {str(e)}')
        log.append(f'LLM strategy formulation error ({str(e)}). Applied baseline strategy (provenance: {fallback.strategy_provenance}).')
        return {'attack_strategy_plan': fallback.model_dump(), 'planned_tests': fallback.selected_tests, 'execution_plan_log': log}

def node_execute_sandbox(state: TestingAgentState) -> Dict[str, Any]:
    log = list(state.get('execution_plan_log') or [])
    planned_tests = state.get('planned_tests', TEST_ORDER)
    strategy_config = state.get('attack_strategy_plan', {})
    log.append(f'Dispatching dynamic tests to sandbox orchestrator: {planned_tests}')
    resolved_text_col = state.get('text_column') or (state.get('dataset_profile') or {}).get('text_column') or 'text'
    resolved_label_col = state.get('label_column') or (state.get('dataset_profile') or {}).get('label_column') or 'label'
    sandbox_result = dispatch_sandbox(model_path=state.get('model_path', 'data/model.pkl'), dataset_path=state.get('dataset_path', 'data/dataset.csv'), pipeline_path=state.get('pipeline_path') or 'data/pipeline.py', vectorizer_path=state.get('vectorizer_path'), text_column=resolved_text_col, label_column=resolved_label_col, planned_tests=planned_tests, strategy_config=strategy_config, agent_1_results=state.get('agent_1_results'), audit_id=state.get('audit_id'))
    log.extend(sandbox_result.get('execution_log', []))
    return {'sandbox_status': sandbox_result.get('sandbox_status', 'unknown'), 'sandbox_telemetry': sandbox_result.get('telemetry', {}), 'poisoning_evidence': sandbox_result.get('poisoning_evidence', {}), 'adversarial_evidence': sandbox_result.get('adversarial_evidence', {}), 'preprocessing_evidence': sandbox_result.get('preprocessing_evidence', {}), 'validation_evidence': sandbox_result.get('validation_evidence', {}), 'execution_plan_log': log}

def node_forensic_diagnosis(state: TestingAgentState) -> Dict[str, Any]:
    log = list(state.get('execution_plan_log') or [])
    sandbox_status = state.get('sandbox_status', '')
    agent_1 = state.get('agent_1_results') or {}
    if sandbox_status == 'skipped_zero_trust':
        log.append('Dynamic tests skipped under Zero-Trust policy. Forensic analysis deferred.')
        return {'forensic_analysis': {'overall_forensic_summary': "Empirical penetration testing was halted on the host because the Docker sandbox was offline. Zero-Trust policy prevented untrusted artifact execution. All findings remain based on Agent 1's static threat model.", 'findings': []}, 'execution_plan_log': log}
    evidence_bundle = {'V1_poisoning': state.get('poisoning_evidence'), 'V4_adversarial': state.get('adversarial_evidence'), 'V2_preprocessing': state.get('preprocessing_evidence'), 'V3_validation': state.get('validation_evidence')}
    if not is_llm_available():
        fallback_forensics = _get_default_forensic_report(evidence_bundle)
        return {'forensic_analysis': fallback_forensics.model_dump(), 'execution_plan_log': log}
    try:
        diag_tools = make_forensic_diagnostic_tools(evidence_bundle=evidence_bundle, agent_1_results=agent_1)
        tool_map = {t.name: t for t in diag_tools}
        llm_with_tools = get_llm(temperature=0.0).bind_tools(diag_tools)
        system_prompt = "You are the AegisML Lead ML Forensic Security Diagnostician (Agent 2).\nYou have diagnostic tools to investigate empirical container telemetry:\n- bound_query_attack_telemetry(test_id): retrieves detailed measurements and degradation curves.\n- bound_evaluate_hypothesis_correlation(vulnerability_id): mathematically correlates Agent 1's claim with empirical telemetry to determine 'Confirmed Risk', 'False Positive (Mitigated)', 'Hidden Risk', or 'Unverified'.\n\nUse these tools to investigate anomalous telemetry, then produce your final ForensicAnalysisReport."
        user_prompt = f'Empirical Test Telemetry Summary:\n{evidence_bundle}\n\nAgent 1 Static Findings:\n{agent_1.get('vulnerability_findings')}\n\nInvestigate the empirical telemetry using your diagnostic tools, then formulate the forensic report.'
        messages: List[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        for _ in range(2):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                t_name = tc['name']
                t_args = tc.get('args', {})
                t_id = tc['id']
                if t_name in tool_map:
                    t_result = tool_map[t_name].invoke(t_args)
                    log.append(f"Forensic diagnostician queried '{t_name}' with args {t_args}.")
                else:
                    t_result = {'error': f'Unknown tool: {t_name}'}
                messages.append(ToolMessage(content=str(t_result), tool_call_id=t_id))
        messages.append(HumanMessage(content='Based on your empirical investigation and tool findings, produce the final ForensicAnalysisReport as a structured object. Document mathematical root causes and explicit hypothesis confirmations for each tested vulnerability.'))
        structured_llm = get_llm(temperature=0.1).with_structured_output(ForensicAnalysisReport)
        report = structured_llm.invoke(messages)
        if isinstance(report, dict):
            report = ForensicAnalysisReport.model_validate(report)
        log.append('Cognitive forensic diagnosis completed successfully via tool-empowered investigation.')
        return {'forensic_analysis': report.model_dump(), 'execution_plan_log': log}
    except Exception as e:
        log.append(f'LLM forensic diagnosis encountered error ({str(e)}). Used deterministic fallback.')
        fallback = _get_default_forensic_report(evidence_bundle)
        return {'forensic_analysis': fallback.model_dump(), 'execution_plan_log': log}

def node_aggregate_results(state: TestingAgentState) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []
    agent_1 = state.get('agent_1_results')
    p_ev = state.get('poisoning_evidence')
    if p_ev:
        results.append(p_ev)
        if agent_1:
            drop = p_ev.get('evidence', {}).get('accuracy_drop') or p_ev.get('evidence', {}).get('generic_test', {}).get('max_accuracy_drop')
            verifications.append(_verification_from_status('V1', 'Data Poisoning', p_ev, {'accuracy_drop': drop}))
    prep_ev = state.get('preprocessing_evidence')
    if prep_ev:
        results.append(prep_ev)
        if agent_1:
            verifications.append(_verification_from_status('V2', 'Preprocessing Attack Surface', prep_ev, prep_ev.get('evidence', {})))
    val_ev = state.get('validation_evidence')
    if val_ev:
        results.append(val_ev)
        if agent_1:
            verifications.append(_verification_from_status('V3', 'Data Validation Weaknesses', val_ev, {'accuracy_drop': val_ev.get('evidence', {}).get('accuracy_drop')}))
    adv_ev = state.get('adversarial_evidence')
    if adv_ev:
        results.append(adv_ev)
        if agent_1:
            asr = adv_ev.get('evidence', {}).get('attack_success_rate_within_budget')
            verifications.append(_verification_from_status('V4', 'Adversarial Robustness', adv_ev, {'attack_success_rate': asr}))
    present_ids = {r.get('vulnerability_id') for r in results}
    for vid, vname in MVP_VULNERABILITIES:
        if vid not in present_ids:
            results.append({'vulnerability_id': vid, 'vulnerability_name': vname, 'status': 'not_tested', 'severity': None, 'evidence': {'reason': 'Test was not scheduled in strategy.'}})
    result_order = {t.split('_')[0]: i for i, t in enumerate(TEST_ORDER)}
    results.sort(key=lambda r: result_order.get(r.get('vulnerability_id', ''), 99))
    return {'structured_test_results': {'results': results, 'hypothesis_verifications': verifications, 'attack_strategy_plan': state.get('attack_strategy_plan'), 'forensic_analysis': state.get('forensic_analysis'), 'sandbox_status': state.get('sandbox_status'), 'sandbox_telemetry': state.get('sandbox_telemetry'), 'execution_log': state.get('execution_plan_log') or []}, 'hypothesis_verifications': verifications, 'status': 'completed'}

def _verification_from_status(vulnerability_id: str, vulnerability_name: str, evidence_dict: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    test_status = evidence_dict.get('status', 'unverified')
    dynamic_severity = evidence_dict.get('severity')
    if test_status == 'vulnerable':
        correlation = 'Confirmed Risk'
        rationale = f'Empirical testing confirmed susceptibility ({dynamic_severity or 'detected'}).'
    elif test_status == 'not_vulnerable':
        correlation = 'False Positive (Mitigated)'
        rationale = 'Empirical tests showed the model resisted attack conditions.'
    elif test_status == 'not_applicable':
        correlation = 'Not Applicable'
        rationale = 'Test was not applicable to the target pipeline structure.'
    else:
        correlation = 'Unverified'
        rationale = 'Empirical testing was inconclusive or skipped under Zero-Trust policy.'
    return {'vulnerability_id': vulnerability_id, 'category': vulnerability_name, 'correlation_status': correlation, 'test_status': test_status, 'dynamic_severity': dynamic_severity, 'correlation_rationale': rationale, 'metrics': metrics}

def _get_default_strategy_plan(explicit_targets: Optional[List[str]], agent_1_results: Dict[str, Any], dataset_profile: Dict[str, Any]) -> AttackStrategyPlan:
    avg_words = dataset_profile.get('text_stats', {}).get('avg_word_count', 100)
    budget_info = calculate_perturbation_budget.invoke({'avg_word_count': avg_words, 'high_sparsity': True})
    adv_cfg = AdversarialAttackConfig(sample_size=budget_info.get('recommended_sample_size', 50), max_relative_perturbation_budget=budget_info.get('recommended_max_relative_budget', 0.4), max_iter=budget_info.get('recommended_max_iter', 45), rationale=budget_info.get('rationale', 'Standard budget for text classification.'))
    planned = list(TEST_ORDER)
    if explicit_targets:
        planned = [t for t in TEST_ORDER if any((x.lower() in t.lower() for x in explicit_targets))]
        if not planned:
            planned = list(TEST_ORDER)
    return AttackStrategyPlan(selected_tests=planned, adversarial_config=adv_cfg, planning_rationale='Deterministic rule-based baseline strategy.', strategy_provenance='static_baseline')

def _get_default_forensic_report(evidence_bundle: Dict[str, Any]) -> ForensicAnalysisReport:
    findings = []
    for test_key, ev in evidence_bundle.items():
        if not ev:
            continue
        status = ev.get('status', 'unverified')
        vid = ev.get('vulnerability_id', test_key[:2])
        vname = ev.get('vulnerability_name', test_key)
        findings.append(ForensicFinding(vulnerability_id=vid, category=vname, hypothesis_confirmation='Confirmed Risk' if status == 'vulnerable' else 'False Positive (Mitigated)' if status == 'not_vulnerable' else 'Unverified', root_cause_diagnosis=ev.get('summary', 'Automated empirical metric diagnosis.'), empirical_metric_summary=str(ev.get('evidence', {})), recommended_focus_area='Remediation guidance deferred to Agent 3.'))
    return ForensicAnalysisReport(findings=findings, overall_forensic_summary='Empirical test results evaluated across dynamic testing modules.')

def build_testing_agent_graph():
    graph = StateGraph(TestingAgentState)
    graph.add_node('prepare_metadata', traced_node('prepare_metadata')(node_prepare_metadata))
    graph.add_node('reason_strategy', traced_node('reason_strategy')(node_reason_strategy))
    graph.add_node('execute_sandbox', traced_node('execute_sandbox')(node_execute_sandbox))
    graph.add_node('forensic_diagnosis', traced_node('forensic_diagnosis')(node_forensic_diagnosis))
    graph.add_node('aggregate_results', traced_node('aggregate_results')(node_aggregate_results))
    graph.set_entry_point('prepare_metadata')
    graph.add_edge('prepare_metadata', 'reason_strategy')
    graph.add_edge('reason_strategy', 'execute_sandbox')
    graph.add_edge('execute_sandbox', 'forensic_diagnosis')
    graph.add_edge('forensic_diagnosis', 'aggregate_results')
    graph.add_edge('aggregate_results', END)
    return graph.compile()
