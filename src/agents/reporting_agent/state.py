from typing import TypedDict, Any, Dict, List, Optional


class ReportingAgentState(TypedDict, total=False):
    """
    Shared state for the Reporting Agent.
    Maintains upstream Agent 1 and Agent 2 results, correlated findings,
    mathematical risk scores, generated audit reports, and self-correction tracking.
    """

    agent_1_results: Optional[Dict[str, Any]]
    agent_2_results: Optional[Dict[str, Any]]
    correlated_findings: List[Dict[str, Any]]
    overall_risk: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    execution_log: List[str]
    status: str

    # Self-correction and reflection tracking
    validation_errors: Optional[str]
    retry_count: int
    max_retries: int