from typing import TypedDict, Any, Dict, List, Optional


class ReportingAgentState(TypedDict, total=False):
    """
    Shared state for the Reporting Agent.
    """

    agent_1_results: Optional[
        Dict[str, Any]
    ]

    agent_2_results: Optional[
        Dict[str, Any]
    ]

    correlated_findings: List[
        Dict[str, Any]
    ]

    overall_risk: Optional[
        Dict[str, Any]
    ]

    final_report: Optional[
        Dict[str, Any]
    ]

    execution_log: List[str]

    status: str