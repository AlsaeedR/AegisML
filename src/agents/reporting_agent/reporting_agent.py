from typing import Any, Dict

from .graph import build_reporting_agent_graph
from .state import ReportingAgentState


def run_reporting_agent(
    agent_1_results: Dict[str, Any],
    agent_2_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the Reporting Agent.

    Combines Pipeline Agent risk findings with Testing Agent
    empirical evidence and generates the final audit report.
    """

    app = build_reporting_agent_graph()

    initial_state: ReportingAgentState = {
        "agent_1_results": agent_1_results,
        "agent_2_results": agent_2_results,
        "correlated_findings": [],
        "execution_log": [],
        "status": "initialized",
    }

    return app.invoke(
        initial_state
    )