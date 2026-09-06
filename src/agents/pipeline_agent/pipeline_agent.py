from .graph import build_pipeline_agent_graph
from .state import PipelineAgentState

def run_pipeline_agent(code: str, testing_agent_results=None):
    app = build_pipeline_agent_graph()

    initial_state: PipelineAgentState = {
        "code": code,
        "testing_agent_results": testing_agent_results or {},
    }

    return app.invoke(initial_state)

