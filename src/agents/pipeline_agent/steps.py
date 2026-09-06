import json
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import get_llm
from .code_parser import parse_python_pipeline, build_networkx_graph


def extract_pipeline_from_code(code: str) -> Dict[str, Any]:
    return parse_python_pipeline(code)


def build_threat_model(code, pipeline_graph, testing_agent_results=None):
    llm = get_llm()
    parser = JsonOutputParser()

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Pipeline Agent. Build a threat model from the extracted pipeline.\n"
         "Return ONLY valid JSON. No prose, no explanations."),
        ("user",
         "CODE:\n{code}\n\n"
         "PIPELINE_GRAPH:\n{pipeline_graph}\n\n"
         "Testing Agent results:\n{testing_agent_results}\n\n"
         "Follow this JSON format:\n{format_instructions}\n\n"
         "Now output the threat_model as JSON.")
    ])

    chain = prompt | llm | parser

    return chain.invoke({
        "code": code,
        "pipeline_graph": json.dumps(pipeline_graph, indent=2),
        "testing_agent_results": json.dumps(testing_agent_results or {}, indent=2),
        "format_instructions": format_instructions,
    })


def identify_vulnerabilities(
    code: str,
    pipeline_graph: Dict[str, Any],
    threat_model: Dict[str, Any],
    testing_agent_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    llm = get_llm()
    parser = JsonOutputParser()

    strict_schema = {
        "vulnerabilities": [
            {
                "id": "V1",
                "name": "Data Poisoning",
                "description": "",
                "mitigation": [],
                "related_assets": [],
                "related_entry_points": []
            },
            {
                "id": "V2",
                "name": "Preprocessing Attack Surface",
                "description": "",
                "mitigation": [],
                "related_assets": [],
                "related_entry_points": []
            },
            {
                "id": "V3",
                "name": "Data Validation Weaknesses",
                "description": "",
                "mitigation": [],
                "related_assets": [],
                "related_entry_points": []
            },
            {
                "id": "V4",
                "name": "Adversarial Robustness",
                "description": "",
                "mitigation": [],
                "related_assets": [],
                "related_entry_points": []
            }
        ]
    }

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You MUST output ONLY the following 4 vulnerabilities:\n"
         "1. Data Poisoning\n"
         "2. Preprocessing Attack Surface\n"
         "3. Data Validation Weaknesses\n"
         "4. Adversarial Robustness\n\n"
         "Do NOT output risk scores, severity, likelihood, or any scoring.\n"
         "Return ONLY valid JSON matching the schema provided."),
        ("user",
         "CODE:\n{code}\n\n"
         "PIPELINE_GRAPH:\n{pipeline_graph}\n\n"
         "THREAT_MODEL:\n{threat_model}\n\n"
         "Testing Agent results:\n{testing_agent_results}\n\n"
         "JSON Schema:\n{schema}\n\n"
         "Format instructions:\n{format_instructions}\n\n"
         "Now output ONLY the four MVP vulnerabilities as JSON.")
    ])

    chain = prompt | llm | parser

    return chain.invoke({
        "code": code,
        "pipeline_graph": json.dumps(pipeline_graph, indent=2),
        "threat_model": json.dumps(threat_model, indent=2),
        "testing_agent_results": json.dumps(testing_agent_results or {}, indent=2),
        "schema": json.dumps(strict_schema, indent=2),
        "format_instructions": format_instructions,
    })

