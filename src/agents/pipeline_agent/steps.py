import json
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import get_llm
from .schemas import ThreatModel, VulnerabilitiesReport


def generate_threat_model_step(
    code: str,
    pipeline_graph: Dict[str, Any],
    graph_topology: Dict[str, Any],
    validation_errors: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prompts the LLM to inspect the pipeline code and structural graph, then
    derive a threat model grounded in the NIST AI 100-2e2025 taxonomy.
    If previous validation errors exist, includes them for self-correction.
    """
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=ThreatModel)
    format_instructions = parser.get_format_instructions()

    error_feedback = ""
    if validation_errors:
        error_feedback = (
            f"\nATTENTION: A prior validation attempt failed with the following errors:\n"
            f"{validation_errors}\n"
            f"Please adjust your output to strictly resolve these issues.\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the Pipeline & Threat Modeling Agent in the AegisML system.\n"
            "Your objective is to inspect the provided ML pipeline code and its extracted structural graph, "
            "then establish the deployment context and threat model based on the NIST AI 100-2e2025 adversarial ML taxonomy.\n\n"
            "Deployment context must capture:\n"
            "- protected_assets: data, features, weights, configuration, labels.\n"
            "- pipeline_components: distinct functional steps in the code.\n"
            "- trust_boundaries: locations where untrusted user or external data crosses into the pipeline.\n"
            "- attacker_goal: primary objective (e.g. evasion, availability, integrity poisoning).\n"
            "- attacker_knowledge: one of 'black-box', 'grey-box', 'white-box', or 'supply-chain'.\n"
            "- attacker_access: physical, network, API, or data access level.\n"
            "- potential_impact: operational and security consequence.\n"
            "- existing_controls: sanitization, validation, or defenses present in the code.\n\n"
            "Return valid JSON matching the requested schema. Do not include markdown code block ticks or explanations outside the JSON."
        ),
        (
            "user",
            "SOURCE CODE:\n{code}\n\n"
            "EXTRACTED PIPELINE GRAPH:\n{pipeline_graph}\n\n"
            "GRAPH TOPOLOGY SUMMARY:\n{graph_topology}\n"
            "{error_feedback}\n"
            "SCHEMA INSTRUCTIONS:\n{format_instructions}\n\n"
            "Produce the complete threat model as JSON:"
        ),
    ])

    chain = prompt | llm | parser

    return chain.invoke({
        "code": code,
        "pipeline_graph": json.dumps(pipeline_graph, indent=2),
        "graph_topology": json.dumps(graph_topology, indent=2),
        "error_feedback": error_feedback,
        "format_instructions": format_instructions,
    })


def generate_vulnerabilities_step(
    code: str,
    pipeline_graph: Dict[str, Any],
    threat_model: Dict[str, Any],
    validation_errors: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prompts the LLM to analyze the pipeline against the four MVP vulnerability classes:
    1. Data Poisoning (V1)
    2. Preprocessing Attack Surface (V2)
    3. Data Validation Weaknesses (V3)
    4. Adversarial Robustness (V4)
    Generates actionable remediation recommendations for each vulnerability.
    Omits numerical risk scoring until dynamic testing results from Agent 2 become available.
    """
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=VulnerabilitiesReport)
    format_instructions = parser.get_format_instructions()

    error_feedback = ""
    if validation_errors:
        error_feedback = (
            f"\nATTENTION: A prior validation attempt failed with the following errors:\n"
            f"{validation_errors}\n"
            f"Please adjust your output to resolve these validation issues.\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the Pipeline & Threat Modeling Agent in the AegisML system.\n"
            "Analyze the ML pipeline code and established threat model to assess the four MVP vulnerability classes:\n"
            "1. V1 - Data Poisoning\n"
            "2. V2 - Preprocessing Attack Surface\n"
            "3. V3 - Data Validation Weaknesses\n"
            "4. V4 - Adversarial Robustness\n\n"
            "For each of these four classes:\n"
            "- Map to the relevant nist_lifecycle_stage ('Data Ingestion', 'Preprocessing', 'Model Training', or 'Inference').\n"
            "- Identify the specific affected_components in the code.\n"
            "- Provide a clear technical description of the vulnerability mechanism.\n"
            "- Provide concrete, code-level recommendations to remediate the vulnerability.\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Do NOT generate risk severity scores, likelihood scores, or risk rankings. Numerical risk scoring is deferred until Agent 2 empirical testing.\n"
            "- Ensure all 4 MVP vulnerability classes are present in the 'vulnerabilities' array.\n"
            "- Return valid JSON matching the schema instructions."
        ),
        (
            "user",
            "SOURCE CODE:\n{code}\n\n"
            "PIPELINE GRAPH:\n{pipeline_graph}\n\n"
            "THREAT MODEL:\n{threat_model}\n"
            "{error_feedback}\n"
            "SCHEMA INSTRUCTIONS:\n{format_instructions}\n\n"
            "Produce the vulnerabilities report as JSON:"
        ),
    ])

    chain = prompt | llm | parser

    return chain.invoke({
        "code": code,
        "pipeline_graph": json.dumps(pipeline_graph, indent=2),
        "threat_model": json.dumps(threat_model, indent=2),
        "error_feedback": error_feedback,
        "format_instructions": format_instructions,
    })
