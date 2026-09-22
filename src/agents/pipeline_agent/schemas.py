from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

# BaseModel from pydantic = this is how we define a strict "shape" that data
# must match. If the LLM's output doesn't match this shape, it gets rejected
# during validation (that's what validate_threat_model_schema checks against)


class DeploymentContext(BaseModel):
    """
    Captures deployment context and threat landscape based on the
    NIST AI 100-2e2025 adversarial machine learning taxonomy.
    """
    # a list of things we need to protect — like the training data, the
    # trained model weights, or the labels
    protected_assets: List[str] = Field(
        description="Key assets requiring protection, such as training data, feature stores, model weights, and labels."
    )
    # a list of the pipeline stages/modules found in the code
    pipeline_components: List[str] = Field(
        description="Identified pipeline stages and modules handling data or models."
    )
    # where untrusted data crosses into the pipeline
    trust_boundaries: List[str] = Field(
        description="Boundaries where untrusted external data or input interacts with pipeline code."
    )
    # what the attacker is actually trying to achieve
    attacker_goal: str = Field(
        description="Primary adversarial objective, such as evasion, poisoning, availability degradation, or privacy violation."
    )
    # Literal here means ONLY these 4 exact strings are allowed — nothing
    # else. if the LLM writes anything different, validation fails
    attacker_knowledge: Literal["black-box", "grey-box", "white-box", "supply-chain"] = Field(
        description="Attacker knowledge level per NIST AI taxonomy."
    )
    # how/where the attacker can actually touch the system
    attacker_access: str = Field(
        description="Where and how the attacker can interact with the system (e.g. inference API, training set, dependencies)."
    )
    # what happens if the attack succeeds
    potential_impact: str = Field(
        description="Anticipated business and operational fallout if attacks succeed."
    )
    # default_factory=list means: if this field is missing, just default to
    # an empty list instead of throwing an error
    existing_controls: List[str] = Field(
        default_factory=list,
        description="Security, validation, or sanitization mechanisms currently present in the code."
    )


class ThreatItem(BaseModel):
    """
    Individual threat entry mapped to pipeline components and adversarial capabilities.
    """
    # just a simple ID, like "T-01"
    id: str = Field(description="Unique threat identifier, e.g. T-01")
    name: str = Field(description="Name or title of the threat")
    description: str = Field(description="Detailed explanation of how the threat manifests in this pipeline")
    affected_component: str = Field(description="Pipeline component where this threat operates")
    attacker_capability: str = Field(description="Required attacker capabilities or access permissions")
    potential_impact: str = Field(description="Impact of successful exploitation")
    mitigation: str = Field(description="Recommended mitigation strategy")
    # this whole class is a more general/free-form threat entry, separate
    # from the 4 specific V1-V4 vulnerability classes below


class ThreatModel(BaseModel):
    """
    Complete threat model container validated by Pydantic.
    """
    # just bundles the deployment context together with a list of threats.
    # this is the exact top-level shape that validate_threat_model_schema
    # checks the LLM's output against
    deployment_context: DeploymentContext
    threats: List[ThreatItem] = Field(
        default_factory=list,
        description="List of concrete threats identified across the pipeline stages."
    )


class VulnerabilityFinding(BaseModel):
    """
    Individual vulnerability finding mapped to one of the four MVP categories.
    """
    # Literal["V1", "V2", "V3", "V4"] = a hard rule. the LLM CANNOT invent a
    # "V5" or misspell it — only these 4 exact values are accepted
    vulnerability_id: Literal["V1", "V2", "V3", "V4"] = Field(
        description="Standard identifier: V1 (Poisoning), V2 (Preprocessing), V3 (Validation), V4 (Adversarial Robustness)."
    )
    # same idea, locks the category name to these 4 exact strings
    category: Literal[
        "Data Poisoning",
        "Preprocessing Attack Surface",
        "Data Validation Weaknesses",
        "Adversarial Robustness"
    ] = Field(description="Vulnerability category name.")
    description: str = Field(
        description="Technical analysis explaining how this vulnerability affects the target pipeline code."
    )
    # which parts of the code this vulnerability actually touches
    affected_components: List[str] = Field(
        default_factory=list,
        description="Components and functions where the vulnerability is exposed."
    )
    # which ML lifecycle stage this maps to
    nist_lifecycle_stage: Literal[
        "Data Ingestion",
        "Preprocessing",
        "Model Training",
        "Inference"
    ] = Field(description="Pipeline lifecycle stage affected by this vulnerability.")
    # this is THE field that node_validate_vulnerabilities in pipeline_agent.py
    # directly overwrites to "not_applicable" when there's no inference node.
    # default="vulnerable" means if nothing says otherwise, assume it's vulnerable
    status: Literal["vulnerable", "mitigated", "not_applicable"] = Field(
        default="vulnerable",
        description="Assessment status: 'vulnerable' if unmitigated weaknesses exist, 'mitigated' if existing controls defend against the threat, or 'not_applicable' if the lifecycle stage is absent."
    )
    mitigating_controls: List[str] = Field(
        default_factory=list,
        description="Defensive mechanisms, sanitizers, or validation controls identified in target code."
    )
    # suggested fixes — but note the comment in tools.py said Agent 1 doesn't
    # really finalize these; that's more Agent 3's job with full evidence
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable remediation recommendations (deferred to Agent 3 evidence synthesis)."
    )
    # all four of these below are Optional and default to None — meaning
    # Agent 1 leaves them EMPTY on purpose. risk scoring happens later,
    # after Agent 2 actually tests things empirically
    severity: Optional[str] = Field(
        default=None,
        description="Risk severity level calculated by the risk scoring module."
    )
    risk_score: Optional[float] = Field(
        default=None,
        description="Numerical risk score (0-10) calculated by the risk scoring module."
    )
    static_impact: Optional[float] = Field(
        default=None,
        description="Dynamically estimated static impact (0-10) based on pipeline blast radius."
    )
    static_likelihood: Optional[float] = Field(
        default=None,
        description="Dynamically estimated static likelihood (0-10) based on trust boundary exposure and controls."
    )
    static_score_rationale: Optional[str] = Field(
        default=None,
        description="Architectural rationale explaining the dynamic static impact and likelihood derivation."
    )


class VulnerabilitiesReport(BaseModel):
    """
    Container for the four MVP vulnerability findings identified through static threat analysis.
    """
    # just a list of VulnerabilityFinding objects. this exact shape is what
    # validate_vulnerabilities_schema checks, including the rule that all
    # 4 categories (V1-V4) must be present somewhere in this list
    vulnerabilities: List[VulnerabilityFinding] = Field(
        description="Collection of findings for the four MVP vulnerability classes."
    )


# 🔴 different from everything above — this is a TypedDict, not a
# BaseModel. It's not for validating LLM output, it's the shape of the
# STATE that gets passed between every node in the LangGraph. Every
# function in pipeline_agent.py takes "state: PipelineAgentState" as its
# input — this class is literally the definition of what that state contains
class PipelineAgentState(TypedDict, total=False):
    """
    Internal state for Agent 1 (Pipeline & Threat Modeling Agent).
    Maintains code artifacts, extracted topological graphs, intermediate LLM
    reasoning outputs, and validation feedback used in self-correction loops.
    """
    # total=False means none of these fields are strictly required —
    # the state can be built up gradually as it moves station to station
    code: str
    pipeline_graph: dict
    networkx_graph: Any
    graph_topology: dict
    threat_model: dict
    vulnerability_findings: dict
    testing_agent_results: Optional[dict]
    validation_errors: Optional[str]
    retry_count: int
    max_retries: int
    status: str
    audit_id: Optional[str]
