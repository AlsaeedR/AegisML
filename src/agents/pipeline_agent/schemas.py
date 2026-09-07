from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DeploymentContext(BaseModel):
    """
    Captures deployment context and threat landscape based on the
    NIST AI 100-2e2025 adversarial machine learning taxonomy.
    """
    protected_assets: List[str] = Field(
        description="Key assets requiring protection, such as training data, feature stores, model weights, and labels."
    )
    pipeline_components: List[str] = Field(
        description="Identified pipeline stages and modules handling data or models."
    )
    trust_boundaries: List[str] = Field(
        description="Boundaries where untrusted external data or input interacts with pipeline code."
    )
    attacker_goal: str = Field(
        description="Primary adversarial objective, such as evasion, poisoning, availability degradation, or privacy violation."
    )
    attacker_knowledge: Literal["black-box", "grey-box", "white-box", "supply-chain"] = Field(
        description="Attacker knowledge level per NIST AI taxonomy."
    )
    attacker_access: str = Field(
        description="Where and how the attacker can interact with the system (e.g. inference API, training set, dependencies)."
    )
    potential_impact: str = Field(
        description="Anticipated business and operational fallout if attacks succeed."
    )
    existing_controls: List[str] = Field(
        default_factory=list,
        description="Security, validation, or sanitization mechanisms currently present in the code."
    )


class ThreatItem(BaseModel):
    """
    Individual threat entry mapped to pipeline components and adversarial capabilities.
    """
    id: str = Field(description="Unique threat identifier, e.g. T-01")
    name: str = Field(description="Name or title of the threat")
    description: str = Field(description="Detailed explanation of how the threat manifests in this pipeline")
    affected_component: str = Field(description="Pipeline component where this threat operates")
    attacker_capability: str = Field(description="Required attacker capabilities or access permissions")
    potential_impact: str = Field(description="Impact of successful exploitation")
    mitigation: str = Field(description="Recommended mitigation strategy")


class ThreatModel(BaseModel):
    """
    Complete threat model container validated by Pydantic.
    """
    deployment_context: DeploymentContext
    threats: List[ThreatItem] = Field(
        default_factory=list,
        description="List of concrete threats identified across the pipeline stages."
    )


class VulnerabilityFinding(BaseModel):
    """
    Individual vulnerability finding mapped to one of the four MVP categories.
    """
    vulnerability_id: Literal["V1", "V2", "V3", "V4"] = Field(
        description="Standard identifier: V1 (Poisoning), V2 (Preprocessing), V3 (Validation), V4 (Adversarial Robustness)."
    )
    category: Literal[
        "Data Poisoning",
        "Preprocessing Attack Surface",
        "Data Validation Weaknesses",
        "Adversarial Robustness"
    ] = Field(description="Vulnerability category name.")
    description: str = Field(
        description="Technical analysis explaining how this vulnerability affects the target pipeline code."
    )
    affected_components: List[str] = Field(
        default_factory=list,
        description="Components and functions where the vulnerability is exposed."
    )
    nist_lifecycle_stage: Literal[
        "Data Ingestion",
        "Preprocessing",
        "Model Training",
        "Inference"
    ] = Field(description="Pipeline lifecycle stage affected by this vulnerability.")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable, code-level remediation recommendations."
    )


class VulnerabilitiesReport(BaseModel):
    """
    Container for the four MVP vulnerability findings and their recommendations.
    """
    vulnerabilities: List[VulnerabilityFinding] = Field(
        description="Collection of findings for the four MVP vulnerability classes."
    )

