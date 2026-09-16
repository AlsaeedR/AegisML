from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator


class ReportFinding(BaseModel):
    vulnerability_id: Literal[
        "V1",
        "V2",
        "V3",
        "V4",
    ]

    category: Literal[
        "Data Poisoning",
        "Preprocessing Attack Surface",
        "Data Validation Weaknesses",
        "Adversarial Robustness",
    ]

    # Agent 1 theoretical/static assessment
    static_impact: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
    )

    static_risk_score: float = Field(
        ge=0,
        le=10,
    )

    static_severity: str

    static_score_rationale: str = ""

    status: Optional[str] = Field(
        default="vulnerable",
        description="Static assessment status: vulnerable, mitigated, or not_applicable."
    )

    mitigating_controls: List[str] = Field(
        default_factory=list,
        description="Defensive controls observed during static inspection."
    )

    control_verdict: Optional[str] = Field(
        default=None,
        description="Post-testing review of mitigating controls: 'verified_effective', 'bypassed', 'ineffective', or 'none'."
    )

    # Agent 2 empirical/dynamic assessment
    test_status: str

    dynamic_severity: Optional[str] = None

    evidence: Dict[str, Any] = Field(
        default_factory=dict
    )

    # Agent 3 correlation result
    correlation_status: str

    correlation_rationale: str

    # Agent 3 final risk assessment
    impact: float = Field(
        ge=0,
        le=10,
    )

    static_likelihood: float = Field(
        ge=0,
        le=10,
    )

    final_likelihood: float = Field(
        ge=0,
        le=10,
    )

    # Primary Authoritative Risk Assessment (NIST SP 800-30 Rev. 1)
    risk_score: float = Field(
        ge=0,
        le=10,
        description="Authoritative, empirical risk score reconciled from static and dynamic evidence.",
    )

    severity: str = Field(
        description="Authoritative severity tier: Critical, High, Medium, or Low.",
    )

    final_risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="Backward-compatible alias for risk_score.",
    )

    final_severity: Optional[str] = Field(
        default=None,
        description="Backward-compatible alias for severity.",
    )

    @model_validator(mode="before")
    @classmethod
    def reconcile_authoritative_and_legacy_scores(cls, data: Any) -> Any:
        if isinstance(data, dict):
            r_score = data.get("risk_score")
            f_score = data.get("final_risk_score")
            if r_score is None and f_score is not None:
                data["risk_score"] = f_score
            elif f_score is None and r_score is not None:
                data["final_risk_score"] = r_score
            elif r_score is None and f_score is None:
                s_score = data.get("static_risk_score", 0.0)
                data["risk_score"] = s_score
                data["final_risk_score"] = s_score

            r_sev = data.get("severity")
            f_sev = data.get("final_severity")
            if r_sev is None and f_sev is not None:
                data["severity"] = f_sev
            elif f_sev is None and r_sev is not None:
                data["final_severity"] = r_sev
            elif r_sev is None and f_sev is None:
                s_sev = data.get("static_severity", "Low")
                data["severity"] = s_sev
                data["final_severity"] = s_sev
        return data

    risk_rationale: str

    affected_components: List[str] = Field(
        default_factory=list
    )

    description: str = ""

    recommendations: List[str] = Field(
        default_factory=list
    )


class OverallRiskSummary(BaseModel):
    overall_risk_score: float = Field(
        ge=0,
        le=10,
    )

    overall_severity: str

    total_findings: int = Field(
        ge=0
    )

    confirmed_findings: int = Field(
        default=0,
        ge=0,
    )

    false_positive_findings: int = Field(
        default=0,
        ge=0,
    )

    hidden_risk_findings: int = Field(
        default=0,
        ge=0,
    )

    unverified_findings: int = Field(
        default=0,
        ge=0,
    )

    not_applicable_findings: int = Field(
        default=0,
        ge=0,
    )

    mitigated_findings: int = Field(
        default=0,
        ge=0,
    )

    critical_findings: int = Field(
        default=0,
        ge=0,
    )

    high_findings: int = Field(
        default=0,
        ge=0,
    )

    medium_findings: int = Field(
        default=0,
        ge=0,
    )

    low_findings: int = Field(
        default=0,
        ge=0,
    )


class AuditReport(BaseModel):
    executive_summary: str

    overall_risk: OverallRiskSummary

    findings: List[ReportFinding] = Field(
        default_factory=list
    )

    recommendations: List[str] = Field(
        default_factory=list
    )


class ReportingAgentOutput(BaseModel):
    final_report: AuditReport


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
    validation_errors: Optional[str]
    retry_count: int
    max_retries: int
    audit_id: Optional[str]
