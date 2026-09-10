from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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

    final_risk_score: float = Field(
        ge=0,
        le=10,
    )

    final_severity: str

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