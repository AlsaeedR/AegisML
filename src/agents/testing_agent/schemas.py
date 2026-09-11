from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AdversarialAttackConfig(BaseModel):
    """Configuration for adversarial evasion testing (e.g. HopSkipJump via ART)."""
    algorithm: str = Field(
        default="HopSkipJump",
        description="Adversarial attack algorithm to execute."
    )
    sample_size: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Number of evaluation samples to target with boundary perturbations."
    )
    max_relative_perturbation_budget: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Relative perturbation budget threshold for realistic evasion."
    )
    max_iter: int = Field(
        default=50,
        ge=5,
        le=100,
        description="Maximum iterations for the iterative boundary optimization."
    )
    rationale: str = Field(
        default="",
        description="Cognitive rationale for selecting these specific attack parameters."
    )


class PoisoningAttackConfig(BaseModel):
    """Configuration for training data poisoning and corruption checks."""
    method: str = Field(
        default="clean_label_label_flip",
        description="Poisoning testing methodology."
    )
    poison_fractions: List[float] = Field(
        default_factory=lambda: [0.05, 0.1, 0.2],
        description="Fraction of dataset to corrupt or mislabel during retrain testing."
    )
    target_classes: Optional[List[str]] = Field(
        default=None,
        description="Target classes prioritized for poisoning."
    )
    rationale: str = Field(
        default="",
        description="Cognitive rationale for selecting these poisoning parameters."
    )


class PreprocessingTestConfig(BaseModel):
    """Configuration for dynamic text preprocessing stress-testing."""
    include_fuzzing_cases: bool = Field(
        default=True,
        description="Whether to run edge-case malformed string inputs."
    )
    semantic_degradation_checks: bool = Field(
        default=True,
        description="Whether to verify if preprocessed tokens collapse unexpectedly."
    )
    rationale: str = Field(
        default="",
        description="Cognitive rationale for preprocessing test parameters."
    )


class ValidationTestConfig(BaseModel):
    """Configuration for schema and data boundary integrity tests."""
    test_missing_values: bool = Field(
        default=True,
        description="Inject missing and null values to verify input validation gates."
    )
    test_duplicate_injection: bool = Field(
        default=True,
        description="Inject duplicate records to test deduplication resilience."
    )
    rationale: str = Field(
        default="",
        description="Cognitive rationale for data validation checks."
    )


class AttackStrategyPlan(BaseModel):
    """
    Comprehensive attack strategy formulated by Agent 2's Cognitive Pre-Attack Reasoning node.
    """
    selected_tests: List[str] = Field(
        default_factory=lambda: [
            "V1_poisoning",
            "V4_adversarial",
            "V2_preprocessing",
            "V3_validation",
        ],
        description="List of test identifiers to execute."
    )
    adversarial_config: AdversarialAttackConfig = Field(
        default_factory=AdversarialAttackConfig,
        description="Tactical configuration for adversarial evasion tests."
    )
    poisoning_config: PoisoningAttackConfig = Field(
        default_factory=PoisoningAttackConfig,
        description="Tactical configuration for data poisoning tests."
    )
    preprocessing_config: PreprocessingTestConfig = Field(
        default_factory=PreprocessingTestConfig,
        description="Tactical configuration for preprocessing tests."
    )
    validation_config: ValidationTestConfig = Field(
        default_factory=ValidationTestConfig,
        description="Tactical configuration for data validation checks."
    )
    planning_rationale: str = Field(
        default="",
        description="High-level cognitive justification synthesizing Agent 1 threat model and data profile."
    )


class ForensicFinding(BaseModel):
    """
    Forensic interpretation produced by Agent 2's Post-Attack Reasoning node.
    """
    vulnerability_id: str = Field(
        description="ID of the vulnerability (e.g. V1, V2, V3, V4)."
    )
    category: str = Field(
        description="Category name of the vulnerability."
    )
    hypothesis_confirmation: str = Field(
        description="Empirical confirmation: 'Confirmed Risk', 'False Positive (Mitigated)', 'Hidden Risk', or 'Unverified'."
    )
    root_cause_diagnosis: str = Field(
        description="Mathematical and code-level root cause deduced from empirical test telemetry."
    )
    empirical_metric_summary: str = Field(
        description="Key numerical telemetry observed during the attack."
    )
    recommended_focus_area: str = Field(
        description="Specific architectural or code element requiring mitigation."
    )


class ForensicAnalysisReport(BaseModel):
    """
    Complete forensic post-attack synthesis for Agent 2.
    """
    findings: List[ForensicFinding] = Field(
        default_factory=list,
        description="List of forensic interpretations per tested vulnerability."
    )
    overall_forensic_summary: str = Field(
        default="",
        description="Holistic post-attack forensic assessment for the target pipeline."
    )

