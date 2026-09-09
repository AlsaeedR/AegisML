from typing import Any, Dict, Tuple


# Static risk context derived from the threat-modeling stage.
# These values represent the initial impact and likelihood
# before Agent 2 empirical testing is considered.
STATIC_RISK_PROFILE = {
    "Data Poisoning": {
        "impact": 9.0,
        "likelihood": 7.0,
    },
    "Preprocessing Attack Surface": {
        "impact": 6.0,
        "likelihood": 8.0,
    },
    "Data Validation Weaknesses": {
        "impact": 5.0,
        "likelihood": 9.0,
    },
    "Adversarial Robustness": {
        "impact": 10.0,
        "likelihood": 9.0,
    },
    "default": {
        "impact": 3.0,
        "likelihood": 3.0,
    },
}


DYNAMIC_SEVERITY_LIKELIHOOD = {
    "low": 3.0,
    "medium": 6.0,
    "high": 8.0,
    "critical": 10.0,
}


def severity_from_score(score: float) -> str:
    """
    Convert a 0-10 final risk score into
    the AegisML severity scale.
    """

    if score >= 7.5:
        return "Critical"

    if score >= 5.0:
        return "High"

    if score >= 2.5:
        return "Medium"

    return "Low"


def _get_static_context(
    finding: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Get the initial impact and likelihood from
    the static threat-modeling context.

    Impact can be increased when several pipeline
    components are affected, matching Agent 1's
    existing contextual logic.
    """

    category = finding.get(
        "category",
        "default",
    )

    profile = STATIC_RISK_PROFILE.get(
        category,
        STATIC_RISK_PROFILE["default"],
    )

    impact = float(
        profile["impact"]
    )

    likelihood = float(
        profile["likelihood"]
    )

    affected_components = finding.get(
        "affected_components",
        [],
    )

    if len(affected_components) >= 3:
        impact = min(
            10.0,
            impact + 1.0,
        )

    return impact, likelihood


def _dynamic_likelihood(
    finding: Dict[str, Any],
    static_likelihood: float,
) -> Tuple[float, str]:
    """
    Determine an evidence-adjusted likelihood
    using Agent 2 empirical testing.

    If testing is unavailable or inconclusive,
    the static likelihood is retained rather
    than inventing a lower risk.
    """

    status = str(
        finding.get(
            "test_status",
            "not_tested",
        )
    ).lower()

    dynamic_severity = finding.get(
        "dynamic_severity"
    )

    evidence = finding.get(
        "evidence",
        {},
    ) or {}

    vulnerability_id = finding.get(
        "vulnerability_id"
    )

    if status == "not_applicable":
        return (
            0.0,
            "Dynamic testing determined that this "
            "vulnerability is not applicable to "
            "the evaluated pipeline.",
        )

    if status == "not_vulnerable":
        return (
            2.0,
            "Dynamic testing did not confirm the "
            "vulnerability; residual likelihood "
            "is retained at a low level.",
        )

    if status in {
        "not_tested",
        "inconclusive",
        "error",
    }:
        return (
            static_likelihood,
            "Dynamic evidence was unavailable or "
            "inconclusive, so the static likelihood "
            "estimate was retained.",
        )

    if status != "vulnerable":
        return (
            static_likelihood,
            "Unknown dynamic status; static "
            "likelihood estimate was retained.",
        )

    # V4 provides a direct empirical attack-success metric.
    if vulnerability_id == "V4":
        attack_success_rate = evidence.get(
            "attack_success_rate_within_budget"
        )

        if attack_success_rate is not None:
            try:
                attack_success_rate = float(
                    attack_success_rate
                )

                likelihood = max(
                    0.0,
                    min(
                        10.0,
                        attack_success_rate * 10.0,
                    ),
                )

                return (
                    likelihood,
                    (
                        "Likelihood derived from Agent 2 "
                        "attack success rate within the "
                        "perturbation budget."
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

    # Other dynamic tests use Agent 2's empirical severity.
    if dynamic_severity:
        severity_key = str(
            dynamic_severity
        ).lower()

        likelihood = (
            DYNAMIC_SEVERITY_LIKELIHOOD.get(
                severity_key
            )
        )

        if likelihood is not None:
            return (
                likelihood,
                (
                    "Likelihood adjusted using "
                    "Agent 2 empirical test status "
                    "and dynamic severity."
                ),
            )

    return (
        static_likelihood,
        "Vulnerability was dynamically confirmed, "
        "but no usable empirical severity metric "
        "was available; static likelihood retained.",
    )


def _correlation_status(
    finding: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Compare Agent 1's theoretical/static assessment
    with Agent 2's empirical test result.

    False Positive:
        Agent 1 rates the finding High or Critical,
        but Agent 2 does not confirm the vulnerability.

    Hidden Risk:
        Agent 1 rates the finding Low,
        but Agent 2 dynamically confirms it.
    """

    static_severity = str(
        finding.get(
            "static_severity",
            "Low",
        )
    ).lower()

    test_status = str(
        finding.get(
            "test_status",
            "not_tested",
        )
    ).lower()

    if (
        static_severity
        in {"high", "critical"}
        and test_status == "not_vulnerable"
    ):
        return (
            "False Positive",
            (
                "Agent 1 identified a high theoretical "
                "risk, but Agent 2 did not confirm the "
                "vulnerability during dynamic testing."
            ),
        )

    if (
        static_severity == "low"
        and test_status == "vulnerable"
    ):
        return (
            "Hidden Risk",
            (
                "Agent 1 assigned a low theoretical "
                "risk, but Agent 2 dynamically confirmed "
                "the vulnerability."
            ),
        )

    if test_status == "vulnerable":
        return (
            "Confirmed Risk",
            (
                "Agent 2 dynamically confirmed the "
                "vulnerability identified by Agent 1."
            ),
        )

    if test_status == "not_vulnerable":
        return (
            "Not Confirmed",
            (
                "Agent 2 did not confirm the vulnerability "
                "during dynamic testing."
            ),
        )

    if test_status == "not_applicable":
        return (
            "Not Applicable",
            (
                "Agent 2 determined that the vulnerability "
                "is not applicable to this pipeline."
            ),
        )

    return (
        "Unverified",
        (
            "Dynamic evidence was unavailable, incomplete, "
            "or inconclusive."
        ),
    )


def calculate_final_risk(
    finding: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute the final AegisML risk assessment.

    Final Risk =
        Impact × Evidence-Adjusted Likelihood / 10

    Agent 1 contributes the theoretical/static
    threat context.

    Agent 2 contributes empirical security
    testing evidence.

    Agent 3 correlates both assessments,
    identifies discrepancies such as False
    Positives and Hidden Risks, and calculates
    the final evidence-informed risk.
    """

    (
        impact,
        static_likelihood,
    ) = _get_static_context(
        finding
    )

    (
        final_likelihood,
        evidence_rationale,
    ) = _dynamic_likelihood(
        finding,
        static_likelihood,
    )

    (
        correlation_status,
        correlation_rationale,
    ) = _correlation_status(
        finding
    )

    final_risk_score = round(
        (
            impact
            * final_likelihood
        )
        / 10.0,
        1,
    )

    final_severity = severity_from_score(
        final_risk_score
    )

    return {
        "impact": impact,

        "static_likelihood": (
            static_likelihood
        ),

        "final_likelihood": round(
            final_likelihood,
            1,
        ),

        "final_risk_score": (
            final_risk_score
        ),

        "final_severity": (
            final_severity
        ),

        "correlation_status": (
            correlation_status
        ),

        "correlation_rationale": (
            correlation_rationale
        ),

        "risk_rationale": (
            f"Impact {impact}/10 combined with "
            f"evidence-adjusted likelihood "
            f"{final_likelihood:.1f}/10. "
            f"{evidence_rationale} "
            f"{correlation_rationale} "
            f"Final risk score: "
            f"{final_risk_score}/10 "
            f"({final_severity})."
        ),
    }