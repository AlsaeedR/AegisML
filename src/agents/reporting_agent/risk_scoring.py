from typing import Any, Dict, Optional, Tuple


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
) -> Tuple[float, float, str]:
    """
    Get the initial impact, likelihood, and rationale from
    the static threat-modeling context.

    Impact is adjusted when multiple pipeline components are affected.
    """

    category = finding.get(
        "category",
        "default",
    )

    profile = STATIC_RISK_PROFILE.get(
        category,
        STATIC_RISK_PROFILE["default"],
    )

    status = str(finding.get("status", "vulnerable")).strip().lower()
    controls = finding.get("mitigating_controls", [])

    if status == "not_applicable":
        return (
            0.0,
            0.0,
            f"Static threat evaluation: {category} is not applicable to the evaluated pipeline architecture.",
        )

    supplied_impact = finding.get("static_impact")
    supplied_likelihood = finding.get("static_likelihood")

    try:
        impact = (
            max(0.0, min(10.0, float(supplied_impact)))
            if supplied_impact is not None
            else float(profile["impact"])
        )
    except (TypeError, ValueError):
        impact = float(profile["impact"])

    try:
        likelihood = (
            max(0.0, min(10.0, float(supplied_likelihood)))
            if supplied_likelihood is not None
            else float(profile["likelihood"])
        )
    except (TypeError, ValueError):
        likelihood = float(profile["likelihood"])

    if status == "mitigated":
        # Defended control reduces static likelihood significantly
        likelihood = min(likelihood, 2.5)
        controls_desc = f" ({', '.join(controls)})" if controls else ""
        rationale = (
            f"Static threat evaluation: impact {impact:.1f}/10, likelihood {likelihood:.1f}/10 for {category}; "
            f"mitigated by observed security controls{controls_desc}."
        )
    elif supplied_impact is not None or supplied_likelihood is not None:
        rationale = (
            f"Static threat context: impact {impact:.1f}/10, "
            f"likelihood {likelihood:.1f}/10 for {category}; "
            "available Agent 1 static values were preserved."
        )
    else:
        affected_components = finding.get(
            "affected_components",
            [],
        )

        rationale = (
            f"Theoretical baseline: impact {impact:.1f}/10, "
            f"likelihood {likelihood:.1f}/10 for {category}"
        )

        if len(affected_components) >= 3:
            impact = min(
                10.0,
                impact + 1.0,
            )
            rationale += (
                f"; impact adjusted to {impact:.1f}/10 "
                f"({len(affected_components)} affected components)"
            )

    return impact, likelihood, rationale


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
    static_severity_override: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Compare the theoretical/static assessment
    with Agent 2's empirical test result.

    False Positive:
        Theoretical severity is High or Critical,
        but Agent 2 does not confirm the vulnerability.

    Hidden Risk:
        Theoretical severity is Low or status is 'mitigated',
        but Agent 2 dynamically confirms the vulnerability.
    """

    static_status = str(finding.get("status", "vulnerable")).strip().lower()

    if static_severity_override:
        static_severity = static_severity_override.lower()
    else:
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

    if static_status == "not_applicable" or test_status == "not_applicable":
        return (
            "Not Applicable",
            (
                "Static and/or dynamic analysis determined that this "
                "threat class is not applicable to the pipeline."
            ),
        )

    if static_status == "mitigated":
        if test_status == "vulnerable":
            return (
                "Hidden Risk",
                (
                    "Agent 1 identified mitigating controls, but Agent 2 "
                    "empirical testing successfully bypassed them and confirmed "
                    "the vulnerability."
                ),
            )
        if test_status == "not_vulnerable":
            return (
                "Defended / Mitigated",
                (
                    "Agent 1 identified mitigating controls, and Agent 2 "
                    "empirical testing confirmed the defenses resisted attack."
                ),
            )
        return (
            "Mitigated",
            (
                "Agent 1 identified mitigating controls protecting this "
                "component from static threat exposure."
            ),
        )

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
    Compute both the static baseline and the final empirical
    AegisML risk assessment as the centralized scoring engine.

    Theoretical Risk = Impact × Static Likelihood / 10
    Final Risk       = Impact × Evidence-Adjusted Likelihood / 10
    """

    (
        impact,
        static_likelihood,
        static_score_rationale,
    ) = _get_static_context(
        finding
    )

    static_risk_score = round(
        (
            impact
            * static_likelihood
        )
        / 10.0,
        1,
    )

    static_severity = severity_from_score(
        static_risk_score
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
        finding,
        static_severity_override=static_severity,
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

    # Post-testing review of mitigating controls against empirical evidence
    controls = finding.get("mitigating_controls", [])
    static_status = str(finding.get("status", "vulnerable")).strip().lower()
    test_status = str(finding.get("test_status", "not_tested")).strip().lower()

    if not controls:
        control_verdict = "none"
    elif static_status == "not_applicable" or test_status == "not_applicable":
        control_verdict = "not_applicable"
    elif test_status == "not_vulnerable":
        control_verdict = "verified_effective"
    elif test_status == "vulnerable":
        if static_status == "mitigated":
            control_verdict = "bypassed"
        else:
            control_verdict = "ineffective"
    else:
        control_verdict = "unverified"

    return {
        "impact": impact,
        "control_verdict": control_verdict,

        "static_likelihood": (
            static_likelihood
        ),

        "static_risk_score": (
            static_risk_score
        ),

        "static_severity": (
            static_severity
        ),

        "static_score_rationale": (
            static_score_rationale
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