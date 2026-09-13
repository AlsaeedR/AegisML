"""
AegisML Testing Agent — Shared Constants

Single source of truth for MVP vulnerability definitions and execution order.
Both graph.py (host-side orchestration) and sandbox_worker.py (container-side
execution) import from here.
"""

# Canonical execution order for the 4 MVP vulnerability tests.
# Order is intentional:
#   - V1 and V4 are model-level attacks (heavier, higher severity) → run first
#     so critical results are captured even if the sandbox hits a timeout.
#   - V4 must run on the untouched baseline model, before V1's retrain state
#     could affect in-memory artifacts.
#   - V2 and V3 are pipeline-level static checks (fast) → run last.
TEST_ORDER = [
    "V1_poisoning",
    "V4_adversarial",
    "V2_preprocessing",
    "V3_validation",
]

# Canonical mapping from short vulnerability IDs to human-readable names.
# Used in node_aggregate_results (fill-in for untested classes) and
# hypothesis verification labels. Order matches TEST_ORDER priority.
MVP_VULNERABILITIES = [
    ("V1", "Data Poisoning"),
    ("V4", "Adversarial Robustness"),
    ("V2", "Preprocessing Attack Surface"),
    ("V3", "Data Validation Weaknesses"),
]

