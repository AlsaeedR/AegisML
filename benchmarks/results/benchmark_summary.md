# AegisML Empirical Evaluation Summary: Operational & Agent Performance

Operational telemetry across standardized ML pipeline benchmark cases from `benchmarks/pipelines/`, measuring automated multi-agent runtime, throughput, token volume, and dual-component cost of automation.

## 1. Operational Efficiency, Throughput & Cost of Automation Metrics

| Case ID | Pipeline Scenario | Target Script | LOC | Runtime | Throughput | Total Tokens | Cost (LLM) | Cost (Compute) | Total Cost | Warm Cache |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CASE-01** | `all_defended` | `pipeline.py` | 327 | 65.39s | 300.0 LOC/min | 65,319 | $0.0139 | $0.00090 | **$0.0139** | 0.0s |
| **CASE-02** | `v1_data_poisoning` | `train.py` | 76 | 56.84s | 80.2 LOC/min | 58,244 | $0.0127 | $0.00068 | **$0.0127** | 0.01s |
| **CASE-03** | `v2_preprocessing` | `pipeline.py` | 98 | 58.7s | 100.2 LOC/min | 53,268 | $0.0118 | $0.00068 | **$0.0118** | 0.0s |
| **CASE-04** | `v3_data_validation` | `main.py` | 229 | 84.63s | 162.4 LOC/min | 91,818 | $0.0190 | $0.00094 | **$0.0190** | 0.0s |
| **CASE-05** | `v4_adversarial` | `main.py` | 374 | 71.01s | 316.0 LOC/min | 96,728 | $0.0192 | $0.00085 | **$0.0192** | 0.01s |
| **CASE-06** | `v4_v1_compound` | `main.py` | 386 | 75.37s | 307.3 LOC/min | 97,435 | $0.0193 | $0.00086 | **$0.0193** | 0.0s |
| **CASE-07** | `inference_only_serving` | `pipeline.py` | 21 | 18.72s | 67.3 LOC/min | 24,100 | $0.0050 | $0.00000 | **$0.0050** | 0.01s |
| **CASE-08** | `completely_unhardened_baseline` | `pipeline.py` | 35 | 7.68s | 273.3 LOC/min | 11,007 | $0.0021 | $0.00000 | **$0.0021** | 0.0s |

## 2. Multi-Agent Goal Completion & Quality Evaluation

| Case ID | Agent 1 Hallucination | Agent 1 Control Recall | Agent 2 Plan Coverage | Agent 2 Budget Adherence | Agent 2 Contradiction | Agent 3 NIST Invariant | Agent 3 Actionability | Trajectory (15 states) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CASE-01** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |
| **CASE-02** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |
| **CASE-03** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |
| **CASE-04** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |
| **CASE-05** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |
| **CASE-06** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |
| **CASE-07** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |
| **CASE-08** | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | Pass | 100.0% | Valid (15/15) |

## 3. Path-Level Trajectory & Observability

| Case ID | Trajectory Valid | Self-Correction Retries | Node Trajectory Sequence |
| :--- | :---: | :---: | :--- |
| **CASE-01** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |
| **CASE-02** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |
| **CASE-03** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |
| **CASE-04** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |
| **CASE-05** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |
| **CASE-06** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |
| **CASE-07** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |
| **CASE-08** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threa...` |

---
Generated automatically by `benchmarks/metrics_reporter.py`.
