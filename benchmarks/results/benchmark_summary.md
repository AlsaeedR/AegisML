# AegisML Empirical Evaluation Summary: Operational & Agent Performance

Evaluation results across standardized ML pipeline benchmark cases from `benchmarks/pipelines/`, comparing automated multi-agent performance against the realistic expert human baseline (2.5 hours, 9,000s @ $100/hr = $250.00).

## 1. Operational Efficiency, Throughput & Cost Metrics

| Case ID | Pipeline Scenario | Target Script | LOC | Runtime | Human Speedup | Throughput | Total Tokens | Automated Cost | Cost Reduction |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CASE-01** | `all_defended` | `pipeline.py` | 327 | 65.39s | **99.27%** | 300.0 LOC/min | 65,319 | $0.0139 | **99.9944%** |
| **CASE-02** | `v1_data_poisoning` | `train.py` | 76 | 56.84s | **99.37%** | 80.2 LOC/min | 58,244 | $0.0127 | **99.9949%** |
| **CASE-03** | `v2_preprocessing` | `pipeline.py` | 98 | 58.7s | **99.35%** | 100.2 LOC/min | 53,268 | $0.0118 | **99.9953%** |
| **CASE-04** | `v3_data_validation` | `main.py` | 229 | 84.63s | **99.06%** | 162.4 LOC/min | 91,818 | $0.0190 | **99.9924%** |
| **CASE-05** | `v4_adversarial` | `main.py` | 374 | 71.01s | **99.21%** | 316.0 LOC/min | 96,728 | $0.0192 | **99.9923%** |
| **CASE-06** | `v4_v1_compound` | `main.py` | 386 | 75.37s | **99.16%** | 307.3 LOC/min | 97,435 | $0.0193 | **99.9923%** |
| **CASE-07** | `inference_only_serving` | `pipeline.py` | 21 | 18.72s | **99.79%** | 67.3 LOC/min | 24,100 | $0.0050 | **99.998%** |
| **CASE-08** | `completely_unhardened_baseline` | `pipeline.py` | 35 | 7.68s | **99.91%** | 273.3 LOC/min | 11,007 | $0.0021 | **99.9992%** |
| **Mean / Agg** | **8 Pipeline Benchmark Suite** | **8 Files** | **1,546** | **54.79s** | **99.39%** | **200.8 LOC/min** | **62,240** | **$0.0129** | **99.9949%** |

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
| **Aggregate** | **0.0% (0 errors)** | **100.0% Recall** | **100.0% Coverage** | **100.0% Budget** | **0.0% Contradiction** | **100.0% Pass** | **100.0% Actionable** | **100.0% Valid (8/8)** |

## 3. Path-Level Trajectory & Observability

| Case ID | Trajectory Valid | Self-Correction Retries | Node Trajectory Sequence |
| :--- | :---: | :---: | :--- |
| **CASE-01** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |
| **CASE-02** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |
| **CASE-03** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |
| **CASE-04** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |
| **CASE-05** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |
| **CASE-06** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |
| **CASE-07** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |
| **CASE-08** | Yes | 0 | `extract_pipeline -> reason_threat_model -> validate_threat_model -> reason_vulnerabilities...` |

---
Generated automatically by `benchmarks/metrics_reporter.py`.
