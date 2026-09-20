# AegisML Empirical Evaluation Summary: Pillar 1 & Pillar 2

Evaluation results across standardized ML pipeline benchmark cases from `benchmark_pipelines/`, comparing automated multi-agent performance against the realistic expert human baseline (2.5 hours, 9,000s @ $100/hr).

## 1. Efficiency, Speedup & Cost Metrics

| Case ID | Pipeline Scenario | Target Script | LOC | AegisML Runtime | Manual Baseline | Speedup (%) | Throughput (LOC/min) | Cache Run | Cost Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CASE-01** | `all_defended` | `pipeline.py` | 327 | 24.12s | 9000.0s (2.5h) | **99.73%** | 813.3 LOC/min | 99.95% faster | **99.994%** |
| **CASE-02** | `v1_data_poisoning` | `train.py` | 76 | 8.22s | 9000.0s (2.5h) | **99.91%** | 555.0 LOC/min | 99.91% faster | **99.994%** |
| **CASE-03** | `v2_preprocessing` | `pipeline.py` | 98 | 7.97s | 9000.0s (2.5h) | **99.91%** | 738.2 LOC/min | 99.94% faster | **99.994%** |
| **CASE-04** | `v3_data_validation` | `main.py` | 229 | 9.83s | 9000.0s (2.5h) | **99.89%** | 1397.3 LOC/min | 99.92% faster | **99.994%** |
| **CASE-05** | `v4_adversarial` | `main.py` | 374 | 9.51s | 9000.0s (2.5h) | **99.89%** | 2360.9 LOC/min | 99.92% faster | **99.994%** |
| **CASE-06** | `v4_v1_compound` | `main.py` | 386 | 9.53s | 9000.0s (2.5h) | **99.89%** | 2429.4 LOC/min | 99.92% faster | **99.994%** |
| **CASE-07** | `inference_only_serving` | `pipeline.py` | 21 | 7.77s | 9000.0s (2.5h) | **99.91%** | 162.1 LOC/min | 99.88% faster | **99.994%** |
| **CASE-08** | `completely_unhardened_baseline` | `pipeline.py` | 35 | 8.0s | 9000.0s (2.5h) | **99.91%** | 262.6 LOC/min | 99.91% faster | **99.994%** |

## 2. LLM Evaluation & Docker Execution Status

| Case ID | LLM Generation Status | Dynamic Docker Execution | Plan Adherence | Trajectory Valid |
| :--- | :--- | :--- | :--- | :--- |
| **CASE-01** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |
| **CASE-02** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |
| **CASE-03** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |
| **CASE-04** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |
| **CASE-05** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |
| **CASE-06** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |
| **CASE-07** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |
| **CASE-08** | LLM is not available (Skipped) | 100% Executed in Container | 100.0% (Docker dispatch) | Valid |

> [!NOTE]
> **LLM Transparency Notice**: When external LLM APIs are unconfigured or quota-limited, AegisML executes full static AST parsing, Docker container adversarial attacks, and deterministic risk scoring, explicitly skipping LLM narrative generation without fabricating synthetic responses.

## 3. Path-Level Trajectory & Observability

| Case ID | Trajectory Valid | Self-Correction Retries | Node Trajectory Sequence |
| :--- | :--- | :--- | :--- | :--- |
| **CASE-01** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |
| **CASE-02** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |
| **CASE-03** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |
| **CASE-04** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |
| **CASE-05** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |
| **CASE-06** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |
| **CASE-07** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |
| **CASE-08** | Yes | 0 | `extract_pipeline -> execute_dynamic_sandbox -> synthesize...` |

---
Generated automatically by `benchmarks/metrics_reporter.py`.
