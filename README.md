# AegisML - Multi-Agent ML Security Auditor

AegisML is an agentic security audit system designed to analyze machine learning pipelines, evaluate ML-specific attack surfaces, perform cognitive empirical adversarial testing inside isolated sandboxes, and generate evidence-informed security reports.

The system is organized into three specialized collaborative agents:
- **Agent 1: Pipeline & Threat Modeling Agent (SAST)**: Performs static AST analysis of Python ML code, extracts pipeline topology graphs using NetworkX, derives deployment context and threat models following the **NIST AI 100-2e2025** adversarial ML taxonomy, and identifies theoretical vulnerability classes.
- **Agent 2: Vulnerability Testing Agent (DAST)**: Operates as a cognitive penetration testing engine using the **Bounded Agency Pattern**. It leverages LangChain tools within LangGraph nodes to formulate an attack strategy, dispatches tests into an air-gapped **Zero-Trust Docker Sandbox**, and conducts post-attack forensic root-cause analysis.
- **Agent 3: Risk Scoring & Reporting Agent (Governance)**: Serves as the centralized mathematical scoring authority. It correlates Agent 1's static baseline with Agent 2's empirical testing evidence, computes NIST AI 100-2 risk scores, synthesizes evidence-informed remediation recommendations, generates executive summaries, and formats comprehensive audit reports.

---

## Core Vulnerability Classes (MVP)

AegisML evaluates ML pipelines across four primary vulnerability categories:

1. **Data Poisoning (V1)**: Ingestion of unverified training data without cryptographic provenance or outlier filtering. Evaluated empirically via label-flipping degradation benchmarks and decision-boundary poisoning.
2. **Preprocessing Attack Surface (V2)**: Unsanitized input cleaning, regex vulnerabilities, and unconstrained tokenization. Evaluated empirically via malformed, long, and edge-case string fuzzing.
3. **Data Validation Weaknesses (V3)**: Missing schema, type, range, or null checks before training. Evaluated empirically by injecting duplicate records and missing values into training data and measuring accuracy degradation.
4. **Adversarial Robustness (V4)**: Susceptibility to evasion attacks at inference time. Evaluated empirically using IBM ART's `HopSkipJump` decision-boundary black-box attack under a strict relative $L_2$ perturbation budget ($\le 0.35 - 0.50$).

---

## System Architecture

```mermaid
flowchart TD
    subgraph Core ["Centralized Core Engine (src/core/)"]
        LLMFactory["Centralized LLM Factory\n(src/core/llm.py)"]
    end

    subgraph A1 ["Agent 1: Pipeline & Threat Modeling (SAST)"]
        Code["Python Source Code"] --> AST["AST Parser & Topology Graph"]
        AST --> ThreatModel["NIST AI 100-2e2025 Threat Model"]
        ThreatModel --> Vulns["Qualitative Vulnerability Findings"]
    end

    subgraph A2 ["Agent 2: Cognitive Testing & Sandbox (DAST)"]
        Meta["prepare_metadata\n(Safe Host Metadata Inspection)"] --> Reason["reason_strategy\n(LLM + LangChain Planning Tools)"]
        
        subgraph Tools ["LangChain Tools in Strategy Node"]
            T1["@tool inspect_dataset_profile"]
            T2["@tool calculate_perturbation_budget"]
            T3["@tool resolve_threat_surface"]
        end
        Reason <--> Tools

        Reason --> Gate{"Docker Available?"}
        Gate -- "YES" --> Docker["execute_sandbox\n(Air-Gapped Docker Sandbox)"]
        Gate -- "NO" --> FailClosed["Fail-Closed: Mark Tests Skipped\n(Zero-Trust Policy Enforced)"]

        Docker --> Forensic["forensic_diagnosis\n(LLM Forensic Root-Cause Analysis)"]
        FailClosed --> Forensic
        Forensic --> Agg["aggregate_results\n(Telemetry & Structured Results)"]
    end

    subgraph A3 ["Agent 3: Risk Scoring & Reporting (Governance)"]
        Correlate["node_correlate_findings\n(Static Context + Empirical Evidence)"] --> Scoring["score_finding_tool\n(Impact × Likelihood)"]
        Scoring --> Correlation["Status: Confirmed, False Positive, Hidden Risk"]
        Correlation --> Builder["compile_audit_report_tool\n(Evidence-Informed Recommendations)"]
        Builder --> Validation["validate_audit_report_schema\n(Pydantic Schema Validation)"]
        Validation --> FinalReport["Validated Audit Report (JSON / PDF)"]
    end

    LLMFactory -.-> A1
    LLMFactory -.-> A2
    LLMFactory -.-> A3

    Vulns --> Reason
    Vulns --> Correlate
    Agg --> Correlate
```

---

## Zero-Trust Sandbox Isolation

AegisML treats target pipelines and trained models as untrusted code:
* **Host Protection**: The host system **never** calls `pickle.load()` on untrusted models or executes target code in-process. This eliminates Remote Code Execution (RCE) via `__reduce__` deserialization exploits.
* **Air-Gapped Container**: Sandboxed tests run inside an isolated Docker container with `--network none`, strict memory caps (`--memory="4g"`), CPU limits (`--cpus="2.0"`), and read-only data volume mounts (`:ro`).
* **Fail-Closed Policy**: If the Docker daemon is offline, dynamic execution halts safely and tests are marked as `Unverified (Sandbox Offline)`. The host is never compromised.
* **Developer Override**: For offline development, explicitly setting `AEGISML_ALLOW_INSECURE_LOCAL_TESTING=true` enables in-process execution accompanied by security warnings.

---

## Project Structure

```text
AegisML/
├── api.py                            # FastAPI REST service exposing /audit and health checks
├── app.py                            # Streamlit interactive dashboard UI with PDF generation
├── main.py                           # CLI entry point running Agent 1 -> Agent 2 -> Agent 3 end-to-end
├── generate_model_and_dataset.py     # Utility: generates sample evaluation fixtures
├── requirements.txt                  # Python project dependencies
├── data/                             # Target pipeline and evaluation dataset
│   ├── pipeline.py                   # Sample target pipeline to audit
│   └── dataset.csv                   # Target evaluation dataset
├── docker/                           # Containerized sandbox isolation
│   └── sandbox.Dockerfile            # Hardened, non-root air-gapped sandbox image
├── ui/                               # Streamlit UI dashboard rendering modules and CSS
│   ├── dashboard.py                  # Component renderers for pipeline topology & findings
│   └── styles.css                    # Custom dashboard styling
└── src/
    ├── core/                         # Centralized platform services
    │   └── llm.py                    # Unified LLM factory and environment configuration
    └── agents/
        ├── pipeline_agent/           # Agent 1: Static AST & Threat Modeling
        │   ├── code_parser.py        # AST pipeline structural extraction
        │   ├── networkx_utils.py     # NetworkX topology analysis
        │   ├── schemas.py            # Pydantic v2 schemas for NIST threat modeling
        │   ├── steps.py              # Prompt chains & indirect injection sanitization
        │   ├── tools.py              # AST extraction and schema validation tools
        │   ├── graph.py              # LangGraph StateGraph & self-repair loop
        │   └── pipeline_agent.py     # Execution entry point
        ├── testing_agent/            # Agent 2: Cognitive Dynamic Penetration Testing
        │   ├── schemas.py            # Pydantic schemas for attack strategy & forensics
        │   ├── tools.py              # LangChain tools for dataset profiling & budgets
        │   ├── state.py              # TestingAgentState definition
        │   ├── sandbox_runner.py     # Host-side Fail-Closed Zero-Trust Docker dispatcher
        │   ├── sandbox_worker.py     # In-container test execution worker
        │   ├── loader.py             # Artifact loading utilities (used inside sandbox)
        │   ├── poisoning_test.py     # V1 empirical poisoning & label-flipping tests
        │   ├── preprocess_test.py    # V2 preprocessing edge-case & fuzzing checks
        │   ├── validation_test.py    # V3 data corruption retraining degradation tests
        │   ├── adversarial_test.py   # V4 IBM ART HopSkipJump evasion attack
        │   ├── graph.py              # 5-node cognitive LangGraph workflow
        │   └── testing_agent.py      # Execution entry point
        └── reporting_agent/          # Agent 3: Evidence-Informed Risk & Reporting
            ├── schemas.py            # Pydantic v2 schemas for audit reports and findings
            ├── tools.py              # Scoring, compilation, and schema validation tools
            ├── risk_scoring.py       # Centralized quantitative risk calculation engine
            ├── report_builder.py     # Evidence-informed recommendation engine & summary
            ├── graph.py              # LangGraph correlation and reporting workflow
            └── reporting_agent.py    # Execution entry point
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AlsaeedR/AegisML.git
cd AegisML
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to create your local `.env` file:

**Windows (PowerShell)**:
```powershell
Copy-Item .env.example .env
```

**macOS / Linux**:
```bash
cp .env.example .env
```

Configure your OpenAI credentials in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
```

### 5. Build the Sandbox Container (Optional, for Docker Execution)
To run empirical tests in the isolated container:
```bash
docker build -t aegisml-sandbox:latest -f docker/sandbox.Dockerfile .
```

---

## Execution Modes

AegisML can be executed via CLI, as a headless FastAPI REST service, or as an interactive Streamlit web dashboard.

### Mode 1: Command-Line Interface (CLI)

Run the end-to-end multi-agent pipeline from the terminal:

```bash
python main.py
```

**Execution Pipeline:**
1. **Agent 1**: Audits `data/pipeline.py`, builds the topology graph, establishes the NIST threat model, and detects vulnerabilities.
2. **Agent 2**: Reasons on dataset properties, formulates an attack strategy plan via LangChain tools, executes tests inside the air-gapped Docker sandbox (or safely fails closed if Docker is offline), and performs forensic diagnosis.
3. **Agent 3**: Correlates static context with dynamic evidence, computes final risk scores, synthesizes evidence-informed recommendations, and outputs the validated JSON report to stdout.

---

### Mode 2: FastAPI Backend Service

Run the REST API backend using `uvicorn`:

```bash
uvicorn api:app --reload --port 8000
```

* **Health Check**: `GET http://127.0.0.1:8000/` (returns `{"message": "AegisML API is running."}`)
* **Interactive Swagger UI**: Navigate to `http://127.0.0.1:8000/docs` in your browser.
* **Audit Endpoint (`POST /audit`)**: Accepts `multipart/form-data`:
  * `pipeline_file`: Python script (`.py`)
  * `model_file`: Serialized model (`.pkl`)
  * `dataset_file`: Evaluation dataset (`.csv`)
  * `text_column`: Column name for text features (default: `text`)
  * `label_column`: Column name for target labels (default: `label`)

Example cURL request:
```bash
curl -X POST "http://127.0.0.1:8000/audit" \
  -F "pipeline_file=@data/pipeline.py" \
  -F "model_file=@data/model.pkl" \
  -F "dataset_file=@data/dataset.csv" \
  -F "text_column=text" \
  -F "label_column=label"
```

---

### Mode 3: Streamlit Web Dashboard

The web dashboard provides an interactive interface for uploading pipeline assets, inspecting graphs, viewing correlation analyses, and downloading executive PDF reports.

#### Running Full Stack (API + Dashboard):

**Terminal 1 (Backend API):**
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2 (Streamlit UI):**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

#### Dashboard Features:
* **Asset Upload Screen**: Upload pipeline `.py`, model `.pkl`, and dataset `.csv` with clear visual status indicators.
* **Overview Tab**: Displays overall risk score (out of 10), severity badge, confirmed findings count, and LLM executive summary.
* **Pipeline & Findings Tab**: Visualizes pipeline nodes, affected components, dynamic test metrics, and adaptive finding card framing (`Root cause` + `Suggested fix` for Confirmed Risks vs. `Theoretical concern` + `Verification outcome` for False Positives).
* **Governance Mapping Tab**: Maps empirical findings to NIST AI Risk Management framework controls and highlights False Positives vs. Hidden Risks.
* **Full Report Tab**: Generates and downloads a multi-page security audit PDF document built with ReportLab.

---

## Risk Scoring Methodology

AegisML centralizes risk quantification in **Agent 3**:

$$\text{Theoretical Risk} = \frac{\text{Base Impact} \times \text{Static Likelihood}}{10}$$

$$\text{Final Evidence-Informed Risk} = \frac{\text{Base Impact} \times \text{Evidence-Adjusted Likelihood}}{10}$$

* **Confirmed Risk**: Agent 1 detected the threat and Agent 2 empirically confirmed vulnerability.
* **False Positive (Mitigated)**: Agent 1 flagged High/Critical static risk, but Agent 2 proved the model was resilient (`not_vulnerable`). Likelihood is scaled down to residual baseline ($2.0/10$).
* **Hidden Risk**: Agent 1 rated static risk Low, but Agent 2 successfully attacked the model (`vulnerable`). Likelihood is scaled up.
* **Unverified**: Empirical evidence was unavailable, incomplete, or skipped under the Zero-Trust policy. Static likelihood estimate is retained.
