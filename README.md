# AegisML - Multi-Agent ML Security Auditor

AegisML is an agentic security audit system designed to analyze machine learning pipelines, evaluate ML-specific attack surfaces, and provide actionable security findings.

The system is organized into specialized collaborative agents:
- **Agent 1: Pipeline & Threat Modeling Agent** (Implemented): Performs static analysis of Python ML code, extracts pipeline execution graphs, derives deployment context and threat models following the **NIST AI 100-2e2025** adversarial ML taxonomy, and identifies the four MVP vulnerability classes with remediation recommendations.
- **Agent 2: Vulnerability Testing Agent** (Planned Next Phase): Performs dynamic testing (data poisoning checks, ART adversarial robustness attacks) and provides empirical test results.
- **Agent 3: Risk Scoring & Reporting Agent** (Planned Next Phase): Combines the static threat model and dynamic test findings to compute risk severity and produce the final audit report.

> Note: Numerical risk scoring is intentionally omitted in Agent 1, as accurate risk scores require empirical attack metrics from Agent 2.

---

## Core MVP Vulnerabilities

Agent 1 analyzes pipelines for four high-impact vulnerability classes:
1. **Data Poisoning (V1)**: Ingestion of unverified or tampered training data, lack of provenance checks.
2. **Preprocessing Attack Surface (V2)**: Unsanitized inputs, unconstrained vectorizers, and exploitable feature engineering.
3. **Data Validation Weaknesses (V3)**: Missing schema, type, range, or null checks at trust boundaries.
4. **Adversarial Robustness (V4)**: Susceptibility of trained models to evasion attacks or input perturbations.

---

## Project Structure

```text
AegisML/
├── src/
│   └── agents/
│       ├── pipeline_agent/
│       │   ├── schemas.py            # NIST AI 100-2e2025 Pydantic schemas
│       │   ├── code_parser.py        # Generic AST pipeline extraction
│       │   ├── networkx_utils.py     # Graph construction & topology analysis
│       │   ├── tools.py              # Modular agent tools & Pydantic validation
│       │   ├── steps.py              # LLM prompt chains & reasoning logic
│       │   ├── graph.py              # LangGraph StateGraph & self-correction loop
│       │   ├── state.py              # Agent state definitions (TypedDict)
│       │   └── pipeline_agent.py     # Agent 1 execution entry point
│       └── testing_agent/
│           ├── state.py              # Agent state definitions (TypedDict)
│           ├── loader.py             # Trained model + CSV dataset loading
│           ├── poisoning_test.py     # Test 1: Data Poisoning (label-flip + ART SVM attack)
│           ├── adversarial_test.py   # Test 2: Adversarial Robustness (ART HopSkipJump evasion)
│           ├── graph.py              # LangGraph StateGraph & evidence aggregation
│           └── testing_agent.py      # Agent 2 execution entry point
├── data/                             # Sample target pipelines + generated model/dataset
├── generate_model_and_dataset.py     # Utility: trains sample model + dataset for Agent 2
├── run_agent2.py                     # Test execution harness for Agent 2
├── main.py                           # Test execution harness for Agent 1
├── requirements.txt                  # Project dependencies
└── README.md
```

> Important: The `data/` folder contains external sample pipelines used for evaluation and testing. The core AegisML system is fully decoupled from this folder and can analyze any arbitrary Python ML pipeline code string.

---

## Getting Started

### 1. Get the Code

**For first-time setup (new team members):**
```bash
git clone https://github.com/AlsaeedR/AegisML.git
cd AegisML
```

**For existing team members (updating local branch):**
```bash
git pull origin main
```

---

### 2. Create and Activate a Virtual Environment

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell)**
```powershell
.\.venv\Scripts\activate
```

**Windows (Command Prompt)**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

---

### 3. Install Dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

To regenerate the entire file with all current dependencies:

```bash
pip freeze > requirements.txt
```

---

### 4. Configure Environment Variables

A template `.env.example` is provided. Copy it to create your local `.env` file:

**Windows (PowerShell)**
```powershell
copy .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

*Optional:* You can also configure the model (defaults to `gpt-4o-mini`):
```env
OPENAI_MODEL_NAME=gpt-4o-mini
```

---

### 5. Configure the Target ML Pipeline

Agent 1 takes the source code of any Python ML pipeline. In `main.py`, the target file is specified by `target_path`:

```python
# Target ML pipeline to audit
target_path = os.path.join("data", "21011088.py")
```

You can change this path to point to any Python pipeline file you want to audit.

---

### 6. Run Agent 1

Execute the agent harness:

```bash
python main.py
```

### Execution Flow:
1. **Perception**: Python AST and NetworkX tools parse the target code to build the pipeline graph (ingestion, preprocessing, validation, models, persistence) and identify entry/sink trust boundaries.
2. **Threat Modeling**: The LLM establishes deployment context and threats aligned with the **NIST AI 100-2e2025** taxonomy.
3. **Pydantic Validation & Reflection**: The generated threat model is validated against strict Pydantic schemas. If schema or taxonomy validation fails, diagnostic errors trigger an autonomous self-repair loop (up to 3 retries).
4. **Vulnerability Reasoning**: The LLM evaluates the four MVP vulnerabilities and formulates concrete code-level remediation recommendations.
5. **Output**: The validated pipeline graph, threat model, and vulnerability findings are printed to stdout as formatted JSON.



### 7. Run Agent 2

Generate a sample trained model + dataset (one-time setup):
```bash
python generate_model_and_dataset.py
```

Execute the agent harness:
```bash
python run_agent2.py
```

### Execution Flow:
1. **Loading**: The trained model (`.pkl`) and CSV dataset are loaded via joblib and pandas, and a sample of text/label pairs is prepared for testing.
2. **Data Poisoning Test**: A shadow clone of the model is retrained on a label-flipped copy of the training data (15% flip rate); the accuracy drop between the clean and poisoned retrain quantifies susceptibility to training-data corruption. Where the classifier is an SVM, ART's `PoisoningAttackSVM` additionally crafts a targeted poisoning point to confirm decision-boundary sensitivity.
3. **Adversarial Robustness Test**: The vectorizer/classifier pair is split out of the pipeline, the classifier is wrapped with ART's `SklearnClassifier`, and ART's `HopSkipJump` black-box evasion attack perturbs TF-IDF vectors of a text sample; the fraction of flipped predictions quantifies robustness.
4. **Evidence Aggregation**: Each test's status (`vulnerable` / `not_vulnerable`), severity, and evidence are aggregated into a structured results list, with placeholder entries (`not_tested`) for the two vulnerability categories not yet implemented (Preprocessing Attack Surface, Data Validation Weaknesses).
5. **Output**: The structured test results (`vulnerability_id`, `status`, `severity`, `evidence`) are printed to stdout as formatted JSON, ready to feed back into Agent 1's Risk Scoring Engine or forward to Agent 3's Report Generator.
