# AegisML — ML Pipeline Security Analysis Agent

AegisML is an agentic security analysis system built using LangGraph.  
It takes a Python ML pipeline as input and produces:

- **Pipeline Graph** (nodes + edges)
- **Threat Model** (assets, entry points, threats, mitigations)
- **Four MVP Vulnerabilities**
  - Data Poisoning
  - Preprocessing Attack Surface
  - Data Validation Weaknesses
  - Adversarial Robustness

Agent 1 is fully implemented.  
Agent 2 (testing agent) will be added next.

---

## 1. Clone the Repository

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

---

## 2. Create a virtual environment

Create an isolated Python environment to avoid dependency conflicts.

```bash
python -m venv .venv
```
Activate the environment:
**Windows (PowerShell / CMD)**
```powershell
.\.venv\Scripts\activate
```
**macOS / Linux**
```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

Install required packages from requirements.txt.

```bash
pip install -r requirements.txt
```

To regenerate the entire file with all current dependencies:
```bash
pip freeze > requirements.txt
```

---

## 4. Environment variables

A template .env.example is provided. Copy it to create your working .env file and fill in required values.

**Windows (PowerShel)**
```powershell
copy .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```
--- 

## 5. Configure the input file for Agent 1

Agent 1 expects a target Python file to analyze. Edit main.py (or the configured entry) to point to the Python pipeline file you want analyzed. Example placeholder in main.py:

```python
TARGET_PYTHON_FILE = "path/to/your_pipeline.py"
```

---

## 6. Run Agent 1 (pipeline extraction + threat model + vulnerabilities)

```python
python main.py
```

Expected behavior:

The agent parses the target Python file and extracts a pipeline graph (nodes and edges).

The agent builds a threat model (assets, entry points, threats, mitigations).

The agent identifies the four MVP vulnerabilities.

Results are printed or written as JSON (check main.py for output location).