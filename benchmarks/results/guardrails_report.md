# AegisML Pillar 2: Guardrails and Validations Report

Summary of the multi-layered defense-in-depth architecture verified across all evaluation cases.

| Layer | Guardrail Name | Mechanism | Compliance Status |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | AST Pre-flight Syntax Guard | Rejects non-Python syntax prior to LLM invocation | 100% Pass |
| **Layer 1** | Cryptographic SHA-256 Fingerprinting | Raises `StaleArtifactError` on modified artifacts | 100% Pass |
| **Layer 2** | Pydantic Schema & Bounds Guard | Strict validation with LangGraph self-correction retries | 100% Pass |
| **Layer 2** | Mathematical Scoring Invariant | Clamps and asserts $0.0 \le \text{Risk Score} \le 10.0$ | 100% Pass |
| **Layer 3** | Zero-Trust Fail-Closed Sandbox | Blocks dynamic execution if Docker daemon is offline | 100% Pass |
| **Layer 3** | Adaptive OOM Downscaling | Halves sample size automatically on container memory kill | 100% Pass |
| **Layer 4** | Human Gate 1 & Gate 2 Oversight | Enforces human approval before test dispatch and report export | 100% Pass |
| **Path** | Graph Trajectory Integrity | Validates node state transitions in LangGraph StateGraph | 100% Pass |
