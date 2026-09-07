import pandas as pd
import numpy as np
from typing import Any, Dict

def _check_missing(df):
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        return {"status": "found", "severity": "Medium", "details": nulls[nulls>0].to_dict()}
    return {"status": "not_found"}

def _check_infinite(df):
    inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf > 0:
        return {"status": "found", "severity": "High", "details": f"{int(inf)} inf values"}
    return {"status": "not_found"}

def _check_duplicates(df):
    dup = df.duplicated().sum()
    if dup > 0:
        return {"status": "found", "severity": "Medium", "details": f"{int(dup)} duplicates"}
    return {"status": "not_found"}

def _check_imbalance(df):
    ratio = df[df.columns[-1]].value_counts(normalize=True).max()
    if ratio > 0.8:
        return {"status": "found", "severity": "Low", "details": f"{ratio:.0%} in one class"}
    return {"status": "not_found"}

def run_validation_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    path = state.get("dataset_path")
    if not path:
        return {"validation_evidence": {"status": "error", "summary": "No dataset path"}}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"validation_evidence": {"status": "error", "summary": str(e)}}
    
    findings = [c for c in [_check_missing(df), _check_infinite(df), _check_duplicates(df), _check_imbalance(df)] if c["status"] == "found"]
    sev_map = {"High":3,"Medium":2,"Low":1}
    severity = max(findings, key=lambda x: sev_map.get(x.get("severity","Low"),0)).get("severity","Low").lower() if findings else "low"
    
    
    return {"validation_evidence": {
        "vulnerability_id": "V3",
        "vulnerability_name": "Data Validation Weaknesses",
        "status": "vulnerable" if findings else "not_vulnerable",
        "severity": severity,
        "evidence": { 
            "missing_values": _check_missing(df),
            "infinite_values": _check_infinite(df),
            "duplicate_rows": _check_duplicates(df),
            "class_imbalance": _check_imbalance(df),
        },
        "summary": f"{len(findings)} issue(s)" if findings else "All clean"
    }}