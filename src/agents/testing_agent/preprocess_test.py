import ast
from typing import Any, Dict

def _scan_pickle(tree):
    return [{"type": "Pickle Deserialization", "severity": "Critical", "line": n.lineno} 
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) 
            and n.func.attr == "load" and "pickle" in ast.unparse(n.func.value).lower()]

def _scan_eval(tree):
    return [{"type": "Dynamic Execution", "severity": "Critical", "line": n.lineno} 
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) 
            and n.func.id in ["eval","exec","compile"]]

def _scan_shell(tree):
    return [{"type": "Shell Injection", "severity": "High", "line": n.lineno} 
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) 
            and n.func.attr in ["system","call"] and "os" in ast.unparse(n.func.value).lower()]

def run_preprocess_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    path = state.get("model_path")
    if not path:
        return {"preprocessing_evidence": {"status": "error", "summary": "No model path"}}
    try:
        with open(path) as f: tree = ast.parse(f.read())
    except Exception as e:
        return {"preprocessing_evidence": {"status": "error", "summary": str(e)}}
    
    risks = _scan_pickle(tree) + _scan_eval(tree) + _scan_shell(tree)
    sev_map = {"Critical":4,"High":3}
    severity = max(risks, key=lambda x: sev_map.get(x["severity"],0)).get("severity","Low").lower() if risks else "low"
    
    
    return {"preprocessing_evidence": {
        "vulnerability_id": "V2",
        "vulnerability_name": "Preprocessing Attack Surface",
        "status": "vulnerable" if risks else "not_vulnerable",
        "severity": severity,
        "evidence": { 
            "insecure_deserialization": _scan_pickle(tree),
            "dynamic_execution": _scan_eval(tree),
            "shell_command_injection": _scan_shell(tree),
        },
        "summary": f"{len(risks)} risk(s)" if risks else "All clean"
    }}