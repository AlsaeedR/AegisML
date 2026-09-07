from typing import Any, Optional, Set


def extract_target_ids(test_targets: Any) -> Optional[Set[str]]:
    
    if not test_targets:
        return None

    if isinstance(test_targets, dict) and "vulnerabilities" in test_targets:
        items = test_targets["vulnerabilities"]
    elif isinstance(test_targets, list):
        items = test_targets
    else:
        return None

    ids = set()
    for item in items:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict) and "id" in item:
            ids.add(item["id"])

    return ids or None
