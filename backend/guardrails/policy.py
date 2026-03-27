# backend/guardrails/policies.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

def enforce_grounded_policy(
    answer: str,
    citations: List[Dict[str, Any]],
    retrieved_matches: List[Dict[str, Any]],
    allow_no_citations: bool = False,
) -> bool:
    """
    Enforces:
    1) Citations (source,page) must come from retrieved matches.
    2) If allow_no_citations=True, then empty citations are allowed (smalltalk/memory/general).

    retrieved_matches format (your code):
      [{"score":..., "metadata": {"source": "...", "page": 1, "text": "..."}}, ...]
    citations format:
      [{"source": "...", "page": 1}, ...]
    """

    if not citations:
        return bool(allow_no_citations)

    allowed: Set[Tuple[str, Optional[int]]] = set()
    for m in retrieved_matches:
        md = (m or {}).get("metadata") or {}
        src = md.get("source") or md.get("file") or "unknown"
        src = Path(str(src)).name
        page = md.get("page", None)
        try:
            page = int(page) if page is not None else None
        except Exception:
            page = None
        allowed.add((src, page))
        allowed.add((src, None))  # allow citation without page

    for c in citations:
        src = Path(str(c.get("source") or "")).name
        page = c.get("page", None)
        try:
            page = int(page) if page is not None else None
        except Exception:
            page = None

        if (src, page) not in allowed and (src, None) not in allowed:
            return False

    return True