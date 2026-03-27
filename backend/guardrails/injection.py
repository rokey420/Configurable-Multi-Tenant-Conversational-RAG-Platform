# backend/guardrails/injection.py
import re
from typing import Pattern, List

# Patterns for prompt-injection / secret-exfiltration attempts
_INJECTION_REGEX: List[Pattern] = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+(instructions|rules)\b", re.I),
    re.compile(r"\b(reveal|show|print|dump)\s+(the\s+)?(system|developer)\s+(prompt|message|instructions)\b", re.I),
    re.compile(r"\b(system\s+prompt|developer\s+message|hidden\s+instructions)\b", re.I),
    re.compile(r"\b(api\s*key|access\s*token|secret\s*key|password|credentials)\b", re.I),
    re.compile(r"\b(\.env|environment\s+variables|os\.environ)\b", re.I),
    re.compile(r"\b(bypass|override)\s+(safety|policy|filters|guardrails)\b", re.I),
    re.compile(r"\bdo\s+anything\s+now\b", re.I),  # DAN style
    re.compile(r"\bact\s+as\b", re.I),
]

def looks_like_injection(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _INJECTION_REGEX)