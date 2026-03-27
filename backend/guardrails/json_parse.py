# backend/guardrails/json_parse.py
import json
import re
from typing import Optional, Tuple
from pydantic import ValidationError
from guardrails.output_schema import LLMResponse

# Prefer fenced JSON first, then fallback to first JSON-ish object
_FENCED_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_JSON_OBJ_RE = re.compile(r"(\{.*?\})", re.DOTALL)

def extract_json(text: str) -> Optional[str]:
    text = (text or "").strip()

    # 1) If already clean JSON
    if text.startswith("{") and text.endswith("}"):
        return text

    # 2) ```json ... ```
    m = _FENCED_JSON_RE.search(text)
    if m:
        return m.group(1).strip()

    # 3) first { ... } block (non-greedy)
    m = _FIRST_JSON_OBJ_RE.search(text)
    return m.group(1).strip() if m else None


def parse_and_validate(text: str) -> Tuple[Optional[LLMResponse], Optional[str]]:
    js = extract_json(text)
    if not js:
        return None, "No JSON object found"

    try:
        data = json.loads(js)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"

    try:
        obj = LLMResponse.model_validate(data)
        return obj, None
    except ValidationError as e:
        return None, f"Schema validation error: {e}"