########## Agentic Helpers ##########
# JSON parsing and output scrubbing borrowed from the classroom notebooks.

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.S | re.I)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.S | re.I)


def strip_think(text: str) -> str:
    """Remove DeepSeek-style <think> blocks to keep outputs clean."""

    return THINK_BLOCK_RE.sub("", text or "").strip()


def _first_code_block(text: str) -> Optional[str]:
    """Return the first fenced code block content if present."""

    match = CODE_FENCE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return None


def _extract_balanced_json(text: str) -> Optional[str]:
    """Find the first balanced {...} structure inside text."""

    payload = text or ""
    start = payload.find("{")
    if start == -1:
        return None
    depth = 0
    for index, char in enumerate(payload[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return payload[start : index + 1]
    return None


def parse_json_loose(raw: str) -> Optional[Dict[str, Any]]:
    """Try several strategies to tease out JSON from a chatty LLM output."""

    cleaned = strip_think(raw)

    fence = _first_code_block(cleaned)
    if fence:
        try:
            return json.loads(fence)
        except Exception:
            pass

    balanced = _extract_balanced_json(cleaned)
    if balanced:
        try:
            return json.loads(balanced)
        except Exception:
            pass

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    has_braces = first != -1 and last != -1 and last > first
    if has_braces:
        fragment = cleaned[first : last + 1]
        try:
            return json.loads(fragment)
        except Exception:
            pass

    return None
