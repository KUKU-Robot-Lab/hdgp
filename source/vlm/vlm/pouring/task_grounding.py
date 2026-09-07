from __future__ import annotations

import json
from typing import Any

from .contracts import TaskSpecification

_EXPECTED_KEYS = {"task", "source_id", "target_id", "nominal_plan", "allowed_skills"}


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract exactly one JSON object from plain or fenced model output."""
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Qwen response does not contain a JSON object")
    if cleaned[:start].strip():
        raise ValueError("Qwen response must contain only one JSON object")

    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(cleaned, start)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen response contains invalid JSON") from exc
    if cleaned[end:].strip():
        raise ValueError("Qwen response must contain exactly one JSON object")
    if not isinstance(value, dict):
        raise ValueError("Qwen response root must be an object")
    return value


def _string_array(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} must be an array of non-empty strings")
    return tuple(value)


def parse_task_specification(text: str) -> TaskSpecification:
    """Convert untrusted Qwen output into the stable task contract."""
    data = extract_json_object(text)
    if set(data) != _EXPECTED_KEYS:
        raise ValueError(f"TaskSpecification keys must be exactly {sorted(_EXPECTED_KEYS)}")
    if not all(isinstance(data[key], str) for key in ("task", "source_id", "target_id")):
        raise ValueError("task, source_id, and target_id must be strings")
    return TaskSpecification(
        task=data["task"],
        source_id=data["source_id"],
        target_id=data["target_id"],
        nominal_plan=_string_array(data, "nominal_plan"),
        allowed_skills=_string_array(data, "allowed_skills"),
    )
