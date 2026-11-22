"""Validate metadata using the same JSON schemas shipped with the application."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator


@lru_cache(maxsize=4)
def schema_validator(name: str) -> Draft202012Validator:
    path = Path(__file__).resolve().parents[2] / "schemas" / name
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_schema(payload: object, name: str) -> None:
    error = next(schema_validator(name).iter_errors(payload), None)
    if error is not None:
        location = ".".join(map(str, error.absolute_path)) or "manifest"
        raise ValueError(f"스키마 검증 실패 ({location}): {error.message}")


def read_json_bytes(raw: bytes) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON에 유한하지 않은 숫자가 있습니다: {value}")

    def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"중복 JSON 키: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
