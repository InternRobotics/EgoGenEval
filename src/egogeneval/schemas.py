"""Load and apply the public JSON schemas."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from .errors import ValidationError

SCHEMA_NAMES = {
    "manifest-row",
    "prediction-row",
    "result",
}


def load_schema(name: str) -> dict[str, Any]:
    if name not in SCHEMA_NAMES:
        raise ValidationError(f"unknown schema: {name}")
    resource = files("egogeneval").joinpath("resources", "schemas", f"{name}.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _error_path(error: JsonSchemaValidationError) -> str:
    return ".".join(str(component) for component in error.absolute_path)


def validate_instance(instance: Any, schema_name: str, *, label: str) -> None:
    validator = Draft202012Validator(load_schema(schema_name))
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    error = errors[0]
    path = _error_path(error)
    location = f"{label}.{path}" if path else label
    raise ValidationError(f"{location}: {error.message}")
