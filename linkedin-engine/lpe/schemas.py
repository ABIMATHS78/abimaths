"""JSON schemas for template data, a forgiving validator, and the subset the Claude API accepts.

The API's structured outputs need `additionalProperties: false` on every object and don't support
length/size constraints, so counts live in descriptions. Our own `default` keys are stripped before
a schema is sent.
"""

from __future__ import annotations

import copy


def string(description: str = "", enum: list[str] | None = None, default=None) -> dict:
    schema: dict = {"type": "string"}
    if description:
        schema["description"] = description
    if enum:
        schema["enum"] = list(enum)
    if default is not None:
        schema["default"] = default
    return schema


def integer(description: str = "", default=None) -> dict:
    schema: dict = {"type": "integer", "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def number(description: str = "") -> dict:
    return {"type": "number", "description": description}


def boolean(description: str = "", default=None) -> dict:
    schema: dict = {"type": "boolean", "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def array(items: dict, description: str = "") -> dict:
    schema: dict = {"type": "array", "items": items}
    if description:
        schema["description"] = description
    return schema


def obj(properties: dict, required: list[str] | None = None, description: str = "") -> dict:
    schema: dict = {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def api_schema(schema: dict) -> dict:
    """A copy safe to send as a structured-output schema.

    Structured outputs allow at most 24 optional properties per request, and each optional property
    makes the compiled grammar larger, so every property is sent as required: Claude fills the ones a
    post doesn't need with an empty string, an empty list or the default choice, which the templates
    treat as absent. Our private keys are removed."""
    schema = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            for key in ("default", "examples"):
                node.pop(key, None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


def optional_count(schema: dict) -> int:
    """How many properties the API would count as optional (should be 0 after api_schema)."""
    count = 0
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            count += len(set(schema.get("properties", {})) - set(schema.get("required", [])))
        for value in schema.values():
            count += optional_count(value) if isinstance(value, (dict, list)) else 0
    elif isinstance(schema, list):
        count += sum(optional_count(v) for v in schema)
    return count


def _empty(schema: dict):
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    kind = schema.get("type")
    if kind == "string":
        return schema["enum"][0] if schema.get("enum") else ""
    if kind == "array":
        return []
    if kind == "object":
        return clean_data({}, schema)[0]
    if kind == "boolean":
        return False
    if kind in ("integer", "number"):
        return 0
    return None


def clean_data(data, schema: dict, path: str = "data") -> tuple:
    """Return (cleaned data, warnings). Never raises on bad content: templates always get usable input."""
    warnings: list[str] = []
    kind = schema.get("type")

    if kind == "object":
        if not isinstance(data, dict):
            warnings.append(f"{path}: expected an object")
            data = {}
        out = {}
        props = schema.get("properties", {})
        for key, value in data.items():
            if key not in props:
                warnings.append(f"{path}.{key}: not part of this template; ignored")
                continue
            out[key], sub = clean_data(value, props[key], f"{path}.{key}")
            warnings += sub
        for key, sub_schema in props.items():
            if key not in out:
                if key in schema.get("required", []):
                    warnings.append(f"{path}.{key}: missing")
                    out[key] = _empty(sub_schema)
                elif "default" in sub_schema:
                    out[key] = copy.deepcopy(sub_schema["default"])
        return out, warnings

    if kind == "array":
        if data is None:
            return [], warnings
        if not isinstance(data, list):
            warnings.append(f"{path}: expected a list")
            data = [data]
        out = []
        for index, item in enumerate(data):
            value, sub = clean_data(item, schema.get("items", {}), f"{path}[{index}]")
            out.append(value)
            warnings += sub
        return out, warnings

    if kind == "string":
        if data is None:
            data = ""
        if not isinstance(data, str):
            data = str(data)
        if schema.get("enum") and data not in schema["enum"]:
            # Structured outputs can return an enum value in a different case; match it case-insensitively.
            match = next((value for value in schema["enum"] if value.lower() == data.strip().lower()), None)
            if match is not None:
                data = match
            else:
                fallback = schema.get("default", schema["enum"][0])
                if data:
                    warnings.append(f"{path}: {data!r} isn't one of {schema['enum']}; used {fallback!r}")
                data = fallback
        return data, warnings

    if kind in ("integer", "number"):
        try:
            value = float(data)
            return (int(value) if kind == "integer" else value), warnings
        except (TypeError, ValueError):
            warnings.append(f"{path}: expected a number")
            return 0, warnings

    if kind == "boolean":
        if isinstance(data, str):
            return data.strip().lower() in ("true", "yes", "1"), warnings
        return bool(data), warnings

    return data, warnings
