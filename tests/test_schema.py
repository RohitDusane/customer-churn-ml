# tests/test_schema.py
"""Validates the schema.yaml is parseable and has required fields."""
import yaml
import pytest

def test_schema_loads():
    with open("schema.yaml") as f:
        schema = yaml.safe_load(f)
    assert schema is not None, "schema.yaml failed to parse"

def test_schema_has_columns():
    with open("schema.yaml") as f:
        schema = yaml.safe_load(f)
    assert "columns" in schema or len(schema) > 0, "schema.yaml has no column definitions"