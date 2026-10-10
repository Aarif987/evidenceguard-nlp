"""Unit tests for Context and Schema Builder."""

from src.context_builder import ContextBuilder, SchemaField
from src.ner import EntityMention

DEMO_TEXT = "On 12 March 2025, ABC Technologies announced that it had acquired XYZ Robotics for $40 million."


def test_schema_field_creation():
    field = SchemaField(
        name="acquirer",
        display_name="Acquirer",
        type="string",
        expected_entity="ORGANIZATION",
        description="The acquiring company",
        required=True,
    )
    d = field.to_dict()
    assert d["name"] == "acquirer"
    assert d["required"] is True


def test_context_builder_payload():
    builder = ContextBuilder()
    mentions = [
        EntityMention(
            id="ent_1",
            text="ABC Technologies",
            label="ORGANIZATION",
            start=18,
            end=34,
            score=0.95,
            source="test",
        )
    ]
    payload = builder.build_prompt_payload(DEMO_TEXT, mentions)

    assert "system_instruction" in payload
    assert "prompt_text" in payload
    assert DEMO_TEXT in payload["prompt_text"]
    assert "ABC Technologies" in payload["prompt_text"]
    assert "acquirer" in payload["schema"]
    assert "CRITICAL CONSTRAINTS" in payload["system_instruction"]
