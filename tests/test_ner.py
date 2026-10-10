"""Unit tests for Named Entity Recognition module."""

import pytest
from src.ner import GLiNEREntityRecognizer, EntityMention, validate_entity_offsets

DEMO_TEXT = "On 12 March 2025, ABC Technologies announced that it had acquired XYZ Robotics for $40 million."


def test_heuristic_fallback_extraction():
    recognizer = GLiNEREntityRecognizer()
    entities = recognizer.extract_entities(DEMO_TEXT, force_fallback=True)

    assert len(entities) >= 3
    # Offset integrity check
    assert validate_entity_offsets(DEMO_TEXT, entities)

    texts = [e.text for e in entities]
    labels = {e.label: e.text for e in entities}

    assert "ABC Technologies" in texts
    assert "XYZ Robotics" in texts
    assert "$40 million" in texts
    assert labels.get("MONEY") == "$40 million"
    assert labels.get("DATE") == "12 March 2025"


def test_offset_integrity_verification():
    recognizer = GLiNEREntityRecognizer()
    entities = recognizer.extract_entities(DEMO_TEXT, force_fallback=True)

    for ent in entities:
        assert ent.start >= 0
        assert ent.end <= len(DEMO_TEXT)
        assert ent.start < ent.end
        # The exact slice must equal entity text
        assert DEMO_TEXT[ent.start:ent.end] == ent.text


def test_empty_input_handling():
    recognizer = GLiNEREntityRecognizer()
    assert recognizer.extract_entities("") == []
    assert recognizer.extract_entities("   ") == []


def test_validate_entity_offsets_catches_tampering():
    good_mention = EntityMention(
        id="e1",
        text="ABC Technologies",
        label="ORGANIZATION",
        start=18,
        end=34,
        score=0.9,
        source="test",
    )
    assert validate_entity_offsets(DEMO_TEXT, [good_mention])

    # Corrupt the offset
    bad_mention = EntityMention(
        id="e2",
        text="ABC Technologies",
        label="ORGANIZATION",
        start=10,  # Wrong start
        end=26,
        score=0.9,
        source="test",
    )
    assert not validate_entity_offsets(DEMO_TEXT, [bad_mention])
