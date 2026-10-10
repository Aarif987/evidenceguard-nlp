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


def test_explicit_force_fallback_clears_diagnostic():
    recognizer = GLiNEREntityRecognizer()
    entities = recognizer.extract_entities(DEMO_TEXT, force_fallback=True)

    assert recognizer.active_backend == "heuristic_fallback"
    assert recognizer.last_diagnostic is None
    assert all(e.source == "heuristic_fallback" for e in entities)


def test_gliner_failure_populates_diagnostic_and_falls_back():
    recognizer = GLiNEREntityRecognizer(
        model_name="nonexistent-org/nonexistent-gliner-model-id",
        local_files_only=True,
    )
    entities = recognizer.extract_entities(DEMO_TEXT, force_fallback=False)

    assert recognizer.active_backend == "heuristic_fallback"
    assert recognizer.last_diagnostic is not None
    assert "nonexistent-org/nonexistent-gliner-model-id" in recognizer.last_diagnostic
    assert len(entities) >= 3
    assert all(e.source == "heuristic_fallback" for e in entities)


def test_gliner_label_mapping_normalization():
    class DummyGLiNERModel:
        def __init__(self):
            self.received_labels = None

        def predict_entities(self, text, labels, threshold=0.35):
            self.received_labels = list(labels)
            return [
                {"start": 3, "end": 16, "text": "12 March 2025", "label": "date", "score": 0.98},
                {"start": 18, "end": 34, "text": "ABC Technologies", "label": "organization", "score": 0.97},
                {"start": 66, "end": 78, "text": "XYZ Robotics", "label": "organization", "score": 0.96},
                {"start": 83, "end": 94, "text": "$40 million", "label": "money", "score": 0.99},
            ]

    recognizer = GLiNEREntityRecognizer()
    dummy = DummyGLiNERModel()
    recognizer._model = dummy
    recognizer._is_loaded = True

    entities = recognizer.extract_entities(
        DEMO_TEXT,
        labels=["ORGANIZATION", "DATE", "MONEY"],
        force_fallback=False,
    )

    assert dummy.received_labels == ["organization", "date", "money"]
    assert recognizer.active_backend == "gliner"
    assert recognizer.last_diagnostic is None
    assert [e.label for e in entities] == ["DATE", "ORGANIZATION", "ORGANIZATION", "MONEY"]
    assert all(e.source == "gliner" for e in entities)
    assert validate_entity_offsets(DEMO_TEXT, entities)


def test_real_gliner_inference_extraction():
    recognizer = GLiNEREntityRecognizer()
    entities = recognizer.extract_entities(DEMO_TEXT, force_fallback=False)

    assert recognizer.is_model_available is True
    assert recognizer.active_backend == "gliner"
    assert recognizer.last_diagnostic is None
    assert len(entities) == 4
    assert all(e.source == "gliner" for e in entities)
    assert validate_entity_offsets(DEMO_TEXT, entities)

    by_text = {e.text: e.label for e in entities}
    assert by_text.get("12 March 2025") == "DATE"
    assert by_text.get("ABC Technologies") == "ORGANIZATION"
    assert by_text.get("XYZ Robotics") == "ORGANIZATION"
    assert by_text.get("$40 million") == "MONEY"


