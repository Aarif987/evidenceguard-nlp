"""Unit tests for Evaluation Metrics (NER P/R/F1, ESR, UER, SVR)."""

import pytest
from src.config import VerificationStatus
from src.context_builder import SchemaField
from src.ner import EntityMention
from src.verifier import VerifiedFieldResult
from src.metrics import calculate_ner_metrics, calculate_extraction_metrics


def test_ner_metrics_perfect_match():
    preds = [
        EntityMention("p1", "ABC Technologies", "ORGANIZATION", 18, 34, 1.0, "test"),
        EntityMention("p2", "$40 million", "MONEY", 79, 90, 1.0, "test"),
    ]
    gold = [
        {"text": "ABC Technologies", "label": "ORGANIZATION", "start": 18, "end": 34},
        {"text": "$40 million", "label": "MONEY", "start": 79, "end": 90},
    ]

    metrics = calculate_ner_metrics(preds, gold, match_mode="exact")
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_ner_metrics_partial_match():
    preds = [
        EntityMention("p1", "ABC Technologies", "ORGANIZATION", 18, 34, 1.0, "test"),
        EntityMention("p2", "Extra Entity", "LOCATION", 50, 62, 1.0, "test"),  # FP
    ]
    gold = [
        {"text": "ABC Technologies", "label": "ORGANIZATION", "start": 18, "end": 34},
        {"text": "$40 million", "label": "MONEY", "start": 79, "end": 90},  # FN
    ]

    metrics = calculate_ner_metrics(preds, gold, match_mode="exact")
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_ner_metrics_empty_inputs_safe():
    empty_res = calculate_ner_metrics([], [])
    assert empty_res["f1"] == 1.0

    no_preds = calculate_ner_metrics([], [{"text": "A", "label": "B"}])
    assert no_preds["precision"] == 0.0
    assert no_preds["recall"] == 0.0

    no_gold = calculate_ner_metrics([EntityMention("p", "A", "B", 0, 1, 1.0, "t")], [])
    assert no_gold["precision"] == 0.0


def test_extraction_metrics_calculation():
    schema = [
        SchemaField("f1", "Field 1", "string", required=True),
        SchemaField("f2", "Field 2", "string", required=True),
        SchemaField("f3", "Field 3", "string", required=False),
        SchemaField("f4", "Field 4", "string", required=True),
    ]

    verified_results = {
        "f1": VerifiedFieldResult("f1", "Val 1", VerificationStatus.VERIFIED, "ok", None, False, 0, 5, None, "exact"),
        "f2": VerifiedFieldResult("f2", "Val 2", VerificationStatus.VERIFIED, "ok", None, False, 6, 11, None, "exact"),
        "f3": VerifiedFieldResult("f3", "Val 3", VerificationStatus.UNSUPPORTED, "bad", None, False, None, None, None, "none"),
        "f4": VerifiedFieldResult("f4", None, VerificationStatus.MISSING, "absent", None, False, None, None, None, "none"),
    }

    metrics = calculate_extraction_metrics(verified_results, schema)
    # Extracted non-null fields: f1, f2, f3 (3 fields)
    # Verified non-null: f1, f2 (2 fields)
    # Unsupported: f3 (1 field)
    # Missing: f4 (1 field)
    assert metrics["non_null_extracted_count"] == 3
    assert metrics["verified_count"] == 2
    assert metrics["unsupported_count"] == 1
    assert metrics["missing_count"] == 1

    # ESR = 2 / 3 = 0.6667
    assert metrics["esr"] == pytest.approx(0.6667, rel=1e-2)
    # UER = 1 / 3 = 0.3333
    assert metrics["uer"] == pytest.approx(0.3333, rel=1e-2)
    # SVR = 2 required present / 3 required = 0.6667 (f1 and f2 present, f4 missing)
    assert metrics["svr"] == pytest.approx(0.6667, rel=1e-2)


def test_extraction_metrics_empty_safe():
    schema = [SchemaField("f1", "Field 1", "string", required=True)]
    verified_results = {
        "f1": VerifiedFieldResult("f1", None, VerificationStatus.MISSING, "absent", None, False, None, None, None, "none")
    }
    metrics = calculate_extraction_metrics(verified_results, schema)
    assert metrics["non_null_extracted_count"] == 0
    assert metrics["esr"] == 0.0
    assert metrics["uer"] == 0.0
    assert metrics["svr"] == 0.0
