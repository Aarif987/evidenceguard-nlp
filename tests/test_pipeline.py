"""Integration tests for the end-to-end HTL-VIE pipeline."""

import json
import pandas as pd
import pytest
from src.pipeline import HTLVIEPipeline
from src.config import VerificationStatus

DEMO_TEXT = "On 12 March 2025, ABC Technologies announced that it had acquired XYZ Robotics for $40 million."

GOLD_ENTITIES = [
    {"text": "12 March 2025", "label": "DATE", "start": 3, "end": 16},
    {"text": "ABC Technologies", "label": "ORGANIZATION", "start": 18, "end": 34},
    {"text": "XYZ Robotics", "label": "ORGANIZATION", "start": 66, "end": 78},
    {"text": "$40 million", "label": "MONEY", "start": 83, "end": 94},
]


def test_pipeline_end_to_end_mock_mode():
    """Verify that the full pipeline executes cleanly in mock mode without API keys."""
    pipeline = HTLVIEPipeline()
    result = pipeline.run(
        document_text=DEMO_TEXT,
        force_mock_llm=True,
        force_fallback_ner=True,
        gold_entities=GOLD_ENTITIES,
    )

    assert result.is_mock is True
    assert result.execution_mode == "mock"
    assert len(result.entities) >= 3

    # Check verified extractions
    ver_results = result.verified_results
    assert "acquirer" in ver_results
    assert "acquired_company" in ver_results
    assert "announcement_date" in ver_results
    assert "acquisition_amount" in ver_results

    # All standard fields should be VERIFIED
    assert ver_results["acquirer"]["status"] == VerificationStatus.VERIFIED.value
    assert ver_results["acquired_company"]["status"] == VerificationStatus.VERIFIED.value
    assert ver_results["announcement_date"]["status"] == VerificationStatus.VERIFIED.value
    assert ver_results["acquisition_amount"]["status"] == VerificationStatus.VERIFIED.value

    # Check metrics
    m = result.extraction_metrics
    assert m["verified_count"] == 4
    assert m["unsupported_count"] == 0
    assert m["esr"] == 1.0
    assert m["uer"] == 0.0
    assert m["svr"] == 1.0

    # Check NER metrics
    assert result.ner_metrics is not None
    assert result.ner_metrics["f1"] > 0.8


def test_pipeline_json_and_dataframe_export():
    """Verify serialization to JSON and conversion to pandas DataFrame for CSV export."""
    pipeline = HTLVIEPipeline()
    result = pipeline.run(
        document_text=DEMO_TEXT,
        force_mock_llm=True,
        force_fallback_ner=True,
    )

    # 1. JSON export integrity
    json_str = json.dumps(result.to_dict(), indent=2)
    reloaded = json.loads(json_str)
    assert reloaded["execution_mode"] == "mock"
    assert len(reloaded["entities"]) > 0

    # 2. DataFrame tabular export
    rows = []
    for field_name, ver in result.verified_results.items():
        assessment = result.calibrated_assessments.get(field_name, {})
        rows.append({
            "Field": field_name,
            "Extracted Value": ver.get("extracted_value"),
            "Verification Status": ver.get("status"),
            "Calibrated Confidence": assessment.get("calibrated_confidence"),
            "Evidence Quote": ver.get("evidence_quote"),
            "Source Offsets": assessment.get("source_offsets"),
            "Explanation": ver.get("status_reason"),
        })
    df = pd.DataFrame(rows)
    assert len(df) == 4
    csv_str = df.to_csv(index=False)
    assert "Verification Status" in csv_str
    assert "VERIFIED" in csv_str


def test_pipeline_rejects_empty_document():
    pipeline = HTLVIEPipeline()
    with pytest.raises(ValueError):
        pipeline.run(document_text="   ")


def test_missing_api_credentials_messaging():
    """Verify that an invalid or missing API key gracefully reports clear error messaging."""
    from src.llm_extractor import GeminiLLMExtractor
    extractor = GeminiLLMExtractor(api_key="your_gemini_api_key_here")
    init_ok = extractor._init_gemini()
    assert init_ok is False
    assert extractor._init_error is not None
    assert "Missing or invalid GEMINI_API_KEY" in extractor._init_error


def test_json_parsing_and_markdown_stripping():
    """Verify that raw LLM markdown JSON fences are correctly parsed."""
    from src.llm_extractor import GeminiLLMExtractor
    extractor = GeminiLLMExtractor()
    raw_response = "```json\n{\"extractions\": {\"acquirer\": {\"value\": \"ABC\", \"evidence_quote\": \"ABC\"}}}\n```"
    parsed = extractor._clean_and_parse_json(raw_response)
    assert "extractions" in parsed
    assert parsed["extractions"]["acquirer"]["value"] == "ABC"


def test_streamlit_app_renders_without_exception():
    """Verify that the Streamlit application starts and executes interactive pipeline actions without unhandled exceptions."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_path)
    at.run(timeout=15)
    assert not at.exception

    # Trigger negative demo button and primary pipeline button
    if len(at.button) >= 2:
        at.button[0].click().run(timeout=15)
        assert not at.exception
        at.button[1].click().run(timeout=15)
        assert not at.exception

