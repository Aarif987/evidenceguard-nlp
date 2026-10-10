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


def test_model_configuration_defaults_and_normalization():
    """Verify default model is gemini-3.8-flash, supported models list is complete, and models/ prefix is stripped."""
    from src.config import Config, normalize_model_id
    from src.llm_extractor import GeminiLLMExtractor

    expected_models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]
    assert Config.DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"
    assert Config.SUPPORTED_GEMINI_MODELS == expected_models
    assert "gemini-1.5-flash" not in Config.SUPPORTED_GEMINI_MODELS

    # Normalization strips 'models/' prefix
    assert normalize_model_id("models/gemini-3.8-flash") == "gemini-3.8-flash"
    assert normalize_model_id("  gemini-2.5-pro  ") == "gemini-2.5-pro"
    assert normalize_model_id("") == "gemini-3.8-flash"

    # Extractor preserves selected model without 'models/' prefix
    extractor = GeminiLLMExtractor(model_name="models/gemini-3.7-flash")
    assert extractor.model_name == "gemini-3.7-flash"


def test_force_live_api_raises_on_missing_key():
    """Verify Force Live API does NOT silently fall back to mock when credentials are missing."""
    pipeline = HTLVIEPipeline(api_key="", model_name="gemini-3.8-flash")
    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run(
            document_text=DEMO_TEXT,
            force_live_llm=True,
            execution_mode="live",
            force_fallback_ner=True,
        )
    err_text = str(exc_info.value)
    assert "gemini-3.8-flash" in err_text
    assert "GEMINI_API_KEY" in err_text


def test_force_live_api_raises_on_api_error(monkeypatch):
    """Verify Force Live API surfaces model ID and API error instead of returning mock output."""
    pipeline = HTLVIEPipeline(api_key="AIzaSyDummyValidLengthKeyForTest12345", model_name="gemini-3.6-flash")

    def mock_init_ok():
        return True

    def mock_extract_live_fail(*args, **kwargs):
        raise Exception("404 Model not found or unavailable")

    monkeypatch.setattr(pipeline.llm_extractor, "_init_gemini", mock_init_ok)
    monkeypatch.setattr(pipeline.llm_extractor, "_extract_live", mock_extract_live_fail)

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run(
            document_text=DEMO_TEXT,
            force_live_llm=True,
            execution_mode="live",
            force_fallback_ner=True,
        )
    err_text = str(exc_info.value)
    assert "gemini-3.6-flash" in err_text
    assert "404 Model not found" in err_text


def test_auto_mode_falls_back_to_mock_transparently_on_error(monkeypatch):
    """Verify Auto mode falls back to mock on error while clearly marking output as mock."""
    pipeline = HTLVIEPipeline(api_key="AIzaSyDummyValidLengthKeyForTest12345", model_name="gemini-3.5-flash")

    monkeypatch.setattr(pipeline.llm_extractor, "_init_gemini", lambda: True)
    monkeypatch.setattr(
        pipeline.llm_extractor,
        "_extract_live",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("Simulated rate limit 429")),
    )

    result = pipeline.run(
        document_text=DEMO_TEXT,
        execution_mode="auto",
        force_fallback_ner=True,
    )
    assert result.is_mock is True
    assert result.execution_mode == "mock"
    assert result.model_name == "gemini-3.5-flash"
    assert "gemini-3.5-flash" in result.raw_extractions["acquirer"]["raw_error"]
    assert "Simulated rate limit 429" in result.raw_extractions["acquirer"]["raw_error"]


def test_streamlit_app_renders_without_exception():
    """Verify that the Streamlit application starts, exposes the updated model list, and executes mock actions without unhandled exceptions."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_path)
    at.run(timeout=15)
    assert not at.exception

    # Verify Gemini Model selectbox options and default selection
    model_selectbox = at.sidebar.selectbox[0]
    assert "gemini-3.8-flash" in model_selectbox.options
    assert "gemini-3.7-flash" in model_selectbox.options
    assert "gemini-3.6-flash" in model_selectbox.options
    assert "gemini-3.5-flash" in model_selectbox.options
    assert "gemini-3.1-flash-lite" in model_selectbox.options
    assert "gemini-2.5-flash" in model_selectbox.options
    assert "gemini-2.5-flash-lite" in model_selectbox.options
    assert "gemini-2.5-pro" in model_selectbox.options
    assert model_selectbox.value == "gemini-3.8-flash"

    # Switch to Force Mock/Demo Mode for offline unit test button clicks
    at.sidebar.radio[0].set_value("Force Mock/Demo Mode").run(timeout=15)
    assert not at.exception

    # Trigger negative demo button and primary pipeline button
    if len(at.button) >= 2:
        at.button[0].click().run(timeout=15)
        assert not at.exception
        at.button[1].click().run(timeout=15)
        assert not at.exception


def test_pipeline_real_gliner_ner_source():
    """Verify that HTLVIEPipeline with force_fallback_ner=False runs real GLiNER and reports source='gliner'."""
    pipeline = HTLVIEPipeline()
    result = pipeline.run(
        document_text=DEMO_TEXT,
        force_mock_llm=True,
        force_fallback_ner=False,
        gold_entities=GOLD_ENTITIES,
    )

    assert result.ner_source == "gliner"
    assert result.ner_diagnostic is None
    assert len(result.entities) >= 3
    assert all(e["source"] == "gliner" for e in result.entities)


def test_configurable_timeout_and_generation_config():
    """Verify request_timeout, max_retries, and max_output_tokens are configurable and passed to SDK."""
    from src.llm_extractor import GeminiLLMExtractor

    captured_options = {}

    class DummyResponse:
        text = '{"extractions":{"acquirer":{"value":"ABC Technologies","evidence_quote":"ABC Technologies","confidence":0.95}}}'

    class DummyClient:
        def generate_content(self, prompt, request_options=None):
            captured_options.update(request_options or {})
            return DummyResponse()

    extractor = GeminiLLMExtractor(
        api_key="AIzaSyDummyValidLengthKeyForTest12345",
        model_name="gemini-3.8-flash",
        request_timeout=75.0,
        max_retries=2,
        max_output_tokens=2048,
    )
    extractor._client = DummyClient()
    extractor._configured_model_name = "gemini-3.8-flash"
    extractor._configured_max_tokens = 2048

    pipeline = HTLVIEPipeline(
        api_key="AIzaSyDummyValidLengthKeyForTest12345",
        model_name="gemini-3.8-flash",
        request_timeout=75.0,
        max_retries=2,
    )
    pipeline.llm_extractor = extractor

    res = pipeline.run(
        document_text=DEMO_TEXT,
        force_live_llm=True,
        execution_mode="live",
        force_fallback_ner=True,
    )
    assert res.is_mock is False
    assert captured_options.get("timeout") == 75.0
    assert captured_options.get("retry") is None
    assert extractor.last_attempt_count == 1


def test_transient_504_timeout_retries_with_exponential_backoff_and_succeeds():
    """Verify transient 504 Deadline Expired errors trigger bounded exponential-backoff retries and recover."""
    from src.llm_extractor import GeminiLLMExtractor

    sleep_delays = []
    call_count = {"n": 0}

    class DummyResponse:
        text = (
            '{"extractions":{'
            '"acquirer":{"value":"ABC Technologies","evidence_quote":"ABC Technologies announced","confidence":0.95},'
            '"acquired_company":{"value":"XYZ Robotics","evidence_quote":"acquired XYZ Robotics","confidence":0.95},'
            '"announcement_date":{"value":"12 March 2025","evidence_quote":"On 12 March 2025","confidence":0.95},'
            '"acquisition_amount":{"value":"$40 million","evidence_quote":"for $40 million","confidence":0.95}'
            '}}'
        )

    class DummyClient:
        def generate_content(self, prompt, request_options=None):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("504 Deadline expired before operation could complete.")
            return DummyResponse()

    extractor = GeminiLLMExtractor(
        api_key="AIzaSyDummyValidLengthKeyForTest12345",
        model_name="gemini-3.8-flash",
        request_timeout=45.0,
        max_retries=2,
        initial_backoff=1.5,
        backoff_multiplier=2.0,
        sleep_fn=lambda d: sleep_delays.append(d),
    )
    extractor._client = DummyClient()
    extractor._configured_model_name = "gemini-3.8-flash"
    extractor._configured_max_tokens = extractor.max_output_tokens

    pipeline = HTLVIEPipeline(api_key="AIzaSyDummyValidLengthKeyForTest12345")
    pipeline.llm_extractor = extractor

    result = pipeline.run(
        document_text=DEMO_TEXT,
        force_live_llm=True,
        execution_mode="live",
        force_fallback_ner=True,
    )

    assert result.is_mock is False
    assert result.execution_mode == "live"
    assert call_count["n"] == 3
    assert extractor.last_attempt_count == 3
    assert sleep_delays == [1.5, 3.0]
    assert result.verified_results["acquirer"]["status"] == VerificationStatus.VERIFIED.value


def test_force_live_timeout_exhaustion_reports_clear_error():
    """Verify Force Live API mode reports a clear timeout diagnostic after bounded retries and never returns mock."""
    from src.llm_extractor import GeminiLLMExtractor

    sleep_delays = []

    class AlwaysTimeoutClient:
        def generate_content(self, prompt, request_options=None):
            raise Exception("504 Deadline expired before operation could complete.")

    extractor = GeminiLLMExtractor(
        api_key="AIzaSyDummyValidLengthKeyForTest12345",
        model_name="gemini-3.8-flash",
        request_timeout=30.0,
        max_retries=2,
        initial_backoff=1.0,
        backoff_multiplier=2.0,
        sleep_fn=lambda d: sleep_delays.append(d),
    )
    extractor._client = AlwaysTimeoutClient()
    extractor._configured_model_name = "gemini-3.8-flash"
    extractor._configured_max_tokens = extractor.max_output_tokens

    pipeline = HTLVIEPipeline(api_key="AIzaSyDummyValidLengthKeyForTest12345")
    pipeline.llm_extractor = extractor

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run(
            document_text=DEMO_TEXT,
            force_live_llm=True,
            execution_mode="live",
            force_fallback_ner=True,
        )

    err_msg = str(exc_info.value)
    assert "Gemini API timeout with model 'gemini-3.8-flash'" in err_msg
    assert "timeout=30.0s" in err_msg
    assert "attempts=3" in err_msg
    assert "504 Deadline expired" in err_msg
    assert extractor.last_attempt_count == 3
    assert sleep_delays == [1.0, 2.0]


def test_invalid_api_key_and_404_errors_do_not_retry():
    """Verify invalid API key (400/401) and model 404 errors fail immediately without retrying."""
    from src.llm_extractor import GeminiLLMExtractor

    sleep_delays = []

    class InvalidKeyClient:
        def generate_content(self, prompt, request_options=None):
            raise Exception("400 API key not valid. Please pass a valid API key. [reason: 'API_KEY_INVALID']")

    extractor = GeminiLLMExtractor(
        api_key="AIzaSyInvalidKeyForTestingNoRetry1234",
        model_name="gemini-3.8-flash",
        request_timeout=30.0,
        max_retries=3,
        sleep_fn=lambda d: sleep_delays.append(d),
    )
    extractor._client = InvalidKeyClient()
    extractor._configured_model_name = "gemini-3.8-flash"
    extractor._configured_max_tokens = extractor.max_output_tokens

    pipeline = HTLVIEPipeline(api_key="AIzaSyInvalidKeyForTestingNoRetry1234")
    pipeline.llm_extractor = extractor

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run(
            document_text=DEMO_TEXT,
            force_live_llm=True,
            execution_mode="live",
            force_fallback_ner=True,
        )

    assert extractor.last_attempt_count == 1
    assert sleep_delays == []
    assert "API_KEY_INVALID" in str(exc_info.value)




