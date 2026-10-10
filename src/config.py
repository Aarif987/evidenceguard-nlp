"""Configuration and environment management for HTL-VIE."""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


class VerificationStatus(str, Enum):
    """Explicit verification status definitions for the independent verifier.

    VERIFIED: The extracted value is demonstrably supported by the source document
              via exact match or verified deterministic normalization, with accurate
              offsets and sentence reference.
    UNSUPPORTED: The extracted value contradicts the source text, has no matching
                 textual evidence, or cites hallucinated/fabricated evidence.
    AMBIGUOUS: The extracted value matches multiple conflicting entities/spans or
               evidence is partially matched but uncertain.
    MISSING: A schema field was omitted or returned null/empty by the extractor.
    """
    VERIFIED = "VERIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    MISSING = "MISSING"


class PipelineMode(str, Enum):
    """Execution mode for the extraction pipeline."""
    AUTO = "auto"
    LIVE = "live"
    MOCK = "mock"


def normalize_model_id(model_name: str, default: str = "gemini-3.8-flash") -> str:
    """Normalize a Gemini model identifier by stripping whitespace and any 'models/' prefix."""
    if not model_name or not str(model_name).strip():
        return default
    cleaned = str(model_name).strip()
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/"):]
    return cleaned or default


class Config:
    """Application configuration container."""
    DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
    SUPPORTED_GEMINI_MODELS: List[str] = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = normalize_model_id(
        os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        default=DEFAULT_GEMINI_MODEL,
    )
    DEFAULT_GEMINI_TIMEOUT: float = 60.0
    DEFAULT_GEMINI_MAX_RETRIES: int = 2
    DEFAULT_GEMINI_INITIAL_BACKOFF: float = 1.5
    DEFAULT_GEMINI_BACKOFF_MULTIPLIER: float = 2.0
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS: int = 2048

    GEMINI_TIMEOUT_SECONDS: float = float(
        os.getenv("GEMINI_TIMEOUT_SECONDS", str(DEFAULT_GEMINI_TIMEOUT))
    )
    GEMINI_MAX_RETRIES: int = int(
        os.getenv("GEMINI_MAX_RETRIES", str(DEFAULT_GEMINI_MAX_RETRIES))
    )
    GEMINI_MAX_OUTPUT_TOKENS: int = int(
        os.getenv("GEMINI_MAX_OUTPUT_TOKENS", str(DEFAULT_GEMINI_MAX_OUTPUT_TOKENS))
    )
    GLINER_MODEL: str = os.getenv("GLINER_MODEL", "urchade/gliner_small-v2.1").strip()
    PIPELINE_MODE: str = os.getenv("PIPELINE_MODE", "auto").strip().lower()

    # Default entity types for GLiNER zero-shot span localization
    DEFAULT_NER_LABELS: List[str] = [
        "ORGANIZATION",
        "PERSON",
        "LOCATION",
        "DATE",
        "MONEY",
        "PRODUCT",
    ]

    # Standard M&A demonstration schema
    DEFAULT_SCHEMA: List[Dict[str, Any]] = [
        {
            "name": "acquirer",
            "display_name": "Acquirer",
            "type": "string",
            "expected_entity": "ORGANIZATION",
            "description": "The acquiring company or organization making the purchase.",
            "required": True,
        },
        {
            "name": "acquired_company",
            "display_name": "Acquired Company",
            "type": "string",
            "expected_entity": "ORGANIZATION",
            "description": "The target company or organization being purchased.",
            "required": True,
        },
        {
            "name": "announcement_date",
            "display_name": "Announcement Date",
            "type": "string",
            "expected_entity": "DATE",
            "description": "The date when the acquisition was officially announced.",
            "required": True,
        },
        {
            "name": "acquisition_amount",
            "display_name": "Acquisition Amount",
            "type": "string",
            "expected_entity": "MONEY",
            "description": "The monetary value or transaction cost of the deal.",
            "required": True,
        },
    ]

    @classmethod
    def is_api_key_valid(cls, api_key: str = None) -> bool:
        """Check if a non-placeholder API key is configured."""
        key = (api_key if api_key is not None else cls.GEMINI_API_KEY).strip()
        if not key:
            return False
        if "your_" in key or "placeholder" in key.lower():
            return False
        return len(key) > 10

    @classmethod
    def get_effective_mode(cls, override_mode: str = None, api_key: str = None) -> PipelineMode:
        """Resolve effective execution mode (live vs mock)."""
        mode = (override_mode or cls.PIPELINE_MODE).strip().lower()
        if mode == PipelineMode.LIVE.value:
            return PipelineMode.LIVE
        elif mode == PipelineMode.MOCK.value:
            return PipelineMode.MOCK
        else:  # AUTO
            return PipelineMode.LIVE if cls.is_api_key_valid(api_key) else PipelineMode.MOCK

