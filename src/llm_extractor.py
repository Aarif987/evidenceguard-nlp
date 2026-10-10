"""LLM Extractor module interfacing with Google Gemini API and deterministic mock mode."""

import json
import re
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Callable
from src.config import Config, PipelineMode, normalize_model_id
from src.context_builder import SchemaField
from src.ner import EntityMention

logger = logging.getLogger(__name__)


@dataclass
class CandidateFieldExtraction:
    """Raw structured field extraction produced by the LLM (before independent verification)."""
    field_name: str
    value: Optional[str]
    evidence_quote: Optional[str]
    model_confidence: float
    is_mock: bool = False
    raw_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GeminiLLMExtractor:
    """Client for schema-guided extraction using Google Gemini API or Mock mode."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        force_mock: bool = False,
        force_live: bool = False,
        request_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        initial_backoff: float = Config.DEFAULT_GEMINI_INITIAL_BACKOFF,
        backoff_multiplier: float = Config.DEFAULT_GEMINI_BACKOFF_MULTIPLIER,
        max_output_tokens: Optional[int] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.api_key = (api_key if api_key is not None else Config.GEMINI_API_KEY).strip()
        self.model_name = normalize_model_id(
            model_name if model_name is not None else Config.GEMINI_MODEL,
            default=Config.DEFAULT_GEMINI_MODEL,
        )
        self.force_mock = force_mock
        self.force_live = force_live
        self.request_timeout = max(
            1.0,
            float(request_timeout if request_timeout is not None else Config.GEMINI_TIMEOUT_SECONDS),
        )
        self.max_retries = max(
            0,
            int(max_retries if max_retries is not None else Config.GEMINI_MAX_RETRIES),
        )
        self.initial_backoff = max(0.0, float(initial_backoff))
        self.backoff_multiplier = max(1.0, float(backoff_multiplier))
        self.max_output_tokens = int(
            max_output_tokens if max_output_tokens is not None else Config.GEMINI_MAX_OUTPUT_TOKENS
        )
        self._sleep_fn = sleep_fn
        self.last_attempt_count: int = 0
        self._client = None
        self._configured_model_name = None
        self._configured_max_tokens = None
        self._init_error = None

    def _init_gemini(self) -> bool:
        """Initializes the google.generativeai client if key is available."""
        self.model_name = normalize_model_id(
            self.model_name, default=Config.DEFAULT_GEMINI_MODEL
        )
        if (
            self._client is not None
            and self._configured_model_name == self.model_name
            and self._configured_max_tokens == self.max_output_tokens
        ):
            return True
        if not Config.is_api_key_valid(self.api_key):
            self._init_error = "Missing or invalid GEMINI_API_KEY. Operating in Mock/Demo Mode."
            return False

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            self._configured_model_name = self.model_name
            self._configured_max_tokens = self.max_output_tokens
            return True
        except Exception as e:
            self._init_error = f"Failed to initialize Gemini API client for model '{self.model_name}': {str(e)}"
            logger.error(self._init_error)
            return False

    @staticmethod
    def _is_non_retryable_error(exc: Exception) -> bool:
        """Identify authentication, invalid API key, model-not-found, or bad-request errors that must not be retried."""
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return True
        cls_name = type(exc).__name__
        if cls_name in {
            "InvalidArgument",
            "Unauthenticated",
            "PermissionDenied",
            "Forbidden",
            "NotFound",
            "BadRequest",
        }:
            return True
        msg = str(exc).lower()
        non_retryable_markers = (
            "api_key",
            "api key",
            "api_key_invalid",
            "unauthenticated",
            "permission denied",
            "permission_denied",
            "401",
            "403",
            "404",
            "not found",
            "not_found",
            "is no longer available",
            "invalid argument",
        )
        return any(marker in msg for marker in non_retryable_markers)

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        """Identify request deadline / timeout exceptions (e.g. 504 Deadline Exceeded)."""
        if isinstance(exc, TimeoutError):
            return True
        cls_name = type(exc).__name__
        if cls_name in {"DeadlineExceeded", "GatewayTimeout", "ReadTimeout", "ConnectTimeout"}:
            return True
        msg = str(exc).lower()
        timeout_markers = (
            "504",
            "deadline expired",
            "deadline_exceeded",
            "deadline exceeded",
            "timed out",
            "timeout",
        )
        return any(marker in msg for marker in timeout_markers)

    @classmethod
    def _is_transient_error(cls, exc: Exception) -> bool:
        """Identify transient timeout or server errors eligible for bounded retry."""
        if cls._is_non_retryable_error(exc):
            return False
        if cls._is_timeout_error(exc):
            return True
        cls_name = type(exc).__name__
        if cls_name in {
            "ServiceUnavailable",
            "InternalServerError",
            "TooManyRequests",
            "ResourceExhausted",
            "Aborted",
            "BadGateway",
        }:
            return True
        msg = str(exc).lower()
        transient_markers = (
            "503",
            "502",
            "500",
            "429",
            "service unavailable",
            "unavailable",
            "high demand",
            "temporarily unavailable",
            "internal error",
            "rate limit",
            "resource exhausted",
        )
        return any(marker in msg for marker in transient_markers)

    def _call_gemini_with_retry(self, full_prompt: str) -> Any:
        """Execute generate_content with explicit timeout and bounded exponential-backoff retries."""
        max_attempts = 1 + max(0, int(self.max_retries))
        self.last_attempt_count = 0

        for attempt in range(max_attempts):
            self.last_attempt_count += 1
            try:
                return self._client.generate_content(
                    full_prompt,
                    request_options={
                        "timeout": float(self.request_timeout),
                        "retry": None,
                    },
                )
            except Exception as exc:
                can_retry = (attempt < max_attempts - 1) and self._is_transient_error(exc)
                if can_retry:
                    delay = self.initial_backoff * (self.backoff_multiplier ** attempt)
                    logger.warning(
                        "Transient Gemini API error on model '%s' (attempt %d/%d, timeout=%.1fs): %s. "
                        "Retrying in %.2fs...",
                        self.model_name,
                        attempt + 1,
                        max_attempts,
                        self.request_timeout,
                        exc,
                        delay,
                    )
                    self._sleep_fn(delay)
                    continue

                if self._is_timeout_error(exc):
                    raise TimeoutError(
                        f"Request timed out after {self.request_timeout:.1f}s "
                        f"({self.last_attempt_count}/{max_attempts} attempt(s)): {exc}"
                    ) from exc
                if attempt > 0:
                    raise RuntimeError(
                        f"Transient server error persisted after {self.last_attempt_count} attempts "
                        f"(timeout={self.request_timeout:.1f}s): {exc}"
                    ) from exc
                raise

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract JSON text from response, handling multi-part thinking model outputs safely."""
        try:
            candidates = getattr(response, "candidates", None)
            if candidates:
                parts = getattr(candidates[0].content, "parts", None)
                if parts and len(parts) > 1:
                    for part in reversed(parts):
                        part_text = getattr(part, "text", "")
                        if part_text and "{" in part_text and "}" in part_text:
                            return part_text.strip()
        except Exception:
            pass
        return response.text.strip()

    def extract(
        self,
        text: str,
        schema_fields: List[SchemaField],
        detected_entities: List[EntityMention],
        prompt_payload: Dict[str, Any],
        execution_mode: Optional[str] = None,
    ) -> Dict[str, CandidateFieldExtraction]:
        """Performs structured extraction across all target schema fields.

        Returns:
            Dict mapping field_name to CandidateFieldExtraction.
        """
        # Determine requested mode
        if self.force_mock or (execution_mode and execution_mode.lower() == PipelineMode.MOCK.value):
            return self._extract_mock(text, schema_fields, detected_entities)

        is_forced_live = self.force_live or (
            execution_mode and execution_mode.lower() == PipelineMode.LIVE.value
        )

        if is_forced_live:
            if not self._init_gemini():
                raise RuntimeError(
                    f"Force Live API failed for model '{self.model_name}': {self._init_error}"
                )
            try:
                return self._extract_live(text, schema_fields, prompt_payload)
            except Exception as e:
                if self._is_timeout_error(e):
                    raise RuntimeError(
                        f"Gemini API timeout with model '{self.model_name}' "
                        f"(timeout={self.request_timeout:.1f}s, attempts={max(1, self.last_attempt_count)}): {str(e)}"
                    ) from e
                raise RuntimeError(
                    f"Gemini API error with model '{self.model_name}': {str(e)}"
                ) from e

        # Auto mode: use live API if key is valid, otherwise fall back to mock
        effective_mode = Config.get_effective_mode(
            override_mode=execution_mode, api_key=self.api_key
        )
        if effective_mode == PipelineMode.LIVE:
            if self._init_gemini():
                try:
                    return self._extract_live(text, schema_fields, prompt_payload)
                except Exception as e:
                    err_msg = f"Model '{self.model_name}' API error: {str(e)}"
                    logger.warning("Live Gemini inference failed (%s). Falling back to mock.", err_msg)
                    return self._extract_mock(
                        text, schema_fields, detected_entities, error_reason=err_msg
                    )
            else:
                logger.info("Live mode unavailable: %s. Using mock.", self._init_error)
                return self._extract_mock(
                    text, schema_fields, detected_entities, error_reason=self._init_error
                )

        # Fallback Mock Mode
        return self._extract_mock(text, schema_fields, detected_entities)

    def _extract_live(
        self,
        text: str,
        schema_fields: List[SchemaField],
        prompt_payload: Dict[str, Any],
    ) -> Dict[str, CandidateFieldExtraction]:
        """Execute live inference call to Gemini API with timeout and bounded retries."""
        prompt = prompt_payload["prompt_text"]
        system_instruction = prompt_payload.get("system_instruction", "")
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        response = self._call_gemini_with_retry(full_prompt)
        response_text = self._extract_response_text(response)
        parsed_json = self._clean_and_parse_json(response_text)

        extractions_data = parsed_json.get("extractions", parsed_json)
        results: Dict[str, CandidateFieldExtraction] = {}

        for field in schema_fields:
            field_data = extractions_data.get(field.name, {})
            if isinstance(field_data, dict):
                val = field_data.get("value")
                quote = field_data.get("evidence_quote")
                conf = float(field_data.get("confidence", 0.85))
            else:
                val = field_data
                quote = str(val) if val else None
                conf = 0.85

            results[field.name] = CandidateFieldExtraction(
                field_name=field.name,
                value=str(val).strip() if val is not None else None,
                evidence_quote=str(quote).strip() if quote is not None else None,
                model_confidence=min(1.0, max(0.0, conf)),
                is_mock=False,
            )

        return results

    def _extract_mock(
        self,
        text: str,
        schema_fields: List[SchemaField],
        detected_entities: List[EntityMention],
        error_reason: Optional[str] = None,
    ) -> Dict[str, CandidateFieldExtraction]:
        """Deterministic mock extractor simulating grounded LLM output.

        Correlates schema fields with detected entity mentions and semantic cues
        without requiring internet connectivity or API credits.
        """
        results: Dict[str, CandidateFieldExtraction] = {}

        # Index detected entities by label
        orgs = [m for m in detected_entities if m.label == "ORGANIZATION"]
        dates = [m for m in detected_entities if m.label == "DATE"]
        money = [m for m in detected_entities if m.label == "MONEY"]

        for field in schema_fields:
            extracted_val = None
            evidence_quote = None
            conf = 0.95

            # M&A schema deterministic extraction heuristics
            if field.name == "acquirer":
                if orgs:
                    extracted_val = orgs[0].text
                    # Quote sentence containing acquirer
                    evidence_quote = self._find_surrounding_quote(text, extracted_val)
            elif field.name == "acquired_company":
                if len(orgs) > 1:
                    extracted_val = orgs[1].text
                    evidence_quote = self._find_surrounding_quote(text, extracted_val)
                elif orgs and "acquired" in text.lower():
                    # Check regex if second org wasn't tagged
                    match = re.search(r"acquired\s+([A-Z][A-Za-z0-9&]+(?:\s+[A-Z][A-Za-z0-9&]+)*)", text)
                    if match:
                        extracted_val = match.group(1).strip()
                        evidence_quote = self._find_surrounding_quote(text, extracted_val)
            elif field.name == "announcement_date":
                if dates:
                    extracted_val = dates[0].text
                    evidence_quote = self._find_surrounding_quote(text, extracted_val)
                else:
                    match = re.search(r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", text)
                    if match:
                        extracted_val = match.group(0).strip()
                        evidence_quote = self._find_surrounding_quote(text, extracted_val)
            elif field.name == "acquisition_amount":
                if money:
                    extracted_val = money[0].text
                    evidence_quote = self._find_surrounding_quote(text, extracted_val)
                else:
                    match = re.search(r"\$\s*\d+(?:\.\d+)?\s*(?:million|billion)?", text, re.IGNORECASE)
                    if match:
                        extracted_val = match.group(0).strip()
                        evidence_quote = self._find_surrounding_quote(text, extracted_val)
            else:
                # Generic fallback for custom fields
                for ent in detected_entities:
                    if ent.label == field.expected_entity:
                        extracted_val = ent.text
                        evidence_quote = self._find_surrounding_quote(text, extracted_val)
                        break

            results[field.name] = CandidateFieldExtraction(
                field_name=field.name,
                value=extracted_val,
                evidence_quote=evidence_quote,
                model_confidence=conf if extracted_val else 0.0,
                is_mock=True,
                raw_error=error_reason,
            )

        return results

    def _find_surrounding_quote(self, text: str, target: str) -> Optional[str]:
        """Locates the sentence or clause surrounding the target mention."""
        if not target or target not in text:
            return None
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            if target in s:
                return s.strip()
        idx = text.find(target)
        start = max(0, idx - 20)
        end = min(len(text), idx + len(target) + 20)
        return text[start:end].strip()

    def _clean_and_parse_json(self, raw_str: str) -> Dict[str, Any]:
        """Strips markdown code fences and parses JSON securely."""
        cleaned = raw_str.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse valid JSON from LLM output: {raw_str[:150]}...")
