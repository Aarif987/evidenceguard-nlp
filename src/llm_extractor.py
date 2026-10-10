"""LLM Extractor module interfacing with Google Gemini API and deterministic mock mode."""

import json
import re
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from src.config import Config, PipelineMode
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
    ):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model_name or Config.GEMINI_MODEL
        self.force_mock = force_mock
        self._client = None
        self._init_error = None

    def _init_gemini(self) -> bool:
        """Initializes the google.generativeai client if key is available."""
        if self._client is not None:
            return True
        if not self.api_key or "your_" in self.api_key or len(self.api_key) < 10:
            self._init_error = "Missing or invalid GEMINI_API_KEY. Operating in Mock/Demo Mode."
            return False

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"response_mime_type": "application/json"},
            )
            return True
        except Exception as e:
            self._init_error = f"Failed to initialize Gemini API client: {str(e)}"
            logger.error(self._init_error)
            return False

    def extract(
        self,
        text: str,
        schema_fields: List[SchemaField],
        detected_entities: List[EntityMention],
        prompt_payload: Dict[str, Any],
    ) -> Dict[str, CandidateFieldExtraction]:
        """Performs structured extraction across all target schema fields.

        Returns:
            Dict mapping field_name to CandidateFieldExtraction.
        """
        effective_mode = Config.get_effective_mode(
            override_mode=PipelineMode.MOCK.value if self.force_mock else None
        )

        if effective_mode == PipelineMode.LIVE:
            if self._init_gemini():
                try:
                    return self._extract_live(text, schema_fields, prompt_payload)
                except Exception as e:
                    logger.warning("Live Gemini inference failed (%s). Falling back to mock.", e)
                    return self._extract_mock(text, schema_fields, detected_entities, error_reason=str(e))
            else:
                logger.info("Live mode requested but API unavailable: %s. Using mock.", self._init_error)
                return self._extract_mock(text, schema_fields, detected_entities, error_reason=self._init_error)

        # Explicit Mock Mode
        return self._extract_mock(text, schema_fields, detected_entities)

    def _extract_live(
        self,
        text: str,
        schema_fields: List[SchemaField],
        prompt_payload: Dict[str, Any],
    ) -> Dict[str, CandidateFieldExtraction]:
        """Execute live inference call to Gemini API."""
        prompt = prompt_payload["prompt_text"]
        system_instruction = prompt_payload.get("system_instruction", "")

        # Call GenerativeModel
        response = self._client.generate_content(
            f"{system_instruction}\n\n{prompt}",
            request_options={"timeout": 30.0},
        )

        response_text = response.text.strip()
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
