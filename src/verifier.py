"""Independent Deterministic Evidence Verification Engine for HTL-VIE.

Cross-examines LLM-extracted structured fields and quoted evidence directly against
the authoritative source document text without relying on the LLM's self-assessment.
"""

import re
import unicodedata
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from src.config import VerificationStatus
from src.context_builder import SchemaField
from src.llm_extractor import CandidateFieldExtraction
from src.ner import EntityMention

logger = logging.getLogger(__name__)


@dataclass
class VerifiedFieldResult:
    """Detailed verification outcome for an individual extracted schema field."""
    field_name: str
    extracted_value: Optional[str]
    status: VerificationStatus
    status_reason: str
    evidence_quote: Optional[str]
    quote_verified_in_source: bool
    source_start_char: Optional[int]
    source_end_char: Optional[int]
    sentence_context: Optional[str]
    matching_mode: str  # 'exact', 'normalized', 'entity_corroborated', 'none'
    corroborating_entity_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["status"] = self.status.value
        return res


class IndependentEvidenceVerifier:
    """Rigorous, independent verification engine validating extractions against source text."""

    def __init__(self, fuzzy_threshold: float = 0.85):
        self.fuzzy_threshold = fuzzy_threshold

    def verify_field(
        self,
        field: SchemaField,
        extraction: CandidateFieldExtraction,
        source_text: str,
        detected_entities: Optional[List[EntityMention]] = None,
    ) -> VerifiedFieldResult:
        """Independently verifies a single field extraction against the authoritative document."""
        val = extraction.value
        quote = extraction.evidence_quote
        detected_entities = detected_entities or []

        # 1. Check for Missing Values
        if val is None or str(val).strip() == "" or str(val).lower() in ("null", "none", "n/a"):
            return VerifiedFieldResult(
                field_name=field.name,
                extracted_value=None,
                status=VerificationStatus.MISSING,
                status_reason=f"Field '{field.name}' is {'required but ' if field.required else ''}absent in extraction.",
                evidence_quote=quote,
                quote_verified_in_source=False,
                source_start_char=None,
                source_end_char=None,
                sentence_context=None,
                matching_mode="none",
            )

        val_str = str(val).strip()

        # 2. Check Quote Validity (if quote was supplied by LLM)
        quote_verified = False
        if quote and str(quote).strip():
            quote_clean = self._normalize_whitespace(str(quote).strip())
            source_norm = self._normalize_whitespace(source_text)
            quote_verified = quote_clean.lower() in source_norm.lower()

        # 3. Exact Substring Search
        exact_matches = self._find_exact_occurrences(source_text, val_str)
        if len(exact_matches) == 1:
            start, end = exact_matches[0]
            sentence = self._extract_sentence_around_span(source_text, start, end)
            matched_ent = self._find_overlapping_entity(start, end, detected_entities)

            return VerifiedFieldResult(
                field_name=field.name,
                extracted_value=val_str,
                status=VerificationStatus.VERIFIED,
                status_reason="Value found as exact substring at unique position in source.",
                evidence_quote=quote,
                quote_verified_in_source=quote_verified,
                source_start_char=start,
                source_end_char=end,
                sentence_context=sentence,
                matching_mode="exact",
                corroborating_entity_id=matched_ent.id if matched_ent else None,
            )

        if len(exact_matches) > 1:
            # Multiple exact occurrences - check if evidence quote can disambiguate
            disambiguated = self._disambiguate_with_quote(source_text, exact_matches, quote)
            if disambiguated:
                start, end = disambiguated
                sentence = self._extract_sentence_around_span(source_text, start, end)
                matched_ent = self._find_overlapping_entity(start, end, detected_entities)
                return VerifiedFieldResult(
                    field_name=field.name,
                    extracted_value=val_str,
                    status=VerificationStatus.VERIFIED,
                    status_reason=f"Value found {len(exact_matches)} times; successfully disambiguated via evidence quote.",
                    evidence_quote=quote,
                    quote_verified_in_source=quote_verified,
                    source_start_char=start,
                    source_end_char=end,
                    sentence_context=sentence,
                    matching_mode="exact",
                    corroborating_entity_id=matched_ent.id if matched_ent else None,
                )
            else:
                return VerifiedFieldResult(
                    field_name=field.name,
                    extracted_value=val_str,
                    status=VerificationStatus.AMBIGUOUS,
                    status_reason=f"Value appears {len(exact_matches)} times in source text without distinct context.",
                    evidence_quote=quote,
                    quote_verified_in_source=quote_verified,
                    source_start_char=exact_matches[0][0],
                    source_end_char=exact_matches[0][1],
                    sentence_context=self._extract_sentence_around_span(source_text, exact_matches[0][0], exact_matches[0][1]),
                    matching_mode="exact",
                )

        # 4. Normalized Search (casing, whitespace, punctuation, date/money variants)
        norm_match = self._find_normalized_match(source_text, val_str, field.type)
        if norm_match:
            start, end, matched_text, mode = norm_match
            sentence = self._extract_sentence_around_span(source_text, start, end)
            matched_ent = self._find_overlapping_entity(start, end, detected_entities)
            return VerifiedFieldResult(
                field_name=field.name,
                extracted_value=val_str,
                status=VerificationStatus.VERIFIED,
                status_reason=f"Value verified via controlled {mode} normalization against source text '{matched_text}'.",
                evidence_quote=quote,
                quote_verified_in_source=quote_verified,
                source_start_char=start,
                source_end_char=end,
                sentence_context=sentence,
                matching_mode=mode,
                corroborating_entity_id=matched_ent.id if matched_ent else None,
            )

        # 5. Check if Entity Mentions can identify a close or conflicting mention
        conflicting_ents = [
            e for e in detected_entities
            if e.label == field.expected_entity
        ]

        # 6. Unverifiable / Hallucinated / Contradicted
        # Deterministic check for numbers/money/dates
        contradiction_note = ""
        if field.expected_entity == "MONEY" and any(e.label == "MONEY" for e in detected_entities):
            actual_money = [e.text for e in detected_entities if e.label == "MONEY"]
            contradiction_note = f" Source document specifies money value(s): {', '.join(actual_money)}."
        elif field.expected_entity == "DATE" and any(e.label == "DATE" for e in detected_entities):
            actual_dates = [e.text for e in detected_entities if e.label == "DATE"]
            contradiction_note = f" Source document specifies date(s): {', '.join(actual_dates)}."

        return VerifiedFieldResult(
            field_name=field.name,
            extracted_value=val_str,
            status=VerificationStatus.UNSUPPORTED,
            status_reason=f"Extracted value '{val_str}' was NOT found in source text.{contradiction_note}",
            evidence_quote=quote,
            quote_verified_in_source=quote_verified,
            source_start_char=None,
            source_end_char=None,
            sentence_context=None,
            matching_mode="none",
        )

    def verify_all(
        self,
        schema_fields: List[SchemaField],
        extractions: Dict[str, CandidateFieldExtraction],
        source_text: str,
        detected_entities: Optional[List[EntityMention]] = None,
    ) -> Dict[str, VerifiedFieldResult]:
        """Runs independent verification over all fields in schema."""
        results: Dict[str, VerifiedFieldResult] = {}
        for field in schema_fields:
            cand = extractions.get(field.name)
            if cand is None:
                cand = CandidateFieldExtraction(
                    field_name=field.name,
                    value=None,
                    evidence_quote=None,
                    model_confidence=0.0,
                )
            results[field.name] = self.verify_field(
                field=field,
                extraction=cand,
                source_text=source_text,
                detected_entities=detected_entities,
            )
        return results

    # ================= Helper Normalization Routines =================

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _find_exact_occurrences(cls, text: str, query: str) -> List[Tuple[int, int]]:
        """Locates all exact case-sensitive occurrences of query in text."""
        matches = []
        if not query:
            return matches
        start = 0
        while True:
            idx = text.find(query, start)
            if idx == -1:
                break
            matches.append((idx, idx + len(query)))
            start = idx + 1
        return matches

    @classmethod
    def _find_normalized_match(
        cls, source_text: str, query: str, field_type: str
    ) -> Optional[Tuple[int, int, str, str]]:
        """Attempts controlled normalization (case-insensitive, punctuation, currency/numbers, dates)."""
        # A. Case-insensitive exact search
        pattern = re.escape(query)
        match = re.search(pattern, source_text, re.IGNORECASE)
        if match:
            return match.start(), match.end(), source_text[match.start():match.end()], "case_insensitive"

        # B. Currency / Financial Normalization
        # e.g., "$40M" vs "$40 million", "$40,000,000" vs "$40 million", "40 million USD"
        money_norm = cls._match_monetary_expression(source_text, query)
        if money_norm:
            return money_norm

        # C. Date Normalization
        date_norm = cls._match_date_expression(source_text, query)
        if date_norm:
            return date_norm

        # D. Punctuation & Whitespace Normalization
        clean_query = re.sub(r"[^\w\s]", "", query).strip()
        if clean_query and len(clean_query) > 3:
            # Scan tokens in source text
            for m in re.finditer(r"\b" + re.escape(clean_query) + r"\b", source_text, re.IGNORECASE):
                return m.start(), m.end(), source_text[m.start():m.end()], "punctuation_tolerant"

        return None

    @classmethod
    def _match_monetary_expression(cls, source_text: str, query: str) -> Optional[Tuple[int, int, str, str]]:
        """Matches equivalent monetary expressions (e.g. $40M vs $40 million)."""
        def parse_amount(s: str) -> Optional[float]:
            s_clean = s.replace(",", "").replace("$", "").strip().lower()
            m = re.match(r"^([\d.]+)\s*(million|billion|trillion|m|b|k)?", s_clean)
            if not m:
                return None
            num, unit = float(m.group(1)), m.group(2)
            multiplier = 1.0
            if unit in ("million", "m"):
                multiplier = 1e6
            elif unit in ("billion", "b"):
                multiplier = 1e9
            elif unit in ("k",):
                multiplier = 1e3
            return num * multiplier

        q_val = parse_amount(query)
        if q_val is None:
            return None

        # Search for money mentions in source
        for match in re.finditer(r"\$\s*\d+(?:\.\d+)?\s*(?:million|billion|trillion|k|m|b)?|\d+(?:\.\d+)?\s*(?:dollars|USD)", source_text, re.IGNORECASE):
            src_val = parse_amount(match.group(0))
            if src_val is not None and abs(src_val - q_val) < 1e-3:
                return match.start(), match.end(), source_text[match.start():match.end()], "numeric_currency_normalized"

        return None

    @classmethod
    def _match_date_expression(cls, source_text: str, query: str) -> Optional[Tuple[int, int, str, str]]:
        """Matches equivalent date formats (e.g., 2025-03-12 vs 12 March 2025)."""
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }

        def parse_date_tuple(s: str) -> Optional[Tuple[int, int, int]]:
            s_clean = s.strip().lower()
            # 12 March 2025 or March 12, 2025
            m1 = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", s_clean)
            if m1 and m1.group(2) in month_map:
                return int(m1.group(3)), month_map[m1.group(2)], int(m1.group(1))
            m2 = re.search(r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", s_clean)
            if m2 and m2.group(1) in month_map:
                return int(m2.group(3)), month_map[m2.group(1)], int(m2.group(2))
            # 2025-03-12
            m3 = re.search(r"(\d{4})-(\d{2})-(\d{2})", s_clean)
            if m3:
                return int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
            return None

        q_date = parse_date_tuple(query)
        if not q_date:
            return None

        # Look for dates in source
        date_pattern = r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b|\b[A-Za-z]+\s+\d{1,2},?\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b"
        for match in re.finditer(date_pattern, source_text):
            src_date = parse_date_tuple(match.group(0))
            if src_date and src_date == q_date:
                return match.start(), match.end(), source_text[match.start():match.end()], "date_format_normalized"

        return None

    @classmethod
    def _disambiguate_with_quote(
        cls, source_text: str, matches: List[Tuple[int, int]], quote: Optional[str]
    ) -> Optional[Tuple[int, int]]:
        """If value appears multiple times, use the evidence quote to select the correct span."""
        if not quote:
            return None
        quote_clean = cls._normalize_whitespace(quote.lower())
        for start, end in matches:
            sentence = cls._extract_sentence_around_span(source_text, start, end)
            if sentence and quote_clean in cls._normalize_whitespace(sentence.lower()):
                return start, end
        return None

    @staticmethod
    def _extract_sentence_around_span(text: str, start: int, end: int) -> str:
        """Finds sentence boundaries enclosing the character slice [start:end]."""
        # Backward search for punctuation or beginning
        prev_bound = max(
            text.rfind(".", 0, start),
            text.rfind("!", 0, start),
            text.rfind("?", 0, start),
            text.rfind("\n", 0, start),
        )
        sent_start = 0 if prev_bound == -1 else prev_bound + 1

        # Forward search for punctuation or end
        next_dot = text.find(".", end)
        next_excl = text.find("!", end)
        next_quest = text.find("?", end)
        next_nl = text.find("\n", end)

        candidates = [p for p in (next_dot, next_excl, next_quest, next_nl) if p != -1]
        sent_end = min(candidates) + 1 if candidates else len(text)

        return text[sent_start:sent_end].strip()

    @staticmethod
    def _find_overlapping_entity(
        start: int, end: int, entities: List[EntityMention]
    ) -> Optional[EntityMention]:
        """Finds any detected entity mention whose span overlaps with [start:end]."""
        for ent in entities:
            if max(start, ent.start) < min(end, ent.end):
                return ent
        return None
