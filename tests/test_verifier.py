"""Unit tests for Independent Evidence Verifier, including deterministic negative test."""

import pytest
from src.config import VerificationStatus
from src.context_builder import SchemaField
from src.llm_extractor import CandidateFieldExtraction
from src.ner import EntityMention
from src.verifier import IndependentEvidenceVerifier

SOURCE_TEXT = "On 12 March 2025, ABC Technologies announced that it had acquired XYZ Robotics for $40 million."

ENTITIES = [
    EntityMention("e1", "12 March 2025", "DATE", 3, 16, 0.95, "gliner"),
    EntityMention("e2", "ABC Technologies", "ORGANIZATION", 18, 34, 0.95, "gliner"),
    EntityMention("e3", "XYZ Robotics", "ORGANIZATION", 66, 78, 0.95, "gliner"),
    EntityMention("e4", "$40 million", "MONEY", 83, 94, 0.95, "gliner"),
]


def test_exact_verification():
    verifier = IndependentEvidenceVerifier()
    field = SchemaField("acquirer", "Acquirer", "string", "ORGANIZATION", required=True)
    cand = CandidateFieldExtraction("acquirer", "ABC Technologies", "ABC Technologies announced", 0.9)

    res = verifier.verify_field(field, cand, SOURCE_TEXT, ENTITIES)
    assert res.status == VerificationStatus.VERIFIED
    assert res.matching_mode == "exact"
    assert res.source_start_char == 18
    assert res.source_end_char == 34
    assert SOURCE_TEXT[res.source_start_char:res.source_end_char] == "ABC Technologies"
    assert res.quote_verified_in_source is True


def test_currency_normalization_verification():
    verifier = IndependentEvidenceVerifier()
    field = SchemaField("acquisition_amount", "Amount", "string", "MONEY", required=True)
    # Extracted format is "$40M" while text has "$40 million"
    cand = CandidateFieldExtraction("acquisition_amount", "$40M", "acquired XYZ Robotics for $40 million", 0.9)

    res = verifier.verify_field(field, cand, SOURCE_TEXT, ENTITIES)
    assert res.status == VerificationStatus.VERIFIED
    assert res.matching_mode == "numeric_currency_normalized"
    assert res.source_start_char == 83
    assert res.source_end_char == 94
    assert SOURCE_TEXT[res.source_start_char:res.source_end_char] == "$40 million"


def test_date_normalization_verification():
    verifier = IndependentEvidenceVerifier()
    field = SchemaField("announcement_date", "Date", "string", "DATE", required=True)
    # Extracted format is ISO "2025-03-12" while text has "12 March 2025"
    cand = CandidateFieldExtraction("announcement_date", "2025-03-12", "On 12 March 2025", 0.9)

    res = verifier.verify_field(field, cand, SOURCE_TEXT, ENTITIES)
    assert res.status == VerificationStatus.VERIFIED
    assert res.matching_mode == "date_format_normalized"
    assert res.source_start_char == 3
    assert res.source_end_char == 16


def test_deterministic_negative_test_fifty_million_rejected():
    """MANDATORY DETERMINISTIC NEGATIVE TEST:

    The candidate extraction claims the acquisition amount was '$50 million',
    whereas the authoritative source text explicitly states '$40 million'.
    The independent verifier MUST reject this value and flag it as UNSUPPORTED.
    """
    verifier = IndependentEvidenceVerifier()
    field = SchemaField("acquisition_amount", "Amount", "string", "MONEY", required=True)
    cand = CandidateFieldExtraction(
        field_name="acquisition_amount",
        value="$50 million",  # Deliberate hallucination / falsified extraction
        evidence_quote="acquired XYZ Robotics for $50 million",  # Fabricated quotation
        model_confidence=0.99,  # High model self-confidence must NOT bypass verification
    )

    res = verifier.verify_field(field, cand, SOURCE_TEXT, ENTITIES)

    # Verifications
    assert res.status == VerificationStatus.UNSUPPORTED
    assert "NOT found in source text" in res.status_reason
    assert "$40 million" in res.status_reason  # Demonstrates detection of conflicting ground truth
    assert res.source_start_char is None
    assert res.source_end_char is None
    assert res.quote_verified_in_source is False  # Fabricated quote rejected


def test_missing_field_verification():
    verifier = IndependentEvidenceVerifier()
    field = SchemaField("optional_notes", "Notes", "string", required=False)
    cand = CandidateFieldExtraction("optional_notes", None, None, 0.0)

    res = verifier.verify_field(field, cand, SOURCE_TEXT, ENTITIES)
    assert res.status == VerificationStatus.MISSING
    assert res.source_start_char is None


def test_ambiguous_evidence_without_quote():
    multi_text = "Apple bought Orange in 2020. Later, Apple bought Pear in 2022."
    verifier = IndependentEvidenceVerifier()
    field = SchemaField("acquirer", "Acquirer", "string", "ORGANIZATION")
    cand = CandidateFieldExtraction("acquirer", "Apple", None, 0.8)

    res = verifier.verify_field(field, cand, multi_text, [])
    assert res.status == VerificationStatus.AMBIGUOUS
    assert "appears 2 times" in res.status_reason
