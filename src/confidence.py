"""Confidence calibration and status scoring module for HTL-VIE."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from src.config import VerificationStatus
from src.llm_extractor import CandidateFieldExtraction
from src.verifier import VerifiedFieldResult


@dataclass
class CalibratedFieldAssessment:
    """Combines model self-confidence with empirical evidence-verification status."""
    field_name: str
    extracted_value: Optional[str]
    verification_status: VerificationStatus
    model_confidence: float
    grounding_score: float
    calibrated_confidence: float
    is_mock: bool
    status_explanation: str
    sentence_context: Optional[str]
    source_offsets: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["verification_status"] = self.verification_status.value
        return res


class ConfidenceScorer:
    """Decouples and calibrates model confidence with independent verification."""

    STATUS_GROUNDING_WEIGHTS = {
        VerificationStatus.VERIFIED: 1.0,
        VerificationStatus.AMBIGUOUS: 0.45,
        VerificationStatus.UNSUPPORTED: 0.0,
        VerificationStatus.MISSING: 0.0,
    }

    @classmethod
    def evaluate_field(
        cls,
        candidate: CandidateFieldExtraction,
        verified: VerifiedFieldResult,
    ) -> CalibratedFieldAssessment:
        """Computes calibrated confidence score and transparent explanation."""
        model_conf = float(candidate.model_confidence)
        grounding_weight = cls.STATUS_GROUNDING_WEIGHTS.get(verified.status, 0.0)

        # Calibrated confidence heavily penalizes unsupported or ambiguous grounding
        if verified.status == VerificationStatus.VERIFIED:
            # Verification confirms factual grounding; blend with model confidence
            calibrated = 0.5 * model_conf + 0.5 * grounding_weight
        elif verified.status == VerificationStatus.AMBIGUOUS:
            # Penalized due to multiple possible referents
            calibrated = min(model_conf * 0.5, 0.45)
        elif verified.status == VerificationStatus.UNSUPPORTED:
            # Empirical failure: independent verifier found no evidence in document
            calibrated = 0.0
        else:  # MISSING
            calibrated = 0.0

        offset_str = (
            f"[{verified.source_start_char}:{verified.source_end_char}]"
            if verified.source_start_char is not None and verified.source_end_char is not None
            else "N/A"
        )

        return CalibratedFieldAssessment(
            field_name=candidate.field_name,
            extracted_value=candidate.value,
            verification_status=verified.status,
            model_confidence=round(model_conf, 3),
            grounding_score=round(grounding_weight, 3),
            calibrated_confidence=round(calibrated, 3),
            is_mock=candidate.is_mock,
            status_explanation=verified.status_reason,
            sentence_context=verified.sentence_context,
            source_offsets=offset_str,
        )
