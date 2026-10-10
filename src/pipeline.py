"""End-to-End Orchestrator Pipeline for HTL-VIE."""

import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from src.config import Config, PipelineMode, VerificationStatus
from src.ner import GLiNEREntityRecognizer, EntityMention, validate_entity_offsets
from src.context_builder import ContextBuilder, SchemaField
from src.llm_extractor import GeminiLLMExtractor, CandidateFieldExtraction
from src.verifier import IndependentEvidenceVerifier, VerifiedFieldResult
from src.confidence import ConfidenceScorer, CalibratedFieldAssessment
from src.metrics import calculate_extraction_metrics, calculate_ner_metrics

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full execution output bundle from the HTL-VIE pipeline."""
    document_text: str
    execution_mode: str  # 'live' or 'mock'
    is_mock: bool
    runtime_seconds: float
    entities: List[Dict[str, Any]]
    schema_fields: List[Dict[str, Any]]
    raw_extractions: Dict[str, Dict[str, Any]]
    verified_results: Dict[str, Dict[str, Any]]
    calibrated_assessments: Dict[str, Dict[str, Any]]
    extraction_metrics: Dict[str, Any]
    ner_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HTLVIEPipeline:
    """Hybrid Transformer-LLM Evidence-Verified Information Extraction Pipeline."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        ner_model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        self.ner_recognizer = GLiNEREntityRecognizer(
            model_name=ner_model_name or Config.GLINER_MODEL,
            device=device,
        )
        self.context_builder = ContextBuilder()
        self.llm_extractor = GeminiLLMExtractor(
            api_key=api_key or Config.GEMINI_API_KEY,
            model_name=model_name or Config.GEMINI_MODEL,
        )
        self.verifier = IndependentEvidenceVerifier()

    def run(
        self,
        document_text: str,
        custom_schema: Optional[List[SchemaField]] = None,
        ner_labels: Optional[List[str]] = None,
        force_mock_llm: bool = False,
        force_fallback_ner: bool = False,
        gold_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> PipelineResult:
        """Executes the complete HTL-VIE pipeline.

        Args:
            document_text: Source document to extract from.
            custom_schema: Optional custom SchemaField list.
            ner_labels: Optional list of entity types to detect.
            force_mock_llm: If True, bypasses API to use mock extraction.
            force_fallback_ner: If True, bypasses GLiNER to use regex heuristic fallback.
            gold_entities: Optional gold NER annotations for benchmark evaluation.

        Returns:
            PipelineResult containing entities, extractions, verified results, and metrics.
        """
        start_time = time.time()
        text = document_text.strip()
        if not text:
            raise ValueError("Input document text cannot be empty.")

        # 1. Resolve active schema
        schema = custom_schema or self.context_builder.schema_fields

        # 2. Named Entity Recognition (GLiNER / Heuristic Fallback)
        entities = self.ner_recognizer.extract_entities(
            text=text,
            labels=ner_labels,
            force_fallback=force_fallback_ner,
        )

        # Offset validation guarantee
        assert validate_entity_offsets(text, entities), "Entity offset integrity check failed!"

        # 3. Context & Prompt Payload Construction
        prompt_payload = self.context_builder.build_prompt_payload(
            text=text,
            detected_entities=entities,
            custom_schema=schema,
        )

        # 4. LLM Structured Extraction (Gemini / Mock)
        if force_mock_llm:
            self.llm_extractor.force_mock = True
        extractions = self.llm_extractor.extract(
            text=text,
            schema_fields=schema,
            detected_entities=entities,
            prompt_payload=prompt_payload,
        )

        # 5. Independent Evidence Verification
        verified_results = self.verifier.verify_all(
            schema_fields=schema,
            extractions=extractions,
            source_text=text,
            detected_entities=entities,
        )

        # 6. Confidence Scoring & Calibration
        calibrated: Dict[str, CalibratedFieldAssessment] = {}
        for field in schema:
            cand = extractions.get(field.name)
            ver = verified_results.get(field.name)
            if cand and ver:
                calibrated[field.name] = ConfidenceScorer.evaluate_field(cand, ver)

        # 7. Compute Extraction Metrics
        ext_metrics = calculate_extraction_metrics(verified_results, schema)

        # 8. Compute NER Metrics (if gold data supplied)
        ner_metrics = None
        if gold_entities:
            ner_metrics = calculate_ner_metrics(entities, gold_entities)

        # Resolve mode indicator
        is_mock = any(c.is_mock for c in extractions.values())
        mode_label = "mock" if is_mock else "live"
        elapsed = round(time.time() - start_time, 3)

        return PipelineResult(
            document_text=text,
            execution_mode=mode_label,
            is_mock=is_mock,
            runtime_seconds=elapsed,
            entities=[e.to_dict() for e in entities],
            schema_fields=[s.to_dict() for s in schema],
            raw_extractions={k: v.to_dict() for k, v in extractions.items()},
            verified_results={k: v.to_dict() for k, v in verified_results.items()},
            calibrated_assessments={k: v.to_dict() for k, v in calibrated.items()},
            extraction_metrics=ext_metrics,
            ner_metrics=ner_metrics,
        )
