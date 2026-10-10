"""Evaluation metrics module for HTL-VIE.

Calculates NER performance (Precision, Recall, F1) against gold annotations,
along with proposed case-study metrics: Evidence-Supported Extraction Rate (ESR),
Unsupported Extraction Rate (UER), and Schema Validity Rate (SVR).
"""

from typing import List, Dict, Any, Optional
from src.config import VerificationStatus
from src.context_builder import SchemaField
from src.verifier import VerifiedFieldResult
from src.ner import EntityMention


def calculate_ner_metrics(
    predictions: List[EntityMention],
    gold_entities: List[Dict[str, Any]],
    match_mode: str = "exact",
) -> Dict[str, Any]:
    """Computes Precision, Recall, and F1 for named entity recognition.

    Args:
        predictions: List of EntityMention detected by GLiNER or fallback.
        gold_entities: Ground truth entities with keys: 'text', 'label', optionally 'start', 'end'.
        match_mode: 'exact' (matching text, label, and offsets) or 'token' (matching text and label).

    Returns:
        Dict with precision, recall, f1, tp, fp, fn.
    """
    if not gold_entities and not predictions:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not gold_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predictions), "fn": 0}
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(gold_entities)}

    matched_gold_indices = set()
    tp = 0
    fp = 0

    for pred in predictions:
        matched = False
        for g_idx, gold in enumerate(gold_entities):
            if g_idx in matched_gold_indices:
                continue

            label_matches = pred.label.upper() == str(gold.get("label", "")).upper()
            text_matches = pred.text.strip().lower() == str(gold.get("text", "")).strip().lower()

            if match_mode == "exact" and "start" in gold and "end" in gold:
                offset_matches = pred.start == gold["start"] and pred.end == gold["end"]
                condition = label_matches and offset_matches
            else:
                condition = label_matches and text_matches

            if condition:
                matched_gold_indices.add(g_idx)
                tp += 1
                matched = True
                break

        if not matched:
            fp += 1

    fn = len(gold_entities) - len(matched_gold_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_predictions": len(predictions),
        "total_gold": len(gold_entities),
    }


def calculate_extraction_metrics(
    verified_results: Dict[str, VerifiedFieldResult],
    schema_fields: List[SchemaField],
) -> Dict[str, Any]:
    """Calculates factual grounding and reliability metrics for structured extraction.

    Proposed Case-Study Metrics:
    - ESR (Evidence-Supported Extraction Rate):
      Formula: Verified Non-Null Fields / Total Extracted Non-Null Fields
      Denominator: Number of candidate values returned by the model.
      Interpretation: Fraction of non-null extractions proven factually grounded in source.

    - UER (Unsupported Extraction Rate):
      Formula: Unsupported Non-Null Fields / Total Extracted Non-Null Fields
      Denominator: Number of candidate values returned by the model.
      Interpretation: Fraction of non-null extractions that are hallucinations or contradictions.

    - SVR (Schema Validity Rate):
      Formula: Non-Null Present Fields / Total Required Fields
      Denominator: Total number of mandatory fields defined in the schema.
    """
    total_schema_fields = len(schema_fields)
    required_fields = [f for f in schema_fields if f.required]
    total_required = len(required_fields)

    verified_count = 0
    unsupported_count = 0
    ambiguous_count = 0
    missing_count = 0
    non_null_extracted_count = 0
    required_present_count = 0

    required_names = {f.name for f in required_fields}

    for name, res in verified_results.items():
        if res.extracted_value is not None and str(res.extracted_value).strip() != "":
            non_null_extracted_count += 1
            if name in required_names:
                required_present_count += 1

        if res.status == VerificationStatus.VERIFIED:
            verified_count += 1
        elif res.status == VerificationStatus.UNSUPPORTED:
            unsupported_count += 1
        elif res.status == VerificationStatus.AMBIGUOUS:
            ambiguous_count += 1
        elif res.status == VerificationStatus.MISSING:
            missing_count += 1

    # Safe denominator division
    if non_null_extracted_count > 0:
        esr = verified_count / non_null_extracted_count
        uer = unsupported_count / non_null_extracted_count
        aer = ambiguous_count / non_null_extracted_count  # Ambiguous extraction rate
    else:
        esr = 0.0
        uer = 0.0
        aer = 0.0

    svr = (
        required_present_count / total_required
        if total_required > 0
        else 1.0
    )

    return {
        "total_schema_fields": total_schema_fields,
        "total_required_fields": total_required,
        "non_null_extracted_count": non_null_extracted_count,
        "verified_count": verified_count,
        "unsupported_count": unsupported_count,
        "ambiguous_count": ambiguous_count,
        "missing_count": missing_count,
        "esr": round(esr, 4),  # Evidence-Supported Extraction Rate
        "uer": round(uer, 4),  # Unsupported Extraction Rate
        "aer": round(aer, 4),  # Ambiguous Extraction Rate
        "svr": round(svr, 4),  # Schema Validity Rate
        "metric_notes": {
            "esr": "Proposed Case-Study Metric: verified non-null extractions / total non-null extractions",
            "uer": "Proposed Case-Study Metric: unsupported non-null extractions / total non-null extractions",
            "svr": "Schema compliance: extracted required fields / total required fields",
        },
    }
