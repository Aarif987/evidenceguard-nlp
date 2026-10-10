"""Named Entity Recognition module using GLiNER with graceful fallback.

Supports zero-shot arbitrary-type entity extraction with strict character offset
preservation and verification against the input text.
"""

import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """Represents a localized named entity mention anchored in source text."""
    id: str
    text: str
    label: str
    start: int
    end: int
    score: float
    source: str  # 'gliner' or 'heuristic_fallback'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GLiNEREntityRecognizer:
    """GLiNER-based entity extractor with CPU support and offline fallback."""

    def __init__(
        self,
        model_name: str = "urchade/gliner_small-v2.1",
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._model = None
        self._is_loaded = False
        self._load_error = None

    def load_model(self, allow_download: Optional[bool] = None) -> bool:
        """Attempt to load GLiNER weights. Returns True if successful."""
        if self._is_loaded:
            return True
        use_local_only = self.local_files_only if allow_download is None else (not allow_download)
        try:
            logger.info("Loading GLiNER model: %s on %s (local_only=%s)", self.model_name, self.device, use_local_only)
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(self.model_name, local_files_only=use_local_only)
            if hasattr(self._model, "to") and self.device:
                self._model = self._model.to(self.device)
            self._is_loaded = True
            logger.info("GLiNER model loaded successfully.")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(
                "GLiNER model loading skipped/failed (%s). Gracefully activating rule-based fallback.",
                e,
            )
            self._is_loaded = False
            return False

    @property
    def is_model_available(self) -> bool:
        return self._is_loaded

    def extract_entities(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        threshold: float = 0.35,
        force_fallback: bool = False,
    ) -> List[EntityMention]:
        """Extract entities from text and guarantee exact substring slice validation.

        Args:
            text: The authoritative raw document text.
            labels: Entity labels to search for (defaults to standard labels).
            threshold: Confidence threshold for GLiNER.
            force_fallback: If True, bypasses GLiNER to run heuristic fallback.

        Returns:
            List of validated EntityMention instances.
        """
        if not text or not text.strip():
            return []

        target_labels = labels or [
            "ORGANIZATION",
            "PERSON",
            "LOCATION",
            "DATE",
            "MONEY",
            "PRODUCT",
        ]

        # Try GLiNER inference if available and not forced to fallback
        if not force_fallback and (self._is_loaded or self.load_model()):
            try:
                raw_predictions = self._model.predict_entities(
                    text, target_labels, threshold=threshold
                )
                mentions = []
                for idx, pred in enumerate(raw_predictions):
                    start = int(pred["start"])
                    end = int(pred["end"])
                    extracted_text = pred["text"]

                    # Offset integrity check: verify exact character slice against text
                    if text[start:end] != extracted_text:
                        # Attempt to realign if off-by-one whitespace exists
                        found_idx = text.find(extracted_text, max(0, start - 2))
                        if found_idx != -1 and text[found_idx:found_idx + len(extracted_text)] == extracted_text:
                            start = found_idx
                            end = found_idx + len(extracted_text)
                        else:
                            logger.warning(
                                "Skipping entity '%s' due to mismatch with source [%d:%d]",
                                extracted_text, start, end
                            )
                            continue

                    mention = EntityMention(
                        id=f"ent_{idx+1}",
                        text=extracted_text,
                        label=pred["label"],
                        start=start,
                        end=end,
                        score=float(pred.get("score", 1.0)),
                        source="gliner",
                    )
                    mentions.append(mention)

                return mentions
            except Exception as e:
                logger.error("Error during GLiNER prediction: %s. Using fallback.", e)

        # Fallback to deterministic regex-based entity spotter
        return self._heuristic_fallback(text, target_labels)

    def _heuristic_fallback(self, text: str, labels: List[str]) -> List[EntityMention]:
        """Deterministic regex-based entity spotter for offline or CPU constrained testing."""
        mentions: List[EntityMention] = []
        ent_id = 1

        # Patterns for canonical types with appropriate regex flags
        patterns = {
            "MONEY": (
                [
                    r"\$\s*\d+(?:\.\d+)?\s*(?:million|billion|trillion|k|m|b)?",
                    r"\d+(?:\.\d+)?\s*(?:dollars|USD|euros|EUR|GBP)",
                ],
                re.IGNORECASE,
            ),
            "DATE": (
                [
                    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
                    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                    r"\b\d{4}-\d{2}-\d{2}\b",
                    r"\bQ[1-4]\s+\d{4}\b",
                ],
                re.IGNORECASE,
            ),
            "ORGANIZATION": (
                [
                    r"\b[A-Z][A-Za-z0-9&]+(?:\s+[A-Z][A-Za-z0-9&]+)*\s+(?:Technologies|Robotics|Corporation|Corp|Inc|LLC|Ltd|Group|Holdings|Solutions|Software|Systems|Ventures|Dynamics)\b",
                    r"\b[A-Z]{2,6}\s+(?:Technologies|Robotics|Holdings|Systems|Corp)\b",
                ],
                0,  # Case-sensitive for proper nouns
            ),
        }

        # Check requested labels
        for label in labels:
            if label not in patterns:
                continue
            pat_list, flags = patterns[label]
            for pat in pat_list:
                for match in re.finditer(pat, text, flags=flags):
                    start, end = match.span()
                    matched_text = text[start:end]

                    # Ensure offset slice exactly matches
                    assert text[start:end] == matched_text

                    # Avoid duplicate or overlapping mentions
                    overlap = any(
                        max(start, m.start) < min(end, m.end)
                        for m in mentions
                    )
                    if not overlap:
                        mentions.append(
                            EntityMention(
                                id=f"ent_fb_{ent_id}",
                                text=matched_text,
                                label=label,
                                start=start,
                                end=end,
                                score=0.90,
                                source="heuristic_fallback",
                            )
                        )
                        ent_id += 1

        # Sort mentions by character start position
        mentions.sort(key=lambda m: m.start)
        return mentions


def validate_entity_offsets(text: str, mentions: List[EntityMention]) -> bool:
    """Validate that every entity in the list accurately maps to text[start:end]."""
    for m in mentions:
        if m.start < 0 or m.end > len(text) or m.start >= m.end:
            return False
        if text[m.start:m.end] != m.text:
            return False
    return True
