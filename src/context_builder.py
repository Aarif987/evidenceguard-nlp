"""Context and prompt builder for schema-guided extraction in HTL-VIE."""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from src.ner import EntityMention


@dataclass
class SchemaField:
    """Definition of a target field to be extracted from the document."""
    name: str
    display_name: str
    type: str  # e.g., 'string', 'number', 'date', 'currency'
    expected_entity: Optional[str] = None  # e.g., 'ORGANIZATION', 'MONEY', 'DATE'
    description: str = ""
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContextBuilder:
    """Assembles prompt and structural context binding the document, NER cues, and schema."""

    def __init__(self, schema_fields: Optional[List[SchemaField]] = None):
        self.schema_fields = schema_fields or self._default_schema()

    @staticmethod
    def _default_schema() -> List[SchemaField]:
        from src.config import Config
        return [
            SchemaField(
                name=f["name"],
                display_name=f["display_name"],
                type=f["type"],
                expected_entity=f.get("expected_entity"),
                description=f.get("description", ""),
                required=f.get("required", True),
            )
            for f in Config.DEFAULT_SCHEMA
        ]

    def build_system_instruction(self) -> str:
        """Constructs system instructions strictly enforcing factual grounding and JSON output."""
        return (
            "You are a high-precision Information Extraction engine adhering to strict factual evidence grounding.\n"
            "Your task is to extract structured values defined in the target schema from the provided document.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Grounding: Every extracted value MUST be directly supported by the source text.\n"
            "2. Hallucinations: Do NOT invent, assume, or extrapolate facts. If a value is absent or ambiguous, set it to null.\n"
            "3. NER Cues: Pre-detected entity spans are provided as localization hints only. They may be incomplete or noisy; do NOT accept them unconditionally.\n"
            "4. Evidence Citation: For every non-null field, you must provide the exact substring quote from the document that justifies your extraction.\n"
            "5. Format: Return ONLY a valid JSON object matching the requested schema.\n"
        )

    def build_prompt_payload(
        self,
        text: str,
        detected_entities: List[EntityMention],
        custom_schema: Optional[List[SchemaField]] = None,
    ) -> Dict[str, Any]:
        """Builds structured prompt payload and human-readable prompt string.

        Args:
            text: The authoritative source document.
            detected_entities: List of GLiNER-extracted entity mentions with character offsets.
            custom_schema: Optional custom list of SchemaFields.

        Returns:
            Dict containing prompt text, formatted schema, and formatted entities.
        """
        active_schema = custom_schema or self.schema_fields

        # Format detected entities as structured hints
        entity_hints = [
            {
                "id": ent.id,
                "label": ent.label,
                "text": ent.text,
                "start_offset": ent.start,
                "end_offset": ent.end,
                "source": ent.source,
            }
            for ent in detected_entities
        ]

        # Format schema fields
        schema_dict = {
            f.name: {
                "display_name": f.display_name,
                "type": f.type,
                "expected_entity_type": f.expected_entity,
                "description": f.description,
                "required": f.required,
            }
            for f in active_schema
        }

        # Expected output structure description
        expected_json_structure = {
            "extractions": {
                f.name: {
                    "value": f"<Extracted value or null if not found>",
                    "evidence_quote": f"<Exact verbatim substring quote from text supporting this extraction, or null>",
                    "confidence": "<Float between 0.0 and 1.0 representing model confidence>",
                }
                for f in active_schema
            }
        }

        prompt_text = (
            f"=== AUTHORITATIVE SOURCE DOCUMENT ===\n"
            f"{text}\n\n"
            f"=== PRE-DETECTED ENTITY SPANS (HINTS ONLY) ===\n"
            f"{json.dumps(entity_hints, indent=2)}\n\n"
            f"=== TARGET EXTRACTION SCHEMA ===\n"
            f"{json.dumps(schema_dict, indent=2)}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Extract each schema field based strictly on the source document.\n"
            f"Output a valid JSON object matching this exact format:\n"
            f"{json.dumps(expected_json_structure, indent=2)}\n\n"
            f"Respond with JSON only. No explanations, no markdown formatting outside json."
        )

        return {
            "system_instruction": self.build_system_instruction(),
            "prompt_text": prompt_text,
            "schema": schema_dict,
            "entity_hints": entity_hints,
        }
