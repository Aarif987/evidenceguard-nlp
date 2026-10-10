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
            "You are a factual Information Extraction engine.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Every extracted value MUST be directly supported by the document; use null if absent or ambiguous.\n"
            "2. Pre-detected entity spans are hints only; verify against the document.\n"
            "3. For every non-null field, provide the exact verbatim substring quote from the document.\n"
            "4. Return ONLY valid JSON matching the requested schema."
        )

    def build_prompt_payload(
        self,
        text: str,
        detected_entities: List[EntityMention],
        custom_schema: Optional[List[SchemaField]] = None,
    ) -> Dict[str, Any]:
        """Builds structured prompt payload and compact prompt string.

        Args:
            text: The authoritative source document.
            detected_entities: List of GLiNER-extracted entity mentions with character offsets.
            custom_schema: Optional custom list of SchemaFields.

        Returns:
            Dict containing prompt text, formatted schema, and formatted entities.
        """
        active_schema = custom_schema or self.schema_fields

        # Full structured entity hints preserved in payload dict
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

        # Full schema dictionary preserved in payload dict
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

        # Compact representations for the LLM prompt to minimize input tokens and latency
        compact_hints = [
            {"text": ent.text, "label": ent.label}
            for ent in detected_entities
        ]
        compact_schema = {
            f.name: {
                "type": f.type,
                "entity": f.expected_entity,
                "desc": f.description,
            }
            for f in active_schema
        }
        compact_output_spec = {
            "extractions": {
                f.name: {
                    "value": "string|null",
                    "evidence_quote": "verbatim quote|null",
                    "confidence": 0.95,
                }
                for f in active_schema
            }
        }

        prompt_text = (
            f"DOCUMENT:\n{text}\n\n"
            f"ENTITY_HINTS:\n{json.dumps(compact_hints, separators=(',', ':'))}\n\n"
            f"SCHEMA:\n{json.dumps(compact_schema, separators=(',', ':'))}\n\n"
            f"OUTPUT_JSON_FORMAT:\n{json.dumps(compact_output_spec, separators=(',', ':'))}\n"
            f"Return JSON only."
        )

        return {
            "system_instruction": self.build_system_instruction(),
            "prompt_text": prompt_text,
            "schema": schema_dict,
            "entity_hints": entity_hints,
        }

