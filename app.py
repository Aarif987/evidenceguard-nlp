"""HTL-VIE Research Prototype Streamlit Application.

Case Study:
Transformer-Based Named Entity Recognition and LLM-Based Information Extraction —
A Hybrid Evidence-Verified Approach for Reliable Information Extraction.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from src.config import Config, VerificationStatus, PipelineMode
from src.context_builder import SchemaField
from src.pipeline import HTLVIEPipeline
from src.ner import validate_entity_offsets

# Page Configuration
st.set_page_config(
    page_title="HTL-VIE Research Prototype",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge-verified {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-unsupported {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-ambiguous {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-missing {
        background-color: #F3F4F6;
        color: #374151;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .mode-badge-live {
        background-color: #3B82F6;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .mode-badge-mock {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_sample_documents() -> List[Dict[str, Any]]:
    """Loads pre-configured sample documents."""
    sample_file = Path(__file__).resolve().parent / "sample_data" / "demo_documents.json"
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_gold_labels() -> Dict[str, Any]:
    """Loads gold annotations for benchmark comparison."""
    gold_file = Path(__file__).resolve().parent / "sample_data" / "demo_gold_labels.json"
    if gold_file.exists():
        with open(gold_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Initialize Session State
if "schema_fields" not in st.session_state:
    st.session_state.schema_fields = [
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

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "document_input" not in st.session_state:
    st.session_state.document_input = (
        "On 12 March 2025, ABC Technologies announced that it had acquired XYZ Robotics for $40 million."
    )

sample_docs = load_sample_documents()
gold_labels = load_gold_labels()

# Sidebar: Configuration & Controls
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/verified_user/materialicons/48dp/2x/baseline_verified_user_black_48dp.png",
        width=48,
    )
    st.title("System Configuration")
    st.caption("HTL-VIE Architecture Controls")

    st.subheader("1. LLM Settings")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=Config.GEMINI_API_KEY,
        type="password",
        help="Optional: Enter a Google Gemini API key or leave blank to run in Deterministic Mock Mode.",
    )

    model_option = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
    )

    exec_mode_option = st.radio(
        "Execution Mode",
        options=["Auto (Use API if key exists)", "Force Mock/Demo Mode", "Force Live API"],
        index=0 if Config.is_api_key_valid() else 1,
    )

    force_mock = (
        True
        if "Mock" in exec_mode_option
        else (False if "Live" in exec_mode_option else not bool(api_key_input and len(api_key_input) > 10))
    )

    st.subheader("2. NER Engine Settings")
    ner_engine = st.selectbox(
        "Entity Detector",
        options=["GLiNER Small (urchade/gliner_small-v2.1)", "Heuristic Rule-Based Fallback"],
        index=0,
    )
    force_fallback_ner = "Fallback" in ner_engine

    st.markdown("---")
    st.markdown("### 🧪 Academic Demonstration")
    st.info(
        "**Deterministic Negative Test:**\n\n"
        "Click the button below to inject a falsified acquisition amount ($50 million instead of $40 million) "
        "to demonstrate independent verification rejection in real time."
    )
    run_negative_demo = st.button("Simulate Hallucination ($50M)", width="stretch")

# Main Header
st.markdown('<div class="main-header">HTL-VIE: Hybrid Transformer–LLM Evidence-Verified IE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<b>Case Study:</b> Transformer-Based Named Entity Recognition and LLM-Based Information Extraction — '
    'A Hybrid Evidence-Verified Approach for Reliable Information Extraction.'
    '</div>',
    unsafe_allow_html=True,
)

# Tabs: [1. Extraction & Verification, 2. Schema Editor, 3. Evaluation & Metrics, 4. Architecture Guide]
tab_pipeline, tab_schema, tab_metrics, tab_arch = st.tabs([
    "🚀 Run Extraction & Verification",
    "📋 Schema Editor",
    "📊 Evaluation & Metrics",
    "🏛️ Architecture & Academic Details",
])

# Tab 1: Extraction & Verification
with tab_pipeline:
    col_input, col_ctrl = st.columns([3, 1])

    with col_input:
        st.markdown("#### Input Document")
        # Sample document selector
        doc_options = ["Custom Document"] + [f"{d['id']}: {d['title']}" for d in sample_docs]
        selected_sample = st.selectbox("Load Sample Document:", options=doc_options, index=1 if sample_docs else 0)

        if selected_sample != "Custom Document":
            for d in sample_docs:
                if f"{d['id']}: {d['title']}" == selected_sample:
                    st.session_state.document_input = d["text"]
                    break

        doc_text = st.text_area(
            "Authoritative Document Text",
            value=st.session_state.document_input,
            height=120,
            key="doc_text_area",
        )

    with col_ctrl:
        st.markdown("#### Pipeline Action")
        st.write("")
        st.write("")
        run_btn = st.button("⚡ Execute HTL-VIE Pipeline", type="primary", width="stretch")

        # Status indicator
        if force_mock:
            st.markdown('<span class="mode-badge-mock">MODE: DEMO / MOCK</span>', unsafe_allow_html=True)
            st.caption("Operating deterministically without external API costs.")
        else:
            st.markdown('<span class="mode-badge-live">MODE: LIVE GEMINI API</span>', unsafe_allow_html=True)
            st.caption(f"Model: {model_option}")

    # Execution Trigger
    if run_btn or run_negative_demo:
        with st.spinner("Executing HTL-VIE Pipeline (NER Localization → Grounded Context → LLM → Verifier)..."):
            try:
                pipeline = HTLVIEPipeline(
                    api_key=api_key_input,
                    model_name=model_option,
                )

                # Execute pipeline
                result = pipeline.run(
                    document_text=doc_text,
                    custom_schema=st.session_state.schema_fields,
                    force_mock_llm=force_mock,
                    force_fallback_ner=force_fallback_ner,
                )

                # If user triggered the deterministic negative demo, tamper with the extraction value
                if run_negative_demo:
                    if "acquisition_amount" in result.raw_extractions:
                        result.raw_extractions["acquisition_amount"]["value"] = "$50 million"
                        result.raw_extractions["acquisition_amount"]["evidence_quote"] = "acquired XYZ Robotics for $50 million"
                        result.raw_extractions["acquisition_amount"]["model_confidence"] = 0.99

                        # Re-run independent verifier on tampered extraction
                        from src.llm_extractor import CandidateFieldExtraction
                        from src.confidence import ConfidenceScorer
                        from src.metrics import calculate_extraction_metrics

                        cand_map = {
                            k: CandidateFieldExtraction(**v)
                            for k, v in result.raw_extractions.items()
                        }
                        ver_map = pipeline.verifier.verify_all(
                            schema_fields=st.session_state.schema_fields,
                            extractions=cand_map,
                            source_text=doc_text,
                        )
                        result.verified_results = {k: v.to_dict() for k, v in ver_map.items()}
                        result.calibrated_assessments = {
                            k: ConfidenceScorer.evaluate_field(cand_map[k], ver_map[k]).to_dict()
                            for k in cand_map
                        }
                        result.extraction_metrics = calculate_extraction_metrics(
                            ver_map, st.session_state.schema_fields
                        )

                st.session_state.pipeline_result = result
                st.success(f"Pipeline executed in {result.runtime_seconds} seconds.")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

    # Display Results
    if st.session_state.pipeline_result:
        res = st.session_state.pipeline_result

        st.markdown("---")
        st.subheader("1. Entity Localization (GLiNER Transformer / Heuristic)")

        if res.entities:
            ent_rows = [
                {
                    "Entity ID": e["id"],
                    "Mention Text": e["text"],
                    "Label": e["label"],
                    "Start Char": e["start"],
                    "End Char": e["end"],
                    "Confidence": f"{e['score']:.2f}",
                    "Source": e["source"],
                }
                for e in res.entities
            ]
            st.dataframe(pd.DataFrame(ent_rows), width="stretch", hide_index=True)
        else:
            st.info("No entities detected.")

        st.markdown("---")
        st.subheader("2. Independent Evidence Verification Results")
        st.caption("Verification decisions generated deterministically without trusting LLM self-reports.")

        # Summary Metric KPI Cards
        m = res.extraction_metrics
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        kpi_col1.metric("ESR (Evidence-Supported)", f"{m['esr']*100:.1f}%", help="Verified non-null extractions / total extracted non-null")
        kpi_col2.metric("UER (Unsupported)", f"{m['uer']*100:.1f}%", help="Unsupported extractions / total extracted non-null")
        kpi_col3.metric("Verified Fields", m["verified_count"])
        kpi_col4.metric("Unsupported Fields", m["unsupported_count"])
        kpi_col5.metric("Schema Validity (SVR)", f"{m['svr']*100:.1f}%")

        # Detailed Verification Table
        table_rows = []
        for f in st.session_state.schema_fields:
            v_data = res.verified_results.get(f.name, {})
            c_data = res.calibrated_assessments.get(f.name, {})
            raw_data = res.raw_extractions.get(f.name, {})

            status_val = v_data.get("status", "MISSING")
            table_rows.append({
                "Field": f.display_name,
                "Extracted Value": v_data.get("extracted_value") or "—",
                "Verification Status": status_val,
                "Model Self-Conf": f"{raw_data.get('model_confidence', 0.0):.2f}",
                "Calibrated Conf": f"{c_data.get('calibrated_confidence', 0.0):.2f}",
                "Offsets": c_data.get("source_offsets", "N/A"),
                "Evidence Quote": v_data.get("evidence_quote") or "—",
                "Quote in Source?": "✅ Yes" if v_data.get("quote_verified_in_source") else "❌ No",
                "Matching Mode": v_data.get("matching_mode", "none"),
                "Verification Explanation": v_data.get("status_reason", "—"),
            })

        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, width="stretch", hide_index=True)

        # Download Controls
        st.markdown("#### 📥 Export Extraction & Verification Audit")
        d_col1, d_col2 = st.columns(2)

        csv_data = df_display.to_csv(index=False).encode("utf-8")
        d_col1.download_button(
            label="📄 Download Results as CSV",
            data=csv_data,
            file_name="htl_vie_verification_audit.csv",
            mime="text/csv",
            width="stretch",
        )

        json_data = json.dumps(res.to_dict(), indent=2).encode("utf-8")
        d_col2.download_button(
            label="📦 Download Complete JSON Payload",
            data=json_data,
            file_name="htl_vie_full_payload.json",
            mime="application/json",
            width="stretch",
        )

        # Raw JSON expander
        with st.expander("🔍 Inspect Full Pipeline Output (JSON)"):
            st.json(res.to_dict())

# Tab 2: Schema Editor
with tab_schema:
    st.subheader("Target Extraction Schema Configuration")
    st.caption("Define the exact fields, data types, and required entity constraints for extraction.")

    schema_table = [
        {
            "Identifier": sf.name,
            "Display Name": sf.display_name,
            "Data Type": sf.type,
            "Expected Entity": sf.expected_entity or "None",
            "Description": sf.description,
            "Required": sf.required,
        }
        for sf in st.session_state.schema_fields
    ]
    st.dataframe(pd.DataFrame(schema_table), width="stretch", hide_index=True)

    st.markdown("#### Add Custom Field")
    with st.form("add_field_form"):
        af_col1, af_col2, af_col3 = st.columns(3)
        new_name = af_col1.text_input("Field Key (e.g., deal_type)")
        new_disp = af_col2.text_input("Display Name (e.g., Deal Type)")
        new_type = af_col3.selectbox("Type", ["string", "currency", "date", "number"])

        af_col4, af_col5 = st.columns([2, 1])
        new_desc = af_col4.text_input("Description (guidance for LLM)")
        new_ent = af_col5.selectbox("Expected Entity", ["None", "ORGANIZATION", "PERSON", "LOCATION", "DATE", "MONEY", "PRODUCT"])
        new_req = st.checkbox("Required Field", value=True)

        submit_field = st.form_submit_button("➕ Add Field to Schema")
        if submit_field and new_name and new_disp:
            st.session_state.schema_fields.append(
                SchemaField(
                    name=new_name.strip().lower().replace(" ", "_"),
                    display_name=new_disp.strip(),
                    type=new_type,
                    expected_entity=None if new_ent == "None" else new_ent,
                    description=new_desc.strip(),
                    required=new_req,
                )
            )
            st.success(f"Field '{new_disp}' added to schema.")
            st.rerun()

    if st.button("↺ Reset Schema to Default M&A"):
        st.session_state.schema_fields = [
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
        st.rerun()

# Tab 3: Evaluation & Metrics
with tab_metrics:
    st.subheader("Academic Evaluation Framework & Metrics")

    st.markdown(
        """
        ### Proposed Case-Study Metrics
        In conventional LLM evaluation, accuracy is often computed against human annotations without
        distinguishing between grounded extraction and hallucinated fluency. The **HTL-VIE** framework introduces:

        1. **Evidence-Supported Extraction Rate (ESR)**:
           $$\\text{ESR} = \\frac{\\text{Verified Non-Null Extractions}}{\\text{Total Non-Null Extracted Fields}}$$
           *Measures the proportion of model claims that are verifiable within the authoritative source document.*

        2. **Unsupported Extraction Rate (UER)**:
           $$\\text{UER} = \\frac{\\text{Unsupported Non-Null Extractions}}{\\text{Total Non-Null Extracted Fields}}$$
           *Measures the hallucination / factual contradiction rate of the LLM.*

        3. **Schema Validity Rate (SVR)**:
           $$\\text{SVR} = \\frac{\\text{Extracted Required Fields}}{\\text{Total Required Fields}}$$
           *Measures schema compliance and completeness.*
        """
    )

    if st.session_state.pipeline_result:
        res = st.session_state.pipeline_result
        m = res.extraction_metrics

        st.markdown("### Current Execution Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("ESR", f"{m['esr']:.4f}")
        m_col2.metric("UER", f"{m['uer']:.4f}")
        m_col3.metric("SVR", f"{m['svr']:.4f}")

        # Gold label NER evaluation if available
        if "doc_1_mna_standard" in gold_labels and "ABC Technologies" in doc_text:
            st.markdown("### Benchmark Entity Evaluation against Gold Annotations")
            gold_data = gold_labels["doc_1_mna_standard"]["entities"]
            from src.metrics import calculate_ner_metrics
            from src.ner import EntityMention

            ent_objs = [
                EntityMention(**e) for e in res.entities
            ]
            ner_eval = calculate_ner_metrics(ent_objs, gold_data, match_mode="token")

            g_col1, g_col2, g_col3 = st.columns(3)
            g_col1.metric("NER Precision", f"{ner_eval['precision']:.4f}")
            g_col2.metric("NER Recall", f"{ner_eval['recall']:.4f}")
            g_col3.metric("NER F1-Score", f"{ner_eval['f1']:.4f}")
            st.caption(f"True Positives: {ner_eval['tp']}, False Positives: {ner_eval['fp']}, False Negatives: {ner_eval['fn']}")

# Tab 4: Architecture Guide
with tab_arch:
    st.subheader("HTL-VIE Architecture & Implementation Details")
    st.markdown(
        """
        ### Theoretical Motivation
        Large Language Models (LLMs) frequently suffer from **ungrounded fluency hallucinations**—generating plausible,
        grammatically coherent outputs that contradict or fabricate facts not present in the input text.
        Conversely, traditional Named Entity Recognition (NER) models achieve high localization precision
        via token-level classification, but lack semantic relationship synthesis and schema mapping abilities.

        **HTL-VIE** solves this tradeoff through a three-stage hybrid architecture:
        1. **Token Span Localization (GLiNER)**: Zero-shot transformer entity extraction preserving exact character offsets `[start, end]`.
        2. **Schema-Guided Relation Extraction (Gemini / LLM)**: Grounded prompt engineering demanding verbatim evidence citations.
        3. **Independent Evidence Verification Engine**: A separate deterministic validator that cross-examines LLM extractions against the original text, penalizing unverified extractions and catching hallucinations.

        ### Verification Status Taxonomy
        | Status | Formal Definition | Calibrated Reliability |
        | :--- | :--- | :--- |
        | **VERIFIED** | Value found in source text via exact or certified normalized matching, with sentence context. | High (0.85 – 1.0) |
        | **UNSUPPORTED** | Value contradicts source document, fabricated, or absent. | Zero (0.0) |
        | **AMBIGUOUS** | Value appears multiple times in conflicting contexts without sufficient disambiguation. | Low (0.2 – 0.45) |
        | **MISSING** | Required or optional schema field returned null or omitted. | Zero (0.0) |
        """
    )
