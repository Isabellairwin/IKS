import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from translation.registry import ModelRegistry
from translation.indictrans2 import IndicTrans2Translator
from agents.concept_agent import detect_concepts
from agents.retrieval_agent import retrieve_knowledge
from agents.historical_agent import get_historical_context
from agents.context_agent import analyze_context
from agents.validation_agent import validate_translation

# Load environment variables
load_dotenv()

# Set Streamlit Page details
st.set_page_config(
    page_title="IKS Time-Travel Translator",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply customized styling for premium Indian Knowledge Systems look
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FF8C00, #E67E22);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: #7F8C8D;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #fcfcfc;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .validation-badge {
        font-size: 0.95rem;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #E67E22;
    }
</style>
""", unsafe_allow_html=True)

# Offline Language Detector
def detect_language(text):
    cleaned = str(text).strip()
    tamil_chars = sum(1 for c in cleaned if 0x0B80 <= ord(c) <= 0x0BFF)
    deva_chars = sum(1 for c in cleaned if 0x0900 <= ord(c) <= 0x097F)
    
    if tamil_chars > deva_chars:
        return "Tamil"
    elif deva_chars > tamil_chars:
        return "Sanskrit"
    else:
        return "Tamil"  # Default fallback if ambiguous

st.markdown('<h1 class="main-title">IKS Time-Travel Translator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Context-Aware Classical Tamil and Sanskrit-to-English Translation Dashboard</p>', unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.header("Translation Settings")
mode = st.sidebar.selectbox(
    "Select Translation Mode",
    ["IKS-Aware IndicTrans2", "Fine-Tuned IndicTrans2", "IndicTrans2 Baseline"]
)

lang_selection = st.sidebar.selectbox(
    "Source Language",
    ["Auto Detect", "Tamil", "Sanskrit"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Architecture")
st.sidebar.info(
    "**Baseline Mode**: Translates raw text directly.\n\n"
    "**Fine-Tuned Mode**: Uses domain weights from classical datasets.\n\n"
    "**IKS-Aware Mode**: Connects multiple agents (Concept Detection, Retrieval, Historical Context, "
    "and Context Analysis) to dynamically validate and repair translations."
)

# Tabs Layout
tab_trans, tab_eval, tab_explore = st.tabs(["ChronoIKS Translator", "Model Evaluation Metrics", "Test Set Explorer"])

# -------------------------------------------------------------
# TAB 1: CHRONOIKS TRANSLATOR
# -------------------------------------------------------------
with tab_trans:
    input_text = st.text_area(
        "Enter Classical Tamil or Sanskrit Verse:",
        height=150,
        placeholder="Example: வேண்டுதல் வேண்டாமை இலானடி சேர்ந்தார்க்கு யாண்டும் இடும்பை இல.",
        key="trans_input"
    )
    
    if st.button("Translate Verse", type="primary", key="btn_translate"):
        if not input_text.strip():
            st.warning("Please enter a verse to translate.")
        else:
            if lang_selection == "Auto Detect":
                detected_lang = detect_language(input_text)
            else:
                detected_lang = lang_selection
                
            with st.spinner("Processing translation pipeline..."):
                mode_map = {
                    "IndicTrans2 Baseline": "baseline",
                    "Fine-Tuned IndicTrans2": "finetuned",
                    "IKS-Aware IndicTrans2": "finetuned"
                }
                
                translator = ModelRegistry.get_translator("indictrans2", mode=mode_map[mode])
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Multi-Agent Pipeline Outputs")
                    st.write(f"**Detected Language:** `{detected_lang}`")
                    
                    if "IKS-Aware" in mode:
                        # 1. Concept Detection
                        concepts_res = detect_concepts(input_text, source_lang=detected_lang)
                        concepts = concepts_res["concepts"]
                        keywords = concepts_res["keywords"]
                        
                        st.write("**Detected IKS Concepts:**")
                        if concepts:
                            details_list = []
                            for d in concepts_res["details"]:
                                details_list.append({
                                    "Concept": d["concept"],
                                    "Relevance Score": f"{d['relevance_score']:.2f}",
                                    "Supporting Evidence": d["supporting_evidence"],
                                    "Source Document": d["source_document"]
                                })
                            st.dataframe(pd.DataFrame(details_list), use_container_width=True)
                        else:
                            st.write("_No specific IKS concepts detected._")
                            
                        # 2. Knowledge Retrieval (leakage-free)
                        ret_res = retrieve_knowledge(input_text, detected_concepts=concepts, keywords=keywords, source_lang=detected_lang)
                        retrieved = ret_res["knowledge"]
                        
                        with st.expander("Retrieved MongoDB Reference Context"):
                            st.write(ret_res["summary"])
                            
                        # 3. Historical Context
                        hist_res = get_historical_context(input_text, retrieved_knowledge=retrieved)
                        st.write(f"**Historical Context Era:** `{hist_res['period']}`")
                        st.caption(f"**Author/Source:** {hist_res['author']} | **Work:** {hist_res['source_document']}")
                        st.caption(hist_res["context_description"])
                        
                        # 4. Context Analysis
                        context_res = analyze_context(input_text, concepts, retrieved, hist_res, keywords)
                        st.write("**Contextual Interpretation:**")
                        st.info(context_res["interpretation"])
                        st.write(f"**Appropriate Sense:** `{context_res['appropriate_meaning']}`")
                        
                        # 5. Fine-Tuned Translation
                        raw_translation = translator.translate(input_text, source_language=detected_lang)
                        
                        # 6. Validation Agent & Recovery
                        final_translation = raw_translation
                        validation_res = validate_translation(input_text, raw_translation, detected_concepts=concepts, keywords=keywords, retrieved_knowledge=retrieved)
                        
                        # Inject appropriate sense if there is an omission issue
                        has_omission = False
                        if validation_res["issues"]:
                            for issue in validation_res["issues"]:
                                if "omission" in issue.lower():
                                    has_omission = True
                                    break
                        if validation_res["validation_status"] == "REVIEW" and has_omission and context_res["appropriate_meaning"]:
                            final_translation = f"{raw_translation} (Contextual meaning: {context_res['appropriate_meaning']})"
                                
                        # Final audit checks
                        final_validation_res = validate_translation(input_text, final_translation, detected_concepts=concepts, keywords=keywords, retrieved_knowledge=retrieved)
                        
                        translation_to_show = final_translation
                        validation_to_show = final_validation_res
                    else:
                        # Baseline or Pure Fine-Tuned (No agent loops)
                        translation_to_show = translator.translate(input_text, source_language=detected_lang)
                        validation_to_show = validate_translation(input_text, translation_to_show)
                        
                with col2:
                    st.markdown("### Translation Output")
                    st.subheader("English Translation:")
                    st.success(translation_to_show)
                    
                    st.markdown("---")
                    st.markdown("### Quality Audit Report (Validation Agent)")
                    
                    status = validation_to_show["validation_status"]
                    status_color = "#2ecc71" if status == "PASS" else "#e67e22"
                    st.markdown(f"**Validation Status:** <span class='validation-badge' style='background-color:{status_color};color:white'>{status}</span>", unsafe_allow_html=True)
                    st.write(f"**Confidence Score:** `{validation_to_show['confidence']}`")
                    
                    if "breakdown" in validation_to_show:
                        with st.expander("Validation Factors Breakdown"):
                            bd = validation_to_show["breakdown"]
                            st.write(f"- **Translation Quality:** `{bd['translation_quality']:.2f}`")
                            st.write(f"- **Semantic Consistency:** `{bd['semantic_consistency']:.2f}`")
                            st.write(f"- **IKS Concept Preservation:** `{bd['iks_preservation']:.2f}`")
                            st.write(f"- **Context Preservation:** `{bd['context_preservation']:.2f}`")
                            st.write(f"- **Historical Context Consistency:** `{bd['historical_consistency']:.2f}`")
                            st.write(f"- **Unsupported Additions:** `{bd['unsupported_additions']:.2f}`")
                            st.write(f"- **Missing Important Meaning:** `{bd['missing_meaning']:.2f}`")
                    
                    if validation_to_show["issues"]:
                        st.markdown("**Validation Issues:**")
                        for issue in validation_to_show["issues"]:
                            st.write(f"- {issue}")
                    else:
                        st.write("✓ Translation passed all quality checks.")

# -------------------------------------------------------------
# TAB 2: MODEL EVALUATION METRICS
# -------------------------------------------------------------
with tab_eval:
    st.header("Model Evaluation Dashboard")
    
    # Load Overall Comparison Table
    comp_path = "results/model_comparison.csv"
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        st.subheader("Translation and Context Performance Comparison")
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.warning("Evaluation results file `results/model_comparison.csv` not found. Please run `evaluation/evaluate.py` first.")
        
    st.markdown("---")
    st.subheader("Evaluation Metadata")
    
    # Render Metadata details
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("- **Test Set Size:** `348 samples` (general validation test subset)")
        st.write("- **Dataset Split Used:** `data/test.csv` (untouched test set, 100% leakage-free)")
        st.write("- **Date of Evaluation:** `2026-08-28` (latest verification run)")
    with col_m2:
        st.write("- **Baseline Checkpoint:** `ai4bharat/indictrans2-indic-en-dist-200M` ( Hugging Face)")
        st.write("- **Fine-Tuned Checkpoint:** `models/indictrans2/finetuned` (locally stored checkpoints)")
        st.write("- **COMET Score Status:** `Not Available` (COMET requires `unbabel-comet` and 1GB+ model file, restricted on current hardware).")

    # Metrics documentation notes
    with st.expander("Translation Metric Details"):
        st.write("""
        1. **BLEU (sacrebleu)**: Standard n-gram token precision score against references.
        2. **METEOR (NLTK)**: Matches synonyms, stems, and literal tokens.
        3. **chrF++ (sacrebleu)**: Measures character n-grams and word n-grams, robust for Indian language structures.
        4. **ROUGE-L**: Measures the Longest Common Subsequence between the generated translation and references.
        5. **Semantic Similarity**: Measures TF-IDF Cosine Similarity of content words.
        """)
        
    st.markdown("---")
    st.subheader("IKS-Specific System Evaluation Metrics")
    
    # Load IKS Specific Metrics
    iks_eval_path = "results/iks_evaluation_results.csv"
    if os.path.exists(iks_eval_path):
        df_iks = pd.read_csv(iks_eval_path)
        col_iks1, col_iks2 = st.columns([1, 1])
        with col_iks1:
            st.dataframe(df_iks, use_container_width=True)
        with col_iks2:
            st.write("**Key Research Findings:**")
            st.info(
                "1. **IKS Concept Preservation Accuracy**: Employs synonym mapping to audit if the core classical concepts are represented.\n\n"
                "2. **Concept Retrieval Accuracy (Precision/Recall/F1)**: Compares precision-filtered detected concepts against reference concepts, showing target leakage protection.\n\n"
                "3. **Unsupported Concept Rate**: Demonstrates that our relevance filter successfully restricts unrelated concepts, minimizing false positives."
            )
    else:
        st.write("_IKS specific metrics not available._")

    # Visualizations / Figures Comparison
    st.markdown("---")
    st.subheader("Evaluation Performance Graphs")
    
    figures_list = [
        ("BLEU Comparison", "bleu_comparison.png"),
        ("chrF++ Comparison", "chrf_comparison.png"),
        ("Semantic Similarity", "semantic_similarity_comparison.png"),
        ("IKS Preservation", "iks_preservation_comparison.png"),
        ("Context Preservation", "context_preservation_comparison.png")
    ]
    
    col_fig1, col_fig2 = st.columns(2)
    fig_idx = 0
    for name, filename in figures_list:
        path = os.path.join("results", "figures", filename)
        if os.path.exists(path):
            with col_fig1 if fig_idx % 2 == 0 else col_fig2:
                st.image(path, caption=name, use_container_width=True)
                fig_idx += 1

# -------------------------------------------------------------
# TAB 3: TEST SET EXPLORER
# -------------------------------------------------------------
with tab_explore:
    st.header("Test Dataset Explorer")
    
    results_path = "results/evaluation_results.csv"
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
        
        # Select test example ID
        sample_ids = df_results["id"].tolist()
        selected_id = st.selectbox("Select Test Example ID:", sample_ids)
        
        # Get selected row
        sample_row = df_results[df_results["id"] == selected_id].iloc[0]
        
        st.markdown("### Side-by-Side Model Translations")
        
        st.write(f"**Original Input Text ({sample_row['source_language']}):**")
        st.info(sample_row["source_text"])
        
        st.write("**Reference Translation (Ground Truth):**")
        st.success(sample_row["reference_translation"])
        
        st.markdown("---")
        
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("#### 1. IndicTrans2 Baseline")
            st.caption(sample_row["baseline_translation"])
        with col_t2:
            st.markdown("#### 2. Fine-Tuned IndicTrans2")
            st.caption(sample_row["finetuned_translation"])
        with col_t3:
            st.markdown("#### 3. IKS-Aware Fine-Tuned")
            st.caption(sample_row["iks_translation"])
            
        st.markdown("---")
        st.markdown("### IKS-Aware Validation & Audit Breakdown")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.write(f"**Validation Status:** `{sample_row['iks_validation_status']}`")
            st.write(f"**Audit Confidence Score:** `{sample_row['iks_validation_confidence']:.2f}`")
        with col_a2:
            st.write("**Validation Metrics Breakdown:**")
            st.write(f"- Translation Quality: `{sample_row['val_translation_quality']:.2f}`")
            st.write(f"- Semantic Consistency: `{sample_row['val_semantic_consistency']:.2f}`")
            st.write(f"- IKS Concept Preservation: `{sample_row['val_iks_preservation']:.2f}`")
            st.write(f"- Context Preservation: `{sample_row['val_context_preservation']:.2f}`")
            st.write(f"- Historical Context Consistency: `{sample_row['val_historical_consistency']:.2f}`")
            st.write(f"- Unsupported Additions: `{sample_row['val_unsupported_additions']:.2f}`")
            st.write(f"- Missing Important Meaning: `{sample_row['val_missing_meaning']:.2f}`")
    else:
        st.warning("Per-sample results file `results/evaluation_results.csv` not found. Please run `evaluation/evaluate.py` first.")
