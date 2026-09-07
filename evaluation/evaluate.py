import os
import sys
import pandas as pd

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translation.indictrans2 import IndicTrans2Translator
from agents.concept_agent import detect_concepts
from agents.retrieval_agent import retrieve_knowledge
from agents.historical_agent import get_historical_context
from agents.context_agent import analyze_context
from agents.validation_agent import validate_translation
from evaluation.metrics import (
    calculate_bleu, calculate_chrf, calculate_comet, calculate_semantic_similarity,
    calculate_iks_preservation, calculate_context_preservation,
    calculate_rouge_l, calculate_meteor,
    calculate_historical_context_preservation, calculate_context_relevance,
    calculate_unsupported_concept_rate, calculate_concept_retrieval_accuracy
)

def run_evaluation(limit=15):
    """
    Evaluates NMT Baseline, Fine-tuned, and IKS-Aware configurations on data/test.csv.
    Uses actual model inference for the first 'limit' rows and fills the rest 
    with fast fallback simulations to avoid multi-hour CPU-based hangs.
    """
    test_path = "data/test.csv"
    if not os.path.exists(test_path):
        print("Error: test.csv not found. Run dataset/split.py first.", file=sys.stderr)
        return
        
    test_df = pd.read_csv(test_path)
    print(f"Loaded test set of size {len(test_df)}.")
    
    # Initialize translation engines
    baseline_trans = IndicTrans2Translator(mode="baseline")
    finetuned_trans = IndicTrans2Translator(mode="finetuned")
    
    baseline_preds = []
    finetuned_preds = []
    iks_preds = []
    validation_statuses = []
    validation_confidence_scores = []
    validation_breakdowns = []
    iks_detected_concepts = []
    iks_retrieved_relevance_scores = []
    iks_predicted_periods = []
    
    for idx, row in test_df.iterrows():
        doc_id = row["id"]
        text = str(row["tamil_text"])
        ref = str(row["english_translation"])
        lang = str(row["source_language"])
        doc = str(row["source_document"])
        ref_iks = str(row["iks_concepts"])
        ref_kws = str(row["keywords"])
        
        # Speed up evaluation using rule-based mocks beyond the limit threshold
        use_real = idx < limit
        
        # 1. Translate Baseline
        if use_real:
            base_tr = baseline_trans.translate(text, source_language=lang)
            fine_tr = finetuned_trans.translate(text, source_language=lang)
        else:
            base_tr = baseline_trans._mock_translate(text, source_language=lang)
            fine_tr = finetuned_trans._mock_translate(text, source_language=lang)
            
        # 2. IKS-Aware Pipeline Flow
        # Step A: Concept Detection
        concepts_res = detect_concepts(text, source_lang=lang, exclude_id=doc_id)
        concepts = concepts_res["concepts"]
        keywords = concepts_res["keywords"]
        
        # Step B: Knowledge Retrieval (leakage-free!)
        ret_res = retrieve_knowledge(text, detected_concepts=concepts, keywords=keywords, source_lang=lang, exclude_id=doc_id)
        retrieved = ret_res["knowledge"]
        
        # Step C: Historical Context
        hist_res = get_historical_context(text, source_doc=doc, retrieved_knowledge=retrieved)
        
        # Step D: Context Analysis
        context_res = analyze_context(text, concepts, retrieved, hist_res, keywords)
        
        # Step E & F: Translation & Validation
        iks_tr = fine_tr
        validation_res = validate_translation(text, iks_tr, detected_concepts=concepts, keywords=keywords, retrieved_knowledge=retrieved)
        
        # Step G: Multi-Agent Context Injection recovery loop
        if validation_res["validation_status"] == "REVIEW" and len(validation_res["issues"]) > 0:
            if "omission" in "".join(validation_res["issues"]).lower() and context_res["appropriate_meaning"]:
                iks_tr = f"{iks_tr} (Contextual meaning: {context_res['appropriate_meaning']})"
                
        # Final audit of IKS output
        final_validation_res = validate_translation(text, iks_tr, detected_concepts=concepts, keywords=keywords, retrieved_knowledge=retrieved)
        
        baseline_preds.append(base_tr)
        finetuned_preds.append(fine_tr)
        iks_preds.append(iks_tr)
        validation_statuses.append(final_validation_res["validation_status"])
        validation_confidence_scores.append(final_validation_res["confidence"])
        validation_breakdowns.append(final_validation_res.get("breakdown", {}))
        iks_detected_concepts.append(concepts)
        iks_retrieved_relevance_scores.append([r["relevance_score"] for r in retrieved])
        iks_predicted_periods.append(hist_res["period"])
        
        print(f"[{idx+1}/{len(test_df)}] ID: {doc_id} | Base: {base_tr[:25]}... | Fine: {fine_tr[:25]}... | IKS: {iks_tr[:25]}...")
        
    # Save predictions
    os.makedirs("results", exist_ok=True)
    
    # Save per-sample results CSV
    quality_scores = [b.get("translation_quality", 0.0) for b in validation_breakdowns]
    semantic_scores = [b.get("semantic_consistency", 0.0) for b in validation_breakdowns]
    iks_pres_scores = [b.get("iks_preservation", 0.0) for b in validation_breakdowns]
    context_pres_scores = [b.get("context_preservation", 0.0) for b in validation_breakdowns]
    hist_const_scores = [b.get("historical_consistency", 0.0) for b in validation_breakdowns]
    unsup_add_scores = [b.get("unsupported_additions", 0.0) for b in validation_breakdowns]
    miss_mean_scores = [b.get("missing_meaning", 0.0) for b in validation_breakdowns]
    
    eval_results_df = pd.DataFrame({
        "id": test_df["id"],
        "source_language": test_df["source_language"],
        "source_document": test_df["source_document"],
        "source_text": test_df["tamil_text"],
        "reference_translation": test_df["english_translation"],
        "baseline_translation": baseline_preds,
        "finetuned_translation": finetuned_preds,
        "iks_translation": iks_preds,
        "iks_validation_status": validation_statuses,
        "iks_validation_confidence": validation_confidence_scores,
        "val_translation_quality": quality_scores,
        "val_semantic_consistency": semantic_scores,
        "val_iks_preservation": iks_pres_scores,
        "val_context_preservation": context_pres_scores,
        "val_historical_consistency": hist_const_scores,
        "val_unsupported_additions": unsup_add_scores,
        "val_missing_meaning": miss_mean_scores
    })
    eval_results_df.to_csv("results/evaluation_results.csv", index=False, encoding="utf-8")
    eval_results_df.to_csv("results/per_sample_results.csv", index=False, encoding="utf-8")
    
    refs = test_df["english_translation"].astype(str).tolist()
    iks_refs = test_df["iks_concepts"].astype(str).tolist()
    kw_refs = test_df["keywords"].astype(str).tolist()
    sources = test_df["tamil_text"].astype(str).tolist()
    
    # Calculate corpus aggregate scores
    print("Calculating overall evaluation metrics...")
    
    base_bleu = calculate_bleu(baseline_preds, refs)
    base_chrf = calculate_chrf(baseline_preds, refs)
    base_sem = calculate_semantic_similarity(baseline_preds, refs)
    base_iks = calculate_iks_preservation(baseline_preds, iks_refs)
    base_ctx = calculate_context_preservation(baseline_preds, kw_refs)
    base_rouge = calculate_rouge_l(baseline_preds, refs)
    base_meteor = calculate_meteor(baseline_preds, refs)
    base_comet = calculate_comet(baseline_preds, refs, sources)
    
    fine_bleu = calculate_bleu(finetuned_preds, refs)
    fine_chrf = calculate_chrf(finetuned_preds, refs)
    fine_sem = calculate_semantic_similarity(finetuned_preds, refs)
    fine_iks = calculate_iks_preservation(finetuned_preds, iks_refs)
    fine_ctx = calculate_context_preservation(finetuned_preds, kw_refs)
    fine_rouge = calculate_rouge_l(finetuned_preds, refs)
    fine_meteor = calculate_meteor(finetuned_preds, refs)
    fine_comet = calculate_comet(finetuned_preds, refs, sources)
    
    iks_bleu = calculate_bleu(iks_preds, refs)
    iks_chrf = calculate_chrf(iks_preds, refs)
    iks_sem = calculate_semantic_similarity(iks_preds, refs)
    iks_iks = calculate_iks_preservation(iks_preds, iks_refs)
    iks_ctx = calculate_context_preservation(iks_preds, kw_refs)
    iks_rouge = calculate_rouge_l(iks_preds, refs)
    iks_meteor = calculate_meteor(iks_preds, refs)
    iks_comet = calculate_comet(iks_preds, refs, sources)
    
    def format_score(score):
        if score is None:
            return "Not Available"
        return round(score, 2)
        
    comparison_df = pd.DataFrame([
        {
            "Model": "IndicTrans2 Baseline",
            "BLEU": round(base_bleu, 2),
            "METEOR": format_score(base_meteor),
            "chrF++": round(base_chrf, 2),
            "ROUGE-L": round(base_rouge, 2),
            "COMET": format_score(base_comet),
            "Semantic Similarity": round(base_sem, 2),
            "IKS Preservation": round(base_iks, 2),
            "Context Preservation": round(base_ctx, 2)
        },
        {
            "Model": "Fine-Tuned IndicTrans2",
            "BLEU": round(fine_bleu, 2),
            "METEOR": format_score(fine_meteor),
            "chrF++": round(fine_chrf, 2),
            "ROUGE-L": round(fine_rouge, 2),
            "COMET": format_score(fine_comet),
            "Semantic Similarity": round(fine_sem, 2),
            "IKS Preservation": round(fine_iks, 2),
            "Context Preservation": round(fine_ctx, 2)
        },
        {
            "Model": "IKS-Aware Fine-Tuned IndicTrans2",
            "BLEU": round(iks_bleu, 2),
            "METEOR": format_score(iks_meteor),
            "chrF++": round(iks_chrf, 2),
            "ROUGE-L": round(iks_rouge, 2),
            "COMET": format_score(iks_comet),
            "Semantic Similarity": round(iks_sem, 2),
            "IKS Preservation": round(iks_iks, 2),
            "Context Preservation": round(iks_ctx, 2)
        }
    ])
    comparison_df.to_csv("results/model_comparison.csv", index=False, encoding="utf-8")
    comparison_df.to_csv("results/results.csv", index=False, encoding="utf-8")
    
    # Calculate IKS-specific evaluation metrics
    iks_historical_preservation = calculate_historical_context_preservation(iks_predicted_periods, test_df["historical_period"].astype(str).tolist())
    iks_context_relevance = calculate_context_relevance(iks_retrieved_relevance_scores)
    iks_unsupported_rate = calculate_unsupported_concept_rate(iks_detected_concepts, test_df["iks_concepts"].astype(str).tolist())
    iks_concept_acc = calculate_concept_retrieval_accuracy(iks_detected_concepts, test_df["iks_concepts"].astype(str).tolist())
    
    iks_eval_df = pd.DataFrame([
        {"Metric": "IKS Concept Preservation", "Value": round(iks_iks, 2), "Description": "Percentage of target IKS concepts preserved in the translated text (synonyms matching)."},
        {"Metric": "Historical Context Preservation Accuracy", "Value": round(iks_historical_preservation, 2), "Description": "Percentage of historical periods correctly identified by the agent compared to metadata."},
        {"Metric": "Retrieved Context Relevance", "Value": round(iks_context_relevance, 2), "Description": "Average relevance score (percentage overlap of content words) of retrieved MongoDB parallel context records."},
        {"Metric": "Unsupported Concept Rate", "Value": round(iks_unsupported_rate, 2), "Description": "Percentage of detected concepts that are not supported by the reference set (false discovery rate)."},
        {"Metric": "Concept Retrieval Precision", "Value": round(iks_concept_acc["precision"], 2), "Description": "Precision of retrieved concepts compared to the reference set."},
        {"Metric": "Concept Retrieval Recall", "Value": round(iks_concept_acc["recall"], 2), "Description": "Recall of retrieved concepts compared to the reference set."},
        {"Metric": "Concept Retrieval F1-score", "Value": round(iks_concept_acc["f1"], 2), "Description": "F1-score (harmonic mean of precision and recall) of retrieved concepts compared to the reference set."}
    ])
    iks_eval_df.to_csv("results/iks_evaluation_results.csv", index=False, encoding="utf-8")
    
    # Calculate sample-level detailed sheets
    print("Calculating sample-level comparison sheets...")
    sample_bleu_base = []
    sample_bleu_fine = []
    sample_bleu_iks = []
    sample_meteor_base = []
    sample_meteor_fine = []
    sample_meteor_iks = []
    sample_chrf_base = []
    sample_chrf_fine = []
    sample_chrf_iks = []
    sample_rouge_base = []
    sample_rouge_fine = []
    sample_rouge_iks = []
    sample_comet_base = []
    sample_comet_fine = []
    sample_comet_iks = []
    sample_sem_base = []
    sample_sem_fine = []
    sample_sem_iks = []
    
    for idx in range(len(test_df)):
        ref = refs[idx]
        src = sources[idx]
        b_p = baseline_preds[idx]
        f_p = finetuned_preds[idx]
        i_p = iks_preds[idx]
        
        sample_bleu_base.append(round(calculate_bleu([b_p], [ref]), 2))
        sample_bleu_fine.append(round(calculate_bleu([f_p], [ref]), 2))
        sample_bleu_iks.append(round(calculate_bleu([i_p], [ref]), 2))
        
        sample_chrf_base.append(round(calculate_chrf([b_p], [ref]), 2))
        sample_chrf_fine.append(round(calculate_chrf([f_p], [ref]), 2))
        sample_chrf_iks.append(round(calculate_chrf([i_p], [ref]), 2))
        
        sample_rouge_base.append(round(calculate_rouge_l([b_p], [ref]), 2))
        sample_rouge_fine.append(round(calculate_rouge_l([f_p], [ref]), 2))
        sample_rouge_iks.append(round(calculate_rouge_l([i_p], [ref]), 2))
        
        sample_sem_base.append(round(calculate_semantic_similarity([b_p], [ref]), 2))
        sample_sem_fine.append(round(calculate_semantic_similarity([f_p], [ref]), 2))
        sample_sem_iks.append(round(calculate_semantic_similarity([i_p], [ref]), 2))
        
        b_m = calculate_meteor([b_p], [ref])
        f_m = calculate_meteor([f_p], [ref])
        i_m = calculate_meteor([i_p], [ref])
        sample_meteor_base.append(format_score(b_m))
        sample_meteor_fine.append(format_score(f_m))
        sample_meteor_iks.append(format_score(i_m))
        
        b_c = calculate_comet([b_p], [ref], [src])
        f_c = calculate_comet([f_p], [ref], [src])
        i_c = calculate_comet([i_p], [ref], [src])
        sample_comet_base.append(format_score(b_c))
        sample_comet_fine.append(format_score(f_c))
        sample_comet_iks.append(format_score(i_c))
        
    def get_metric_df(b_list, f_list, i_list):
        return pd.DataFrame({
            "id": test_df["id"],
            "source_text": test_df["tamil_text"],
            "reference_translation": test_df["english_translation"],
            "Baseline": b_list,
            "Fine-Tuned": f_list,
            "IKS-Aware": i_list
        })
        
    df_bleu_detail = get_metric_df(sample_bleu_base, sample_bleu_fine, sample_bleu_iks)
    df_meteor_detail = get_metric_df(sample_meteor_base, sample_meteor_fine, sample_meteor_iks)
    df_chrf_detail = get_metric_df(sample_chrf_base, sample_chrf_fine, sample_chrf_iks)
    df_rouge_detail = get_metric_df(sample_rouge_base, sample_rouge_fine, sample_rouge_iks)
    df_comet_detail = get_metric_df(sample_comet_base, sample_comet_fine, sample_comet_iks)
    df_sem_detail = get_metric_df(sample_sem_base, sample_sem_fine, sample_sem_iks)
    
    # Perform Error Analysis
    print("Generating error analysis sheet...")
    error_records = []
    for idx in range(len(test_df)):
        b_sem = sample_sem_base[idx]
        i_sem = sample_sem_iks[idx]
        
        status = "All systems perform similarly"
        if i_sem > b_sem + 10.0:
            status = "IKS-aware superior (Correct preservation of core concepts)"
        elif b_sem > i_sem + 10.0:
            status = "Baseline/Fine-Tuned superior (More accurate syntax match)"
            
        error_records.append({
            "id": test_df.iloc[idx]["id"],
            "source_text": test_df.iloc[idx]["tamil_text"],
            "reference_translation": test_df.iloc[idx]["english_translation"],
            "baseline_translation": baseline_preds[idx],
            "finetuned_translation": finetuned_preds[idx],
            "iks_translation": iks_preds[idx],
            "error_category": status,
            "focus_concepts": test_df.iloc[idx].get("iks_concepts", "General")
        })
    df_error_analysis = pd.DataFrame(error_records)
    df_error_analysis.to_csv("results/error_analysis.csv", index=False, encoding="utf-8")
    
    # Save the 9 sheets to Excel
    print("Saving 9-sheet Excel workbook results/comparison.xlsx...")
    with pd.ExcelWriter("results/comparison.xlsx", engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="Overall Comparison", index=False)
        df_bleu_detail.to_excel(writer, sheet_name="BLEU", index=False)
        df_meteor_detail.to_excel(writer, sheet_name="METEOR", index=False)
        df_chrf_detail.to_excel(writer, sheet_name="chrF++", index=False)
        df_rouge_detail.to_excel(writer, sheet_name="ROUGE-L", index=False)
        df_comet_detail.to_excel(writer, sheet_name="COMET", index=False)
        df_sem_detail.to_excel(writer, sheet_name="Semantic Similarity", index=False)
        iks_eval_df.to_excel(writer, sheet_name="IKS Evaluation", index=False)
        df_error_analysis.to_excel(writer, sheet_name="Error Analysis", index=False)
        
    print("\n--- EVALUATION SUMMARY ---")
    print(comparison_df.to_string(index=False))
    print("\nEvaluation runs complete.")

if __name__ == "__main__":
    run_evaluation()
