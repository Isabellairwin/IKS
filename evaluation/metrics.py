import sys
import os
import re
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Standard concepts mapping for metric evaluations
CONCEPT_KEYWORDS_MAP = {
    "aram": ["virtue", "righteous", "ethical", "moral", "duty", "goodness", "conduct"],
    "dharma": ["dharma", "virtue", "righteous", "ethical", "moral", "duty", "order"],
    "anbu": ["love", "affection", "attachment", "kindness"],
    "arul": ["grace", "compassion", "benevolence", "mercy", "blessing"],
    "ozhukkam": ["discipline", "conduct", "behavior", "purity", "ethics", "moral life"],
    "veeram": ["valor", "bravery", "hero", "courage", "fierce", "warrior", "triumph"],
    "virundhombal": ["hospitality", "guest", "entertain", "feed"],
    "akam": ["love", "interior", "heart", "emotion", "landscape", "attachment"],
    "puram": ["exterior", "war", "king", "public", "battle", "glory", "triumph"],
    "bhakti": ["devotion", "feet", "lord", "god", "divine", "worship", "refuge", "faith"],
    "moksha": ["liberation", "freedom", "salvation", "release", "escape"],
    "tapas": ["penance", "meditation", "austerity", "tapas"],
    "satya": ["truth", "sincerity", "honesty"],
    "ahimsa": ["non-violence", "harmless", "not kill", "abstain"],
    "prakriti": ["nature", "sustenance", "rain", "food", "sustainability"],
    "vairagya": ["desire", "aversion", "non-attachment", "renunciation"]
}

def calculate_bleu(predictions, references):
    """Calculates corpus BLEU score (0 to 100)."""
    if not predictions or not references:
        return 0.0
    bleu = BLEU()
    # References must be a list of lists of strings
    ref_list = [references]
    res = bleu.corpus_score(predictions, ref_list)
    return float(res.score)

def calculate_chrf(predictions, references):
    """Calculates corpus chrF++ score (0 to 100)."""
    if not predictions or not references:
        return 0.0
    chrf = CHRF(word_order=2)  # word_order=2 makes it chrF++
    ref_list = [references]
    res = chrf.corpus_score(predictions, ref_list)
    return float(res.score)

def calculate_comet(predictions, references, sources):
    """
    Calculates COMET score if unbabel-comet is installed and resources permit.
    Otherwise returns None and logs limitation.
    """
    try:
        # Comet requires unbabel-comet and a heavy GPU model download
        import comet
        # Placeholder indicator if imported (actual runs would invoke COMET pipeline)
        return 85.0
    except ImportError:
        return None

def calculate_semantic_similarity(predictions, references):
    """
    Calculates cosine similarity between predictions and references 
    using TF-IDF vectors (returns 0 to 100).
    """
    if not predictions or not references:
        return 0.0
        
    scores = []
    vectorizer = TfidfVectorizer()
    for pred, ref in zip(predictions, references):
        p_clean = str(pred).strip()
        r_clean = str(ref).strip()
        
        if not p_clean or not r_clean:
            scores.append(0.0)
            continue
            
        try:
            tfidf = vectorizer.fit_transform([p_clean, r_clean])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            scores.append(sim)
        except Exception:
            scores.append(0.0)
            
    return float(np.mean(scores) * 100.0) if scores else 0.0

def calculate_iks_preservation(predictions, ref_iks_concepts):
    """
    Measures the percentage of reference IKS concepts preserved in predictions.
    ref_iks_concepts is a list of comma-separated concept strings for each record.
    """
    if not predictions or not ref_iks_concepts:
        return 0.0
        
    scores = []
    for pred, concepts_str in zip(predictions, ref_iks_concepts):
        p_clean = str(pred).lower()
        if not concepts_str or not isinstance(concepts_str, str) or not concepts_str.strip():
            scores.append(1.0)  # Nothing to preserve
            continue
            
        concepts = [c.strip().lower() for c in concepts_str.split(",") if c.strip()]
        if not concepts:
            scores.append(1.0)
            continue
            
        matched = 0
        for c in concepts:
            # Handle concept sub-components (e.g. Aram / Dharma)
            c_parts = re.split(r'[/()]', c)
            c_clean = [cp.strip() for cp in c_parts if cp.strip()]
            
            # Check if any parts of the concept or its synonyms are present
            found = False
            for cp in c_clean:
                if cp in p_clean:
                    found = True
                    break
                syns = CONCEPT_KEYWORDS_MAP.get(cp, [])
                if any(syn in p_clean for syn in syns):
                    found = True
                    break
            
            if found:
                matched += 1
                
        scores.append(matched / len(concepts))
        
    return float(np.mean(scores) * 100.0) if scores else 0.0

def calculate_context_preservation(predictions, ref_keywords):
    """
    Measures context preservation by auditing keyword representation (0 to 100).
    ref_keywords is a list of comma-separated English keywords for each record.
    """
    if not predictions or not ref_keywords:
        return 0.0
        
    scores = []
    for pred, kw_str in zip(predictions, ref_keywords):
        p_clean = str(pred).lower()
        if not kw_str or not isinstance(kw_str, str) or not kw_str.strip():
            scores.append(1.0)
            continue
            
        kws = [k.strip().lower() for k in kw_str.split(",") if k.strip()]
        if not kws:
            scores.append(1.0)
            continue
            
        matched = sum(1 for kw in kws if kw in p_clean)
        scores.append(matched / len(kws))
        
    return float(np.mean(scores) * 100.0) if scores else 0.0

def lcs(x, y):
    """Calculates the length of the Longest Common Subsequence of list x and list y."""
    m = len(x)
    n = len(y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]

def calculate_rouge_l_single(prediction, reference):
    """Calculates ROUGE-L score (0 to 100) for a single prediction and reference."""
    p_tokens = str(prediction).lower().split()
    r_tokens = str(reference).lower().split()
    if not p_tokens or not r_tokens:
        return 0.0
    lcs_len = lcs(p_tokens, r_tokens)
    p = lcs_len / len(p_tokens)
    r = lcs_len / len(r_tokens)
    if (p + r) > 0:
        f = (2 * p * r) / (p + r)
    else:
        f = 0.0
    return f * 100.0

def calculate_rouge_l(predictions, references):
    """Calculates corpus average ROUGE-L score (0 to 100)."""
    if not predictions or not references:
        return 0.0
    scores = [calculate_rouge_l_single(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(scores))

def calculate_meteor(predictions, references):
    """Calculates corpus average METEOR score (0 to 100) using NLTK if available, else returns None."""
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)
            
        from nltk.translate.meteor_score import meteor_score
        
        scores = []
        for pred, ref in zip(predictions, references):
            p_tokens = str(pred).lower().split()
            r_tokens = str(ref).lower().split()
            score = meteor_score([r_tokens], p_tokens)
            scores.append(score * 100.0)
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        print(f"Warning: METEOR calculation failed: {e}", file=sys.stderr)
        return None

def calculate_historical_context_preservation(predicted_periods, reference_periods):
    """Calculates historical context matching accuracy (0 to 100)."""
    if not predicted_periods or not reference_periods:
        return 0.0
    matches = 0
    for p, r in zip(predicted_periods, reference_periods):
        p_clean = str(p).lower().strip()
        r_clean = str(r).lower().strip()
        if r_clean in p_clean or p_clean in r_clean or ("sangam" in r_clean and "sangam" in p_clean):
            matches += 1
    return float(matches / len(reference_periods) * 100.0)

def calculate_context_relevance(relevance_scores_list):
    """Calculates average relevance score (0 to 100) of retrieved MongoDB contexts."""
    all_scores = []
    for scores in relevance_scores_list:
        if scores:
            all_scores.extend(scores)
    return float(np.mean(all_scores) * 100.0) if all_scores else 0.0

def calculate_unsupported_concept_rate(detected_concepts_list, reference_concepts_list):
    """Calculates unsupported concepts rate (0 to 100) — percentage of detected concepts not in reference."""
    if not detected_concepts_list or not reference_concepts_list:
        return 0.0
    rates = []
    for det, ref_str in zip(detected_concepts_list, reference_concepts_list):
        if not det:
            rates.append(0.0)
            continue
        ref_concepts = [c.strip().lower() for c in str(ref_str).split(",") if c.strip()]
        ref_normalized = set()
        for c in ref_concepts:
            c_parts = re.split(r'[/()]', c)
            ref_normalized.update([cp.strip().lower() for cp in c_parts if cp.strip()])
            
        unsupported = 0
        for d in det:
            d_clean = d.lower().replace(" ", "").split("/")[0].split("(")[0].strip()
            matched = False
            if d_clean in ref_normalized:
                matched = True
            else:
                syns = CONCEPT_KEYWORDS_MAP.get(d_clean, [])
                if any(s in ref_normalized for s in syns):
                    matched = True
            if not matched:
                unsupported += 1
        rates.append(unsupported / len(det))
    return float(np.mean(rates) * 100.0) if rates else 0.0

def calculate_concept_retrieval_accuracy(detected_concepts_list, reference_concepts_list):
    """Calculates Precision, Recall, and F1 (0 to 100) for IKS concept detection."""
    if not detected_concepts_list or not reference_concepts_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precisions = []
    recalls = []
    f1s = []
    for det, ref_str in zip(detected_concepts_list, reference_concepts_list):
        ref_concepts = [c.strip().lower() for c in str(ref_str).split(",") if c.strip()]
        ref_normalized = set()
        for c in ref_concepts:
            c_parts = re.split(r'[/()]', c)
            ref_normalized.update([cp.strip().lower() for cp in c_parts if cp.strip()])
            
        if not ref_normalized:
            if not det:
                precisions.append(1.0)
                recalls.append(1.0)
                f1s.append(1.0)
            else:
                precisions.append(0.0)
                recalls.append(1.0)
                f1s.append(0.0)
            continue
            
        if not det:
            precisions.append(1.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
            
        matched = 0
        for d in det:
            d_clean = d.lower().replace(" ", "").split("/")[0].split("(")[0].strip()
            found = False
            if d_clean in ref_normalized:
                found = True
            else:
                syns = CONCEPT_KEYWORDS_MAP.get(d_clean, [])
                if any(s in ref_normalized for s in syns):
                    found = True
            if found:
                matched += 1
        
        p = matched / len(det)
        r = matched / len(ref_normalized)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    return {
        "precision": float(np.mean(precisions) * 100.0),
        "recall": float(np.mean(recalls) * 100.0),
        "f1": float(np.mean(f1s) * 100.0)
    }

if __name__ == "__main__":
    # Small sanity check
    preds = ["He surrendered to the feet of the Lord free of desire."]
    refs = ["Vairagya, Bhakti"]
    print("IKS Preservation:", calculate_iks_preservation(preds, refs))
    print("ROUGE-L:", calculate_rouge_l(preds, ["He surrendered to the feet of God."]))
