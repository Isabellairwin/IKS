import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mapping of IKS concepts to English translation synonyms for validation checks
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

def validate_translation(source_text, translation, detected_concepts=None, keywords=None, retrieved_knowledge=None):
    """
    Validates translation outputs using a dynamic 7-factor scoring audit.
    Evaluates:
      1. Translation quality
      2. Semantic consistency
      3. IKS concept preservation
      4. Context preservation
      5. Historical context consistency
      6. Unsupported additions
      7. Missing important meaning
    Returns a status (PASS/REVIEW), a dynamic confidence score, a list of issues, and detailed breakdown.
    """
    import re
    cleaned_translation = str(translation).strip()
    translation_lower = cleaned_translation.lower()
    issues = []
    
    # 1. Empty translation check
    if not cleaned_translation:
        return {
            "validation_status": "REVIEW",
            "confidence": 0.0,
            "issues": ["Translation is empty."],
            "breakdown": {
                "translation_quality": 0.0,
                "semantic_consistency": 0.0,
                "iks_preservation": 0.0,
                "context_preservation": 0.0,
                "historical_consistency": 0.0,
                "unsupported_additions": 0.0,
                "missing_meaning": 0.0
            }
        }
        
    # Helper to strip recovery-loop context annotation when calculating word count ratios and additions
    clean_for_length = cleaned_translation
    if " (contextual meaning: " in translation_lower:
        idx = translation_lower.find(" (contextual meaning: ")
        clean_for_length = cleaned_translation[:idx].strip()
    elif " (context meaning: " in translation_lower:
        idx = translation_lower.find(" (context meaning: ")
        clean_for_length = cleaned_translation[:idx].strip()
        
    source_words_count = len(str(source_text).split())
    trans_words_count = len(clean_for_length.split())
    ratio = trans_words_count / max(source_words_count, 1)
    
    # --- FACTOR 1: Translation Quality (0.0 to 1.0) ---
    # - Checks script bleed (non-ASCII characters)
    # - Checks length bounds
    non_ascii_chars = len([char for char in cleaned_translation if ord(char) > 127])
    ascii_ratio = non_ascii_chars / max(len(cleaned_translation), 1)
    ascii_score = max(0.0, 1.0 - (ascii_ratio / 0.2)) # drops to 0 if 20% or more non-ASCII bleed
    
    if ascii_ratio > 0.2:
        issues.append("Translation contains high non-ASCII character bleed.")
        
    len_score = 1.0
    if ratio < 0.3:
        len_score = 0.3
        issues.append("Translation is abnormally short, indicating truncation.")
    elif ratio > 3.5:
        len_score = 0.3
        issues.append("Translation is excessively long, suggesting hallucinations.")
        
    quality_score = round((ascii_score + len_score) / 2.0, 2)
    
    # --- FACTOR 2: Semantic Consistency (0.0 to 1.0) ---
    # - Checks similarity against parallel retrieved corpus translations
    semantic_score = 1.0
    if retrieved_knowledge and len(retrieved_knowledge) > 0:
        # Find best reference translation in retrieved matches
        best_ref = retrieved_knowledge[0].get("english_translation", "")
        # Clean both and check word overlap
        t_words = set(re.sub(r'[^\w\s]', ' ', clean_for_length.lower()).split())
        r_words = set(re.sub(r'[^\w\s]', ' ', best_ref.lower()).split())
        
        # Remove common short words from overlap calculation to avoid false positives
        stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'it', 'he', 'him', 'who', 'shall', 'never', 'whose', 'are', 'is', 'at', 'on', 'for'}
        t_words_filtered = t_words - stop_words
        r_words_filtered = r_words - stop_words
        
        if r_words_filtered:
            overlap = t_words_filtered.intersection(r_words_filtered)
            jaccard = len(overlap) / max(len(t_words_filtered.union(r_words_filtered)), 1)
            # A jaccard of 0.20 or more for filtered content words is a strong semantic alignment
            semantic_score = min(1.0, jaccard / 0.25)
        else:
            semantic_score = 0.8
            
        if semantic_score < 0.40:
            issues.append("Low semantic consistency compared to reference contexts.")
    else:
        # Fallback if no retrieved contexts are available
        semantic_score = 0.90 if keywords else 1.0
        
    # --- FACTOR 3: IKS Concept Preservation (0.0 to 1.0) ---
    # - Verifies presence of filtered high-relevance concepts or their synonyms
    iks_score = 1.0
    if detected_concepts:
        missing_concepts = []
        for c in detected_concepts:
            c_clean = c.lower().replace(" ", "").split("/")[0].split("(")[0].strip()
            
            matched = False
            if c_clean in translation_lower:
                matched = True
            else:
                synonyms = CONCEPT_KEYWORDS_MAP.get(c_clean, [])
                for syn in synonyms:
                    if syn in translation_lower:
                        matched = True
                        break
                        
            if not matched:
                # Substring matching fallback
                for k, syns in CONCEPT_KEYWORDS_MAP.items():
                    if k in c_clean:
                        for syn in syns:
                            if syn in translation_lower:
                                matched = True
                                break
            if not matched:
                missing_concepts.append(c)
                
        if missing_concepts:
            iks_score = max(0.0, 1.0 - (len(missing_concepts) / len(detected_concepts)))
            issues.append(f"Potential omission of core IKS concept(s): {', '.join(missing_concepts)}")
            
    # --- FACTOR 4: Context Preservation (0.0 to 1.0) ---
    # - Checks presence of keywords from retrieved context summary
    context_score = 1.0
    if retrieved_knowledge and len(retrieved_knowledge) > 0:
        ctx_words = set()
        for r in retrieved_knowledge:
            text_to_extract = r.get("context", "") + " " + r.get("meaning_summary", "")
            words_extracted = set(re.sub(r'[^\w\s]', ' ', text_to_extract.lower()).split())
            ctx_words.update(words_extracted)
            
        stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'it', 'he', 'him', 'who', 'shall', 'never', 'whose', 'are', 'is', 'at', 'on', 'for', 'with', 'from', 'by', 'as', 'an'}
        ctx_words_filtered = ctx_words - stop_words
        
        t_words = set(re.sub(r'[^\w\s]', ' ', cleaned_translation.lower()).split()) - stop_words
        
        if ctx_words_filtered:
            overlap = t_words.intersection(ctx_words_filtered)
            # Check how many context keywords are preserved
            context_score = min(1.0, len(overlap) / 3.0) # require at least 3 context word matches for 1.0
        else:
            context_score = 1.0
            
        if context_score < 0.5:
            issues.append("Low context preservation score from database reference context.")
            
    # --- FACTOR 5: Historical Context Consistency (0.0 to 1.0) ---
    # - Checks for modern anachronisms
    historical_score = 1.0
    ANACHRONISMS = {
        'internet', 'computer', 'website', 'machine', 'factory', 'electricity', 
        'telephone', 'mobile', 'car', 'train', 'airplane', 'robot', 'television', 
        'radio', 'nuclear', 'modern', 'technology'
    }
    
    found_anachronisms = [w for w in ANACHRONISMS if w in translation_lower]
    if found_anachronisms:
        historical_score = 0.5
        issues.append(f"Translation contains historical anachronisms: {', '.join(found_anachronisms)}")
        
    # --- FACTOR 6: Unsupported Additions (0.0 to 1.0) ---
    # - Checks for translation bloating (excessive words not related to source or concepts)
    unsupported_score = 1.0
    if ratio > 2.2:
        unsupported_score = max(0.0, 1.0 - (ratio - 2.2) / 1.5)
        issues.append("Translation contains potential unsupported additions or padding.")
        
    # --- FACTOR 7: Missing Important Meaning (0.0 to 1.0) ---
    # - Checks if the core words / expectations are present
    missing_score = 1.0
    if ratio < 0.5:
        missing_score = max(0.1, ratio / 0.5)
        issues.append("Translation is abnormally short, missing critical parts of meaning.")
    elif cleaned_translation == str(source_text).strip():
        missing_score = 0.0
        issues.append("Translation is identical to source text (no translation was performed).")
        
    # Calculate final confidence score dynamically
    # Weights sum to 1.0:
    # Quality: 0.20, Semantic: 0.20, IKS Preservation: 0.15, Context: 0.15, Historical: 0.10, Additions: 0.10, Meaning: 0.10
    weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10]
    scores = [quality_score, semantic_score, iks_score, context_score, historical_score, unsupported_score, missing_score]
    
    dynamic_confidence = sum(w * s for w, s in zip(weights, scores))
    dynamic_confidence = round(max(0.1, min(0.99, dynamic_confidence)), 2)
    
    # Status determination based on the dynamic score and issues
    if dynamic_confidence < 0.70 or len(issues) >= 3:
        status = "REVIEW"
    else:
        status = "PASS"
        
    return {
        "validation_status": status,
        "confidence": dynamic_confidence,
        "issues": issues,
        "breakdown": {
            "translation_quality": round(quality_score, 2),
            "semantic_consistency": round(semantic_score, 2),
            "iks_preservation": round(iks_score, 2),
            "context_preservation": round(context_score, 2),
            "historical_consistency": round(historical_score, 2),
            "unsupported_additions": round(unsupported_score, 2),
            "missing_meaning": round(missing_score, 2)
        }
    }

if __name__ == "__main__":
    # Test validator
    source = "துப்பார்க்குத் துப்பாய துப்பாக்கித்"
    trans = "This is a sentence about rain and food."
    print(validate_translation(source, trans, detected_concepts=["Prakriti"], keywords=[]))
