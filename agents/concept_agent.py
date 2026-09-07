import os
import sys

# Ensure parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb import get_iks_collection

# Dictionary of common Tamil terms to their standard IKS concepts
CONCEPT_DICTIONARY = {
    "அறம்": "Aram / Dharma (Righteousness/Virtue)",
    "அன்பு": "Anbu (Love/Affection)",
    "அருள்": "Arul (Grace/Benevolence)",
    "ஒழுக்கம்": "Ozhukkam / Achara (Discipline/Conduct)",
    "வீரம்": "Veeram (Valor/Heroism)",
    "நீதி": "Neethi (Justice/Moral law)",
    "விருந்தோம்பல்": "Virundhombal (Hospitality)",
    "அகம்": "Akam (Interior/Love life)",
    "புறம்": "Puram (Exterior/Valor/Public life)",
    "கடவுள்": "Bhakti / God (Devotion)",
    "தவம்": "Tapas / Yoga (Penance/Meditation)",
    "கர்ம": "Karma (Action/Consequence)",
    "மோக்ஷ": "Moksha (Liberation)",
    "சத்யம்": "Satya (Truth)",
    "அஹிம்சை": "Ahimsa (Non-violence)",
}

TAMIL_STOPWORDS = {
    'இல', 'இலன்', 'உம்', 'என்', 'அது', 'மற்று', 'ஒரு', 'பல', 'அடி', 'அவன்', 
    'அவள்', 'அவர்', 'அவை', 'என', 'என்று', 'ஆன', 'பட்ட', 'உள்ள', 'ஆம', 'ஏன்', 
    'வந்து', 'இந்த', 'அந்த', 'இருந்து', 'இருந்தது', 'இருக்கிற', 'செய்து', 'செய்த', 
    'ஒருவன்', 'தனது', 'தன்', 'தமது', 'தான்', 'ஆம்'
}

def detect_concepts(text, source_lang="Tamil", exclude_id=None):
    """
    Identifies IKS-related concepts and keywords in the input text.
    Combines direct dictionary matches and dynamic corpus similarity lookup.
    """
    detected = {}
    cleaned_text = str(text).strip()
    
    # 1. Direct dictionary check (Tamil text)
    if source_lang.lower() == "tamil":
        for tamil_term, concept_name in CONCEPT_DICTIONARY.items():
            if tamil_term in cleaned_text:
                detected[concept_name] = {
                    "concept": concept_name,
                    "relevance_score": 1.0,
                    "supporting_evidence": f"Directly matched Tamil keyword '{tamil_term}' in the input text.",
                    "source_document": "Concept Dictionary"
                }
                
    # 2. Tokenize text to extract potential keywords (preserving combining marks)
    import re
    cleaned_text_no_punc = re.sub(r'[.,:;!?\"\'\u0964\u0965\-\(\)]', ' ', cleaned_text)
    words = cleaned_text_no_punc.split()
    
    # Filter keywords (length > 2 and not stop words)
    keywords = []
    for w in words:
        w_clean = w.strip()
        if len(w_clean) > 2 and w_clean not in TAMIL_STOPWORDS:
            keywords.append(w_clean)
            
    if not keywords:
        keywords = [w for w in words if len(w) > 1]
        
    input_words_set = set([kw.lower() for kw in keywords])
    
    # 3. Dynamic lookup: find matching documents in MongoDB to extract concepts
    try:
        collection = get_iks_collection()
        query = {}
        if source_lang:
            query["source_language"] = source_lang
        if exclude_id:
            query["id"] = {"$ne": exclude_id}
            
        clauses = []
        for w in keywords[:6]:
            clauses.append({"tamil_text": {"$regex": re.escape(w), "$options": "i"}})
            
        if clauses:
            query["$or"] = clauses
            matches = list(collection.find(query).limit(5))
            
            for m in matches:
                # Compute overlap score
                match_text = str(m.get("tamil_text", m.get("sanskrit_text", "")))
                match_words_set = set(re.sub(r'[.,:;!?\"\'\u0964\u0965\-\(\)]', ' ', match_text).lower().split())
                match_keywords_set = set([kw.strip().lower() for kw in str(m.get("keywords", "")).split(",") if kw.strip()])
                target_words_set = match_words_set.union(match_keywords_set)
                
                # Filter out stopwords from target words
                target_words_set = {tw for tw in target_words_set if tw not in TAMIL_STOPWORDS and len(tw) > 2}
                
                overlap = input_words_set.intersection(target_words_set)
                overlap_score = len(overlap) / max(len(input_words_set), 1)
                
                # If there's an overlap, extract concepts
                if overlap_score >= 0.20:
                    m_concepts = m.get("iks_concepts", "")
                    if m_concepts:
                        for c in m_concepts.split(","):
                            c_clean = c.strip()
                            if c_clean:
                                # Update if not present or if this match has higher relevance score
                                if c_clean not in detected or overlap_score > detected[c_clean]["relevance_score"]:
                                    detected[c_clean] = {
                                        "concept": c_clean,
                                        "relevance_score": round(overlap_score, 2),
                                        "supporting_evidence": match_text.replace("\n", " ").strip(),
                                        "source_document": str(m.get("source_document", "Unknown"))
                                    }
    except Exception as e:
        print(f"Warning: Dynamic concept detection bypassed: {e}", file=sys.stderr)
        
    # Sort detected concepts by relevance score
    sorted_details = sorted(detected.values(), key=lambda x: x["relevance_score"], reverse=True)
    sorted_concepts = [d["concept"] for d in sorted_details]
    
    if not sorted_details:
        return {
            "concepts": [],
            "details": [],
            "keywords": keywords,
            "confidence": 0.5,
            "explanation": "No specific IKS concepts detected."
        }
        
    return {
        "concepts": sorted_concepts,
        "details": sorted_details,
        "keywords": keywords,
        "confidence": round(max([d["relevance_score"] for d in sorted_details]), 2),
        "explanation": f"Identified concepts: {', '.join(sorted_concepts)}"
    }

if __name__ == "__main__":
    test_text = "வேண்டுதல் வேண்டாமை இலானடி சேர்ந்தார்க்கு யாண்டும் இடும்பை இல."
    print(detect_concepts(test_text))

