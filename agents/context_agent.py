import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.queries import fetch_record_by_id

def analyze_context(text, detected_concepts, retrieved_knowledge, historical_context, keywords):
    """
    Analyzes the contextual interpretation and appropriate meaning of the input text 
    by grounding it in related records retrieved from the database.
    """
    # Guard clause if no retrieved knowledge is available
    if not retrieved_knowledge or len(retrieved_knowledge) == 0:
        return {
            "interpretation": "No related context available to formulate an interpretation.",
            "appropriate_meaning": "General translation based on literal text",
            "supporting_evidence": "No supporting evidence found in the corpus.",
            "confidence": 0.50
        }
        
    # Find the best matching retrieved record based on concept overlap
    best_match = None
    max_overlap = -1
    input_concepts = set([c.lower() for c in detected_concepts])
    
    for r in retrieved_knowledge:
        # Check overlapping concepts
        r_concepts = set([c.strip().lower() for c in r.get("iks_concepts", "").split(",") if c.strip()])
        overlap = len(input_concepts.intersection(r_concepts))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = r
            
    if not best_match:
        best_match = retrieved_knowledge[0]
        
    # Fetch original document details from database to avoid schema gaps
    try:
        doc_details = fetch_record_by_id(best_match["id"])
    except Exception as e:
        print(f"Warning: Failed to fetch record by ID: {e}", file=sys.stderr)
        doc_details = best_match
        
    meaning = doc_details.get("meaning_summary", "Not available")
    context_notes = doc_details.get("context", "Not available")
    notes = doc_details.get("translation_notes", "Not available")
    
    # 1. Determine contextual interpretation
    concepts_str = ", ".join(detected_concepts) if detected_concepts else "literal translation"
    interpretation = (
        f"Interpreted in the framework of '{concepts_str}' from the era of '{historical_context.get('period')}', "
        f"this verse emphasizes: {meaning}"
    )
    
    # 2. Select appropriate meaning (e.g. for Aram, Dharma)
    appropriate_meaning = "Moral virtue and righteousness"
    for c in detected_concepts:
        c_lower = c.lower()
        if "aram" in c_lower or "dharma" in c_lower:
            section = doc_details.get("section", "")
            chapter = doc_details.get("chapter", "")
            if "Adhikaram 1" in section or "Katavul" in chapter:
                appropriate_meaning = "Divine righteousness, cosmic order (Dharma), and spiritual duty."
            elif "Adhikaram 9" in section or "Kalavaamai" in chapter:
                appropriate_meaning = "Ethical restraint, personal integrity, and moral discipline."
            else:
                appropriate_meaning = "Social righteousness, virtue, and ethical conduct in daily life."
                break
        elif "anbu" in c_lower:
            appropriate_meaning = "Affection, pure love, and emotional attachment."
            break
        elif "arul" in c_lower:
            appropriate_meaning = "Universal grace, compassion, and divine benevolence."
            break
            
    # 3. Compile supporting evidence
    supporting_evidence = (
        f"Usage parallel to {best_match['source']} (ID: {best_match['id']}). "
        f"Contextual metadata: {context_notes} | Translation notes: {notes}"
    )
    
    # Calculate confidence score
    avg_relevance = sum([r.get("relevance_score", 0.5) for r in retrieved_knowledge]) / len(retrieved_knowledge)
    confidence = round(0.40 + avg_relevance * 0.60, 2)
    
    return {
        "interpretation": interpretation,
        "appropriate_meaning": appropriate_meaning,
        "supporting_evidence": supporting_evidence,
        "confidence": confidence,
        "best_match_id": best_match["id"]
    }

if __name__ == "__main__":
    # Test context analysis
    res = analyze_context("வேண்டுதல் வேண்டாமை", ["Aram", "Vairagya"], 
                          [{"id": "IKS000004", "source": "Thirukkural", "relevance_score": 0.95, "historical_period": "Sangam"}], 
                          {"period": "Sangam"}, ["refuge", "desire"])
    print(res)
