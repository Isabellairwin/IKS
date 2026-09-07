import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.queries import search_by_iks_concepts, search_by_keywords

def retrieve_knowledge(text, detected_concepts=None, keywords=None, source_lang="Tamil", exclude_id=None, limit=3):
    """
    Retrieves contextually relevant verses and metadata from MongoDB.
    Prevents exact-record leakage by explicitly filtering out the record being translated
    both by its ID and by checking for identical source text.
    """
    cleaned_text = str(text).strip()
    retrieved = []
    seen_ids = set()
    
    if exclude_id:
        seen_ids.add(exclude_id)
        
    # 1. Search by IKS concepts first
    if detected_concepts:
        try:
            concept_matches = search_by_iks_concepts(detected_concepts, source_lang=source_lang, exclude_id=exclude_id, limit=limit*3)
            for doc in concept_matches:
                doc_id = doc.get("id")
                doc_text = str(doc.get("tamil_text", "")).strip()
                
                # Check for ID deduplication and source text identity leakage
                if doc_id not in seen_ids and doc_text != cleaned_text:
                    # Calculate relevance score based on overlap of concepts
                    doc_concepts_str = doc.get("iks_concepts", "")
                    doc_concepts = [c.strip() for c in doc_concepts_str.split(",") if c.strip()]
                    
                    overlap = set(detected_concepts).intersection(set(doc_concepts))
                    score = 0.5 + (len(overlap) / max(len(detected_concepts), 1)) * 0.5
                    
                    retrieved.append({
                        "id": doc_id,
                        "source": doc.get("source_document", "Unknown"),
                        "original_text": doc.get("tamil_text", ""),
                        "english_translation": doc.get("english_translation", ""),
                        "iks_concepts": doc_concepts_str,
                        "historical_period": doc.get("historical_period", "Unknown"),
                        "context": doc.get("context", ""),
                        "relevance_score": round(min(score, 1.0), 2)
                    })
                    seen_ids.add(doc_id)
        except Exception as e:
            print(f"Warning: Concept search failed: {e}", file=sys.stderr)
            
    # 2. Search by keywords if we need more context
    if len(retrieved) < limit and keywords:
        try:
            kw_matches = search_by_keywords(keywords, source_lang=source_lang, exclude_id=exclude_id, limit=limit*3)
            for doc in kw_matches:
                doc_id = doc.get("id")
                doc_text = str(doc.get("tamil_text", "")).strip()
                
                if doc_id not in seen_ids and doc_text != cleaned_text:
                    # Calculate relevance score based on overlap of keywords
                    doc_kws_str = doc.get("keywords", "")
                    doc_kws = [k.strip().lower() for k in doc_kws_str.split(",") if k.strip()]
                    overlap = set([k.lower() for k in keywords]).intersection(set(doc_kws))
                    score = 0.3 + (len(overlap) / max(len(keywords), 1)) * 0.7
                    
                    retrieved.append({
                        "id": doc_id,
                        "source": doc.get("source_document", "Unknown"),
                        "original_text": doc.get("tamil_text", ""),
                        "english_translation": doc.get("english_translation", ""),
                        "iks_concepts": doc.get("iks_concepts", ""),
                        "historical_period": doc.get("historical_period", "Unknown"),
                        "context": doc.get("context", ""),
                        "relevance_score": round(min(score, 1.0), 2)
                    })
                    seen_ids.add(doc_id)
        except Exception as e:
            print(f"Warning: Keyword search failed: {e}", file=sys.stderr)
            
    # Sort results by relevance score and limit output
    retrieved = sorted(retrieved, key=lambda x: x["relevance_score"], reverse=True)[:limit]
    
    if not retrieved:
        return {
            "knowledge": [],
            "summary": "No relevant IKS knowledge found."
        }
        
    # Compile a human-readable summary of the retrieved records
    summary_parts = []
    for idx, r in enumerate(retrieved):
        summary_parts.append(
            f"Knowledge Source {idx+1}: [{r['source']}]\n"
            f"  Verse: {r['original_text'].replace('\n', ' ')}\n"
            f"  Reference English: {r['english_translation']}\n"
            f"  IKS Concepts: {r['iks_concepts']}\n"
            f"  Period: {r['historical_period']}\n"
            f"  Relevance: {r['relevance_score']}"
        )
        
    return {
        "knowledge": retrieved,
        "summary": "\n\n".join(summary_parts)
    }

if __name__ == "__main__":
    # Test retrieval
    res = retrieve_knowledge("துப்பார்க்குத் துப்பாய துப்பாக்கித்", detected_concepts=["Dharma"], keywords=["food", "rain"])
    print(res["summary"])
