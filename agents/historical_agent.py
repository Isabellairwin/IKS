def get_historical_context(text, source_doc=None, retrieved_knowledge=None):
    """
    Determines the historical/cultural context of the input text using known literature
    mappings and retrieved metadata, preventing ungrounded guessing.
    Returns a dictionary with structured fields: source_document, period, author,
    concepts, context_description, supporting_evidence, and confidence.
    """
    # Known author mappings
    AUTHORS = {
        "thirukkural": "Thiruvalluvar",
        "purananuru": "Various poets, compiled by Koodaloor Kizhar",
        "ainkurunuru": "Various poets, compiled by Koodaloor Kizhar, patronized by Yanaikatchai Mantaran Cheral Irumporai",
        "kuruntokai": "Various poets, compiled by Poorko",
        "narrinai": "Various poets, patronized by Pannadu Thantha Pandiyan Maran Valudi",
        "patthuppattu": "Various poets",
        "tolkappiyam": "Tolkappiyar",
        "inna narpathu": "Kapilar",
        "iniyavai narpathu": "Poothanthevanar"
    }
    
    # Mapping of classical works to their historical periods and contexts
    DOCUMENT_PERIODS = {
        "thirukkural": "Post-Sangam period (c. 1st Century BCE to 5th Century CE) - Didactic literature focusing on Aram (virtue), Porul (wealth), and Inbam (love).",
        "purananuru": "Sangam period (c. 3rd Century BCE to 3rd Century CE) - Heroic poetry describing public life, war, valor (Puram), and kingship.",
        "ainkurunuru": "Sangam period (c. 3rd Century BCE to 3rd Century CE) - Love poetry describing internal emotions and landscapes (Akam).",
        "kuruntokai": "Sangam period - Akam poetry focusing on emotional intimacy.",
        "narrinai": "Sangam period - Akam poetry focusing on interior landscapes.",
        "patthuppattu": "Sangam period - Ten long idylls describing landscapes and royal patrons.",
        "tolkappiyam": "Early Sangam period (c. 3rd Century BCE to 1st Century CE) - Grammar, poetics, and socio-cultural conventions.",
        "inna narpathu": "Post-Sangam period (c. 4th Century CE) - Didactic work listing harmful things to avoid.",
        "iniyavai narpathu": "Post-Sangam period (c. 4th Century CE) - Didactic work listing sweet and beneficial things to pursue."
    }
    
    cleaned_text = str(text).strip()
    
    best_ret = None
    if retrieved_knowledge and len(retrieved_knowledge) > 0:
        best_ret = retrieved_knowledge[0]
        
    period = "Historical context unavailable"
    source_document = source_doc if source_doc else "Unknown"
    author = "Unknown"
    concepts = "Unknown"
    context_description = "Historical context unavailable."
    supporting_evidence = "No supporting evidence found in retrieved knowledge."
    confidence = 0.0
    
    if best_ret:
        period = best_ret.get("historical_period", period)
        source_document = best_ret.get("source", source_document)
        concepts = best_ret.get("iks_concepts", concepts)
        supporting_evidence = f"Retrieved parallel verse: '{best_ret.get('original_text', '').replace('\n', ' ')}' -> '{best_ret.get('english_translation', '')}'"
        confidence = 0.85
        
    # Refine using document periods mapping
    doc_key = source_document.lower().strip()
    for k, v in DOCUMENT_PERIODS.items():
        if k in doc_key:
            period = k.title() if period == "Historical context unavailable" else period
            context_description = v
            author = AUTHORS.get(k, author)
            confidence = max(confidence, 0.90)
            break
            
    # Text heuristics fallback
    if period == "Historical context unavailable":
        if "குறள்" in cleaned_text or "வள்ளுவர்" in cleaned_text:
            period = "Post-Sangam"
            source_document = "Thirukkural"
            author = "Thiruvalluvar"
            context_description = DOCUMENT_PERIODS["thirukkural"]
            confidence = 0.75
            
    return {
        "period": period,
        "source_document": source_document,
        "author": author,
        "concepts": concepts,
        "context_description": context_description,
        "supporting_evidence": supporting_evidence,
        "confidence": confidence
    }

if __name__ == "__main__":
    print(get_historical_context("", source_doc="Thirukkural"))
