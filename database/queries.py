from database.mongodb import get_iks_collection

def fetch_all_records():
    """
    Retrieves all records from the collection.
    """
    collection = get_iks_collection()
    return list(collection.find({}))

def fetch_record_by_id(doc_id):
    """
    Retrieves a single record by its IKS ID.
    """
    collection = get_iks_collection()
    return collection.find_one({"id": doc_id})

def fetch_records_by_language(lang):
    """
    Retrieves all records for a specific source language.
    """
    collection = get_iks_collection()
    return list(collection.find({"source_language": lang}))

def search_by_iks_concepts(concepts, source_lang=None, exclude_id=None, limit=5):
    """
    Retrieves records matching one or more IKS concepts.
    If exclude_id is provided, it excludes that specific record (prevents target leakage).
    """
    collection = get_iks_collection()
    query = {}
    
    if source_lang:
        query["source_language"] = source_lang
        
    if exclude_id:
        query["id"] = {"$ne": exclude_id}
        
    if concepts:
        or_clauses = []
        for c in concepts:
            c_clean = c.strip()
            if c_clean:
                or_clauses.append({"iks_concepts": {"$regex": rf"\b{c_clean}\b", "$options": "i"}})
        if or_clauses:
            query["$or"] = or_clauses
    
    # If no concepts provided, or empty query clauses, return empty list
    if not query.get("$or") and not source_lang and not exclude_id:
        return []
        
    return list(collection.find(query).limit(limit))

def search_by_keywords(keywords, source_lang=None, exclude_id=None, limit=5):
    """
    Retrieves records matching one or more keywords.
    If exclude_id is provided, it excludes that specific record.
    """
    collection = get_iks_collection()
    query = {}
    
    if source_lang:
        query["source_language"] = source_lang
        
    if exclude_id:
        query["id"] = {"$ne": exclude_id}
        
    if keywords:
        or_clauses = []
        for kw in keywords:
            kw_clean = kw.strip()
            if kw_clean:
                or_clauses.append({"keywords": {"$regex": rf"\b{kw_clean}\b", "$options": "i"}})
        if or_clauses:
            query["$or"] = or_clauses
            
    # If no keywords provided, or empty query clauses, return empty list
    if not query.get("$or") and not source_lang and not exclude_id:
        return []
        
    return list(collection.find(query).limit(limit))
