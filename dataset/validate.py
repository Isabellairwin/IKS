import os
import json
import sys

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.queries import fetch_all_records

def validate_dataset():
    """
    Validates the MongoDB dataset for integrity, duplicates, missing data, 
    invalid Unicode, and consistent metadata. Outputs results to JSON.
    """
    try:
        records = fetch_all_records()
    except Exception as e:
        print(f"Error fetching records for validation: {e}")
        sys.exit(1)
        
    total_records = len(records)
    
    valid_count = 0
    duplicate_records_count = 0
    missing_source_text_count = 0
    missing_target_text_count = 0
    missing_metadata_count = 0
    invalid_unicode_count = 0
    invalid_lang_count = 0
    corrupted_count = 0
    
    seen_ids = set()
    seen_source_texts = set()
    seen_pairs = set()
    
    validation_issues = []
    
    # Required core fields
    core_fields = ["id", "source_document", "tamil_text", "english_translation", "source_language"]
    metadata_fields = ["iks_concepts", "keywords", "meaning_summary", "context", "historical_period"]
    
    for r in records:
        record_id = r.get("id", "UNKNOWN")
        is_valid = True
        record_issues = []
        
        # 1. Corrupted records (missing core fields)
        for cf in core_fields:
            if cf not in r or r[cf] is None:
                record_issues.append(f"Missing core field: {cf}")
                is_valid = False
                
        if not is_valid:
            corrupted_count += 1
            validation_issues.append({"id": record_id, "issues": record_issues})
            continue
            
        # Extract text content
        source_text = str(r.get("tamil_text", "")).strip()
        target_text = str(r.get("english_translation", "")).strip()
        source_lang = str(r.get("source_language", "")).strip()
        
        # 2. Duplicate records (by ID or text)
        is_dup = False
        if record_id in seen_ids:
            record_issues.append(f"Duplicate ID: {record_id}")
            is_dup = True
        else:
            seen_ids.add(record_id)
            
        if source_text in seen_source_texts:
            record_issues.append("Duplicate source text")
            is_dup = True
        else:
            seen_source_texts.add(source_text)
            
        pair = (source_text, target_text)
        if pair in seen_pairs:
            record_issues.append("Duplicate source-target pair")
            is_dup = True
        else:
            seen_pairs.add(pair)
            
        if is_dup:
            duplicate_records_count += 1
            is_valid = False
            
        # 3. Missing source text
        if not source_text:
            record_issues.append("Empty source text")
            missing_source_text_count += 1
            is_valid = False
            
        # 4. Missing target text
        if not target_text:
            record_issues.append("Empty English translation")
            missing_target_text_count += 1
            is_valid = False
            
        # 5. Invalid Unicode
        for f, val in [("tamil_text", source_text), ("english_translation", target_text)]:
            try:
                val.encode('utf-8').decode('utf-8')
            except UnicodeError:
                record_issues.append(f"Invalid Unicode encoding in field: {f}")
                invalid_unicode_count += 1
                is_valid = False
                
        # 6. Invalid language labels
        if source_lang not in ["Tamil", "Sanskrit"]:
            record_issues.append(f"Invalid source language label: {source_lang}")
            invalid_lang_count += 1
            is_valid = False
            
        # 7. Missing metadata fields
        missing_meta_fields = []
        for mf in metadata_fields:
            val = r.get(mf)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing_meta_fields.append(mf)
                
        if missing_meta_fields:
            record_issues.append(f"Missing metadata fields: {', '.join(missing_meta_fields)}")
            missing_metadata_count += 1
            
        if not is_valid:
            validation_issues.append({"id": record_id, "issues": record_issues})
        else:
            valid_count += 1
            
    # Compile final JSON report
    report = {
        "Total records": total_records,
        "Valid records": valid_count,
        "Duplicate records": duplicate_records_count,
        "Missing source text": missing_source_text_count,
        "Missing target text": missing_target_text_count,
        "Missing metadata": missing_metadata_count,
        "Invalid Unicode records": invalid_unicode_count,
        "Invalid language label records": invalid_lang_count,
        "Corrupted records": corrupted_count,
        "Issues detailed": validation_issues
    }
    
    # Save the report
    os.makedirs("results", exist_ok=True)
    report_path = "results/dataset_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        
    print(f"Validation completed. Saved report to {report_path}")
    print(f"Total: {total_records} | Valid: {valid_count} | Duplicates: {duplicate_records_count} | Missing Source: {missing_source_text_count} | Missing Target: {missing_target_text_count} | Missing Metadata: {missing_metadata_count}")
    return report

if __name__ == "__main__":
    validate_dataset()
