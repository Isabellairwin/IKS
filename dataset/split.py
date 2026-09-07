import os
import json
import sys
import pandas as pd
from datetime import datetime

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.queries import fetch_all_records

def split_dataset(seed=42, train_pct=0.8, val_pct=0.1, test_pct=0.1):
    """
    Shuffles and splits the dataset into train, validation, and test splits (80/10/10).
    Filters duplicates dynamically to prevent leakage between sets.
    """
    try:
        records = fetch_all_records()
    except Exception as e:
        print(f"Error fetching records for splitting: {e}")
        sys.exit(1)
        
    # Convert to DataFrame
    df = pd.DataFrame(records)
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
        
    print(f"Loaded {len(df)} records from MongoDB.")
    
    # Deduplicate based on source text (tamil_text) to prevent leakage
    initial_len = len(df)
    df = df.drop_duplicates(subset=['tamil_text']).reset_index(drop=True)
    dedup_len = len(df)
    print(f"Removed {initial_len - dedup_len} duplicate source text records. Unique count: {dedup_len}")
    
    # Shuffle with fixed seed
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split bounds
    n = len(df_shuffled)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    
    train_df = df_shuffled.iloc[:train_end]
    val_df = df_shuffled.iloc[train_end:val_end]
    test_df = df_shuffled.iloc[val_end:]
    
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save splits as CSVs with UTF-8 encoding
    train_df.to_csv("data/train.csv", index=False, encoding="utf-8")
    val_df.to_csv("data/validation.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/test.csv", index=False, encoding="utf-8")
    
    # Calculate distributions
    lang_dist = {
        "train": train_df["source_language"].value_counts().to_dict(),
        "validation": val_df["source_language"].value_counts().to_dict(),
        "test": test_df["source_language"].value_counts().to_dict()
    }
    
    # Compile metadata
    metadata = {
        "random_seed": seed,
        "total_records": n,
        "split_percentages": {
            "train": train_pct,
            "validation": val_pct,
            "test": test_pct
        },
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "language_distribution": lang_dist,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = "data/split_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Saved splits: Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Split metadata saved to {metadata_path}")
    return metadata

if __name__ == "__main__":
    split_dataset()
