import unittest
import os
import json
import sys
import pandas as pd

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.validate import validate_dataset
from dataset.split import split_dataset

class TestDataset(unittest.TestCase):
    def test_dataset_validation(self):
        """Runs the validation logic and checks for the output JSON report."""
        report = validate_dataset()
        self.assertIsNotNone(report)
        self.assertTrue(os.path.exists("results/dataset_validation_report.json"))
        
        with open("results/dataset_validation_report.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("Total records", data)
            self.assertIn("Valid records", data)
            self.assertIn("Duplicate records", data)
            self.assertEqual(data["Total records"], report["Total records"])
            
    def test_dataset_splitting(self):
        """Runs the split engine and verifies that the output splits match configuration ratios."""
        metadata = split_dataset(seed=42)
        self.assertIsNotNone(metadata)
        
        self.assertTrue(os.path.exists("data/train.csv"))
        self.assertTrue(os.path.exists("data/validation.csv"))
        self.assertTrue(os.path.exists("data/test.csv"))
        self.assertTrue(os.path.exists("data/split_metadata.json"))
        
        train = pd.read_csv("data/train.csv")
        val = pd.read_csv("data/validation.csv")
        test = pd.read_csv("data/test.csv")
        
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, metadata["total_records"])
        self.assertGreater(len(train), len(val))
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

if __name__ == "__main__":
    unittest.main()
