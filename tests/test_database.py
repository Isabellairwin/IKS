import unittest
import sys
import os

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb import get_mongodb_client, get_iks_collection
from database.queries import fetch_record_by_id

class TestDatabase(unittest.TestCase):
    def test_connection(self):
        """Tests that a connection can be established with MongoDB."""
        try:
            client = get_mongodb_client()
            self.assertIsNotNone(client)
        except Exception as e:
            self.fail(f"MongoDB connection failed: {e}")
            
    def test_collection_retrieval(self):
        """Tests that records can be retrieved from the collection."""
        try:
            coll = get_iks_collection()
            self.assertIsNotNone(coll)
            count = coll.count_documents({})
            self.assertGreater(count, 0, "Collection contains zero records.")
        except Exception as e:
            self.fail(f"Collection retrieval failed: {e}")
            
    def test_query_by_id(self):
        """Tests fetching a single document by its unique IKS ID."""
        try:
            doc = fetch_record_by_id("IKS000004")
            self.assertIsNotNone(doc, "Document IKS000004 was not found.")
            self.assertEqual(doc["id"], "IKS000004")
            self.assertIn("tamil_text", doc)
        except Exception as e:
            self.fail(f"Query by ID failed: {e}")

if __name__ == "__main__":
    unittest.main()
