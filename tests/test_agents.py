import unittest
import sys
import os

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.concept_agent import detect_concepts
from agents.retrieval_agent import retrieve_knowledge
from agents.historical_agent import get_historical_context
from agents.context_agent import analyze_context
from agents.validation_agent import validate_translation

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.sample_text = "வேண்டுதல் வேண்டாமை இலானடி சேர்ந்தார்க்கு யாண்டும் இடும்பை இல."
        self.sample_id = "IKS000004"
        
    def test_concept_agent(self):
        """Checks concept extraction on sample text."""
        res = detect_concepts(self.sample_text, source_lang="Tamil")
        self.assertIn("concepts", res)
        self.assertGreater(len(res["concepts"]), 0, "No concepts extracted on standard verse.")
        
    def test_retrieval_agent_leakage_prevention(self):
        """Verifies that retrieval agent excludes the exact record ID and matching source text."""
        res = retrieve_knowledge(
            self.sample_text, 
            detected_concepts=["Vairagya", "Moksha"], 
            keywords=["desire", "sorrow"], 
            source_lang="Tamil", 
            exclude_id=self.sample_id
        )
        self.assertIn("knowledge", res)
        for doc in res["knowledge"]:
            self.assertNotEqual(doc["id"], self.sample_id, "Target leakage detected: exact record ID retrieved.")
            self.assertNotEqual(doc["original_text"].strip(), self.sample_text.strip(), "Target leakage detected: identical text retrieved.")
            
    def test_historical_agent(self):
        """Validates period matching logic for known documents."""
        res = get_historical_context(self.sample_text, source_doc="Thirukkural")
        self.assertEqual(res["period"], "Thirukkural")
        self.assertIn("Sangam", res["context_description"])
        
    def test_context_agent(self):
        """Checks interpretation generation logic based on mock retrieved knowledge."""
        mock_retrieved = [{
            "id": "IKS000012",
            "source": "Thirukkural",
            "original_text": "துப்பார்க்குத் துப்பாய துப்பாக்கித்",
            "english_translation": "Rain produces food...",
            "iks_concepts": "Prakriti, Dharma",
            "historical_period": "Sangam",
            "relevance_score": 0.95
        }]
        mock_hist = {"period": "Sangam"}
        res = analyze_context(self.sample_text, ["Aram"], mock_retrieved, mock_hist, ["refuge", "desire"])
        self.assertIn("interpretation", res)
        self.assertIn("appropriate_meaning", res)
        self.assertGreater(res["confidence"], 0.0)
        
    def test_validation_agent(self):
        """Audits validation rules against standard translations."""
        trans = "Those who seek refuge at the feet of God will never know sorrow."
        res = validate_translation(self.sample_text, trans, detected_concepts=["Vairagya", "Bhakti"], keywords=["desire"])
        self.assertIn("validation_status", res)
        self.assertIn("confidence", res)
        self.assertIn("issues", res)

if __name__ == "__main__":
    unittest.main()
