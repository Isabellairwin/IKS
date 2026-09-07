import unittest
import sys
import os

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translation.registry import ModelRegistry
from translation.indictrans2 import IndicTrans2Translator
from evaluation.metrics import calculate_bleu, calculate_chrf

class TestPipeline(unittest.TestCase):
    def test_registry_and_loading(self):
        """Verifies that the translation registry loads translation engines correctly."""
        translator = ModelRegistry.get_translator("indictrans2", mode="baseline")
        self.assertIsInstance(translator, IndicTrans2Translator)
        
    def test_translation_execution(self):
        """Checks end-to-end execution of baseline translations."""
        translator = ModelRegistry.get_translator("indictrans2", mode="baseline")
        text = "துப்பார்க்குத் துப்பாய துப்பாக்கித்"
        translation = translator.translate(text, source_language="Tamil")
        self.assertIsNotNone(translation)
        self.assertGreater(len(translation), 0)
        self.assertIn("rain", translation.lower())
        
    def test_metrics_calculation(self):
        """Tests calculation of sacreBLEU and chrF++ metrics on matching sets."""
        preds = ["This is a test translation."]
        refs = ["This is a test translation."]
        bleu = calculate_bleu(preds, refs)
        chrf = calculate_chrf(preds, refs)
        
        self.assertGreater(bleu, 90.0)
        self.assertGreater(chrf, 90.0)

if __name__ == "__main__":
    unittest.main()
