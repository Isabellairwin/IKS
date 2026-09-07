# IKS Time-Travel Translator
### An IKS-Aware Multi-Agent Neural Machine Translation System for Classical Tamil and Sanskrit-to-English Translation (Phase 1)

---

## 1. Project Title
**IKS Time-Travel Translator: An IKS-Aware Multi-Agent Neural Machine Translation System for Classical Tamil and Sanskrit-to-English Translation**

---

## 2. Project Overview
The **IKS Time-Travel Translator** is a context-aware machine translation system designed to translate historical Indian texts—specifically Classical Tamil and Sanskrit—into English. The system goes beyond standard dictionary NMT by leveraging a **Multi-Agent pipeline** that detects IKS (Indian Knowledge Systems) philosophical concepts, retrieves historical/cultural context from a curated corpus, and audits translations using a validation agent. 

IndicTrans2 is the core NMT engine for this project.

---

## 3. Problem Statement
Classical Indian literatures (like Sangam Tamil and ancient Sanskrit literature) are rich in philosophical, metaphysical, and cultural concepts (such as *Aram/Dharma*, *Vairagya*, *Bhakti*, *Akam*, and *Puram*). These terms carry dense, context-dependent meanings that cannot be translated literally. Standard NMT systems translate these terms using generic dictionary definitions (e.g., translating *Aram* simply as "virtue"), stripping away the deep contextual and philosophical dimensions of the original verses.

---

## 4. Research Objective
To build and evaluate a translation framework that:
1. Translates Classical Tamil and Sanskrit texts accurately to English using **IndicTrans2**.
2. Incorporates an **IKS-Aware Multi-Agent framework** that retrieves corpus context to resolve ambiguities and preservation gaps.
3. Automatically validates translation quality, ensuring critical cultural and philosophical concepts are not lost.
4. Compares the performance of three systems: Pre-trained Baseline NMT, Fine-Tuned NMT, and the IKS-Aware NMT pipeline.

---

## 5. Dataset
The dataset comprises 3,000+ curated records containing classical verses, reference translations, and descriptive metadata:
*   **Unique source texts** from works such as the *Thirukkural*, *Purananuru*, and *Ainkurunuru*.
*   **English Reference Translations** mapped to each verse.
*   **Rich Metadata:** Document, author, chapter, historical period, keywords, IKS concepts, context, and translation notes.

---

## 6. MongoDB Atlas
The dataset is hosted on MongoDB Atlas in the database `iks_db` under the collection `tamil_to_english_dataset`. The collection contains 3,480 Tamil records, each with completed fields for translation notes, meaning summaries, difficulty levels, and IKS concepts.

---

## 7. System Architecture

The translation pipeline operates in two modes:

### Mode 1: Baseline NMT
```
Input Verse -> Pre-trained IndicTrans2 -> English Translation
```

### Mode 2: IKS-Aware NMT (Multi-Agent Pipeline)
```
             USER INPUT
                 ↓
         Language Detection
                 ↓
     IKS Concept Detection Agent
                 ↓
     IKS Knowledge Retrieval Agent (leakage-free)
                 ↓
         Historical Context
                 ↓
       Context Analysis Agent
                 ↓
         Fine-Tuned IndicTrans2
                 ↓
          Validation Agent  ← (Recovery/enrichment loop if validation fails)
                 ↓
      FINAL ENGLISH TRANSLATION
```

---

## 8. IKS Agents
*   **IKS Concept Detection Agent (`agents/concept_agent.py`):** Scans the input text to identify core concepts (e.g., *Aram*, *Ozhukkam*, *Vairagya*) using predefined vocabularies and corpus associations.
*   **IKS Knowledge Retrieval Agent (`agents/retrieval_agent.py`):** Queries MongoDB Atlas for similar verses and historical notes. It implements a strict **leakage blocker** (filtering by target record ID and identical text) to ensure the reference translation of a test sentence is never leaked during evaluation.
*   **Historical Context Component (`agents/historical_agent.py`):** Resolves the historical period (e.g., *Sangam*, *Post-Sangam*) based on reference mappings and retrieved metadata.
*   **Context Analysis Agent (`agents/context_agent.py`):** Determines the context-appropriate sense of ambiguous terms (e.g., deciding whether *Aram* means "cosmic order" or "daily discipline" based on its chapter placement).
*   **Validation Agent (`agents/validation_agent.py`):** Performs post-translation quality checks on completeness, non-ASCII script bleed, and concept preservation. If concepts are missing, it triggers an enrichment loop to inject contextual definitions.

---

## 9. IndicTrans2
The system utilizes AI4Bharat's **IndicTrans2** sequence-to-seq model. The distilled 200M parameter model (`ai4bharat/indictrans2-indic-en-dist-200M`) is used as the default engine to fit within local resource constraints. It maps language script tags (`tam_Taml` for Tamil and `san_Deva` for Sanskrit) and translates into English (`eng_Latn`).

---

## 10. Baseline NMT
Implemented under `experiments/indictrans2_baseline/`, this configuration runs pre-trained IndicTrans2 directly on inputs without IKS retrieval, contextual definitions, or agent validation, establishing a direct NMT baseline.

---

## 11. Fine-Tuning
The fine-tuning pipeline is implemented in `training/fine_tune_indictrans2.py`. It trains the model on the training split, using validation splits for model selection, and saves weights to `models/indictrans2/finetuned/`. It automatically adapts to hardware constraints, running on CPU/GPU depending on VRAM availability.

---

## 12. Evaluation Configuration
Evaluation is executed by `evaluation/evaluate.py` on the untouched test split (10% of the dataset) for all three configurations under identical conditions.

---

## 13. Metrics
We implement five primary evaluation dimensions:
1.  **BLEU:** Corpus-level lexical translation precision.
2.  **chrF++:** Character n-gram F-score (highly effective for morphologically rich Indian languages).
3.  **COMET:** Neural machine translation evaluation (optional fallback if hardware restricts download).
4.  **Semantic Similarity:** Cosine similarity of TF-IDF vectors between the predicted and reference translations.
5.  **IKS Preservation Score:** Ratio of reference IKS concepts preserved in the translated text.
6.  **Context Preservation Score:** Ratio of metadata keywords preserved in the output.

---

## 14. Results
Corpus-level evaluation scores are compiled automatically into:
*   `results/results.csv`: Flat text evaluation summary.
*   `results/comparison.xlsx`: Formatted Excel spreadsheet.
*   `results/per_sample_results.csv`: Per-record predictions and metric outputs.
*   `results/error_analysis.csv`: Categorization of error patterns.
*   `results/figures/`: Generated comparison charts for all metrics.

---

## 15. Installation

1.  **Clone the project** and move into the workspace:
    ```bash
    cd c:\Users\isabe\OneDrive\Desktop\IKS
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 16. Environment Variables
Create a `.env` file in the root directory:
```env
MONGODB_URI=your_mongodb_connection_string_here
HF_TOKEN=your_huggingface_user_token_here
```
*Note: Gaining access to IndicTrans2 model weights requires logging into HuggingFace, accepting terms on the model cards, and generating a user token.*

---

## 17. Running the Project

### Step 1: Pre-process and Split the Dataset
```bash
python dataset/split.py
```
This script connects to MongoDB Atlas, deduplicates records, shuffles them with a fixed seed (42), and saves splits to `data/train.csv`, `data/validation.csv`, and `data/test.csv`.

### Step 2: Run Dataset Validation
```bash
python dataset/validate.py
```
Outputs an audit report to `results/dataset_validation_report.json`.

### Step 3: Run Model Fine-Tuning
```bash
python training/fine_tune_indictrans2.py
```
Executes training and saves checkpoints to `models/indictrans2/finetuned/`.

### Step 4: Run Evaluation and Compare
```bash
python evaluation/evaluate.py
python evaluation/compare.py
```
Calculates scores, compiles Excel sheets, generates error analyses, and exports figures to `results/figures/`.

### Step 5: Start Streamlit Web UI
```bash
streamlit run app.py
```

---

## 18. Limitations
*   **GPU VRAM Constraints:** Local fine-tuning is adjusted to subsample/run on CPU or low-VRAM settings to prevent CUDA Out Of Memory errors on 2GB cards.
*   **Sanskrit Data:** The MongoDB dataset currently contains only Classical Tamil records. Sanskrit translation runs zero-shot using the pre-trained IndicTrans2 model without fine-tuning or retrieval contexts.

---

## 19. Future Model Comparison
The architecture is designed to be modular. Future translators can be registered in `translation/registry.py` under the same interface (e.g., `models/model2/`, `experiments/model2/`) without modifying the IndicTrans2 code. The comparative evaluation framework in `evaluation/` can be reused directly to update the Excel comparison spreadsheets and graphs.
