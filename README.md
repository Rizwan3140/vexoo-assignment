# Vexoo Labs AI Engineer Assignment
**Author:** Rizwan Ansar

---

## Repository Structure

```
rizwan_vexoo_assignment/
├── part1_ingestion.py          # Part 1: Sliding Window + Knowledge Pyramid
├── part2_gsm8k_finetune.ipynb  # Part 2: GSM8K Fine-tuning with LoRA
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Requirements

**Python:** 3.9+  
**GPU:** Required for Part 2. Part 1 runs on CPU.

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Part 1: Document Ingestion System

**File:** `part1_ingestion.py`

### What it does
1. Accepts a PDF file and extracts text using PyMuPDF
2. Splits text into overlapping sliding windows snapped to sentence boundaries
3. Builds a 4-layer Knowledge Pyramid per chunk:
   - **Layer 0 — Raw Text:** Original chunk
   - **Layer 1 — Summary:** Query-relevant sentences using NLTK
   - **Layer 2 — Category:** Rule-based keyword classification (10 categories)
   - **Layer 3 — Keywords:** Top-10 keywords using TF-IDF
4. Accepts a query and retrieves the most relevant chunks using cosine similarity

### Run

```bash
python part1_ingestion.py document.pdf
```

The script will then prompt you to enter queries interactively. Type `exit` to quit.

---

## Part 2: GSM8K Fine-tuning

**File:** `part2_gsm8k_finetune.ipynb`

### What it does
- Loads GSM8K from HuggingFace (3000 train / 1000 eval samples)
- Formats samples into instruction-style prompts
- Tokenizes with prompt masking — loss only on answer tokens
- Applies LoRA adapters (r=16, target: `q_proj`, `v_proj`)
- Trains using HuggingFace Trainer with fp16 and gradient accumulation
- Evaluates using Exact Match on extracted `####` answers

### Run
Open the notebook in **Google Colab** with a T4 GPU:

1. Runtime → Change runtime type → T4 GPU
2. Run cells one by one
3. Login with your HuggingFace token when prompted

### Results
| Model | Train Samples | Eval Samples | Exact Match Accuracy |
|---|---|---|---|
| LLaMA 3.2 1B | 3000 | 100 | 13% |