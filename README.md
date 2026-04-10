# Vexoo Labs AI Engineer Assignment
**Author:** Rizwan Ansar  
**Submission:** `rizwan_vexoo_assignment.zip`

---

## Repository Structure

```
rizwan_vexoo_assignment/
├── part1_ingestion.py          # Part 1: Sliding Window + Knowledge Pyramid
├── part2_gsm8k_train.py        # Part 2: GSM8K Fine-tuning with LoRA
├── bonus_reasoning_adapter.py  # Bonus: Reasoning-Aware Adapter
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── summary_report.docx         # 1-page summary report
```

---

## Requirements

**Python:** 3.9+  
**GPU:** Recommended for Part 2 (CUDA 11.8+ or CUDA 12.x). Part 1 and Bonus run on CPU.

### Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
peft>=0.10.0
accelerate>=0.28.0
```

> **Note:** For Part 1 and the Bonus, no GPU or ML libraries are required — they use only Python stdlib.

---

## Part 1: Document Ingestion System

**File:** `part1_ingestion.py`

### What it does
1. Splits a document into overlapping 2-page sliding windows (~2500 chars, 300 char overlap)
2. Builds a 4-layer Knowledge Pyramid per chunk:
   - **Layer 0 — Raw Text:** Original chunk content
   - **Layer 1 — Summary:** First 3 sentences (placeholder)
   - **Layer 2 — Category:** Rule-based keyword classification
   - **Layer 3 — Keywords:** Top-10 TF-based keywords
3. Accepts a query and retrieves top-K most relevant chunks via cosine similarity

### Run

```bash
python part1_ingestion.py
```

**Expected output:**
```
[Ingestion] Document length: 1842 chars
[Ingestion] Created 2 overlapping chunks
  Chunk 00 | Category: machine_learning | Keywords: ['learning', 'model', ...]
  Chunk 01 | Category: retrieval | Keywords: ['retrieval', 'chunk', ...]

Query: How does RAG combine retrieval with language models?
--- Result 1 (score=0.2341) ---
  Category : retrieval
  Keywords : ['retrieval', 'query', 'vector', ...]
  Summary  : Retrieval-Augmented Generation (RAG) combines a retrieval system...
```

### Using with your own document

```python
from part1_ingestion import ingest_document, retrieve

with open("your_document.txt", "r") as f:
    text = f.read()

index = ingest_document(text)
results = retrieve("your query here", index, top_k=3)
```

---

## Part 2: GSM8K Fine-tuning

**File:** `part2_gsm8k_train.py`

### What it does
- Loads GSM8K from HuggingFace (3000 train / 1000 eval samples)
- Tokenizes with LLaMA-compatible tokenizer
- Applies LoRA adapters (r=16, target: `q_proj`, `v_proj`)
- Trains with HuggingFace `Trainer` (fp16, gradient accumulation)
- Evaluates using Exact Match on extracted numeric answers

### Run — Full Training (GPU required)

```bash
python part2_gsm8k_train.py
```

### Run — Dry Run (CPU, no GPU needed)

```bash
python part2_gsm8k_train.py --dry-run
```

### Run — Evaluation Only (on saved checkpoint)

```bash
python part2_gsm8k_train.py --eval-only ./gsm8k_lora_output
```

### Key Configuration (in `TrainConfig`)

| Parameter | Value | Note |
|---|---|---|
| `model_name` | `meta-llama/Llama-3.2-1B` | Requires HF token if gated |
| `train_samples` | 3000 | As specified |
| `eval_samples` | 1000 | As specified |
| `lora_r` | 16 | LoRA rank |
| `num_epochs` | 3 | |
| `learning_rate` | 2e-4 | Standard for LoRA |
| `fp16` | True | Disable on CPU |

### HuggingFace Authentication (if model is gated)

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your HF token from: https://huggingface.co/settings/tokens
```

---

## Bonus: Reasoning-Aware Adapter

**File:** `bonus_reasoning_adapter.py`

### Run

```bash
python bonus_reasoning_adapter.py
```

No dependencies beyond Python stdlib.

**Expected output:**
```
Query: If a train travels 60 km/h for 2.5 hours, how far does it go?
  Detected Domain : math
  Module Used     : math_reasoner
  Strategy        : step_by_step_symbolic
```

---

## Design Decisions

1. **Character-based sliding window** — simpler and more portable than token-based; avoids tokenizer dependency in Part 1
2. **Stub summarization/embeddings** — architecture-first approach; real summarizer (BART/T5) or dense embeddings (sentence-transformers) slot in as drop-in replacements
3. **LoRA over full fine-tuning** — ~2-4M trainable params vs ~1B; practical on consumer GPU (24GB VRAM sufficient)
4. **Rule-based query routing** — zero-latency, zero-cost; upgradeable to a fine-tuned intent classifier without changing interface
5. **Prompt masking in training** — loss computed only on answer tokens, not prompt tokens, for cleaner instruction fine-tuning
