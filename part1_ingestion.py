import sys
import re
import nltk
import fitz  # pymupdf
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# One-time downloads (safe to run multiple times)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    print(f"[PDF] Loaded '{path}' — {len(doc)} pages, {len(full_text)} chars")
    doc.close()
    return full_text


def sliding_window_chunks(text: str, window_size: int = 2500, overlap: int = 300) -> List[Dict[str, Any]]:
    # Split text into overlapping 2-page chunks at sentence boundaries
    sentences = sent_tokenize(text.strip())

    chunks = []
    idx = 0
    i = 0

    while i < len(sentences):
        chunk_sentences = []
        char_count = 0

        j = i
        while j < len(sentences) and char_count < window_size:
            chunk_sentences.append(sentences[j])
            char_count += len(sentences[j])
            j += 1

        chunk_text = " ".join(chunk_sentences)
        chunks.append({
            "id": idx,
            "text": chunk_text,
        })

        idx += 1

        overlap_chars = 0
        step = j - i
        for k in range(i, j):
            overlap_chars += len(sentences[k])
            if overlap_chars >= (window_size - overlap):
                step = k - i + 1
                break

        i += max(1, step)

    return chunks


def summarize_chunk(text: str, query: str = "", n_sentences: int = 3) -> str:
    # Return query-relevant sentences as summary
    sentences = sent_tokenize(text.strip())
    if not sentences:
        return text[:300]

    if query:
        # Score each sentence by how many query words it contains
        query_words = set(query.lower().split())
        scored = sorted(
            sentences,
            key=lambda s: sum(1 for w in query_words if w in s.lower()),
            reverse=True
        )
        return " ".join(scored[:n_sentences])

    return " ".join(sentences[:n_sentences])


# Rule-based category keywords map
CATEGORY_RULES = {
    "machine_learning": ["model", "training", "neural", "deep learning", "gradient", "loss", "accuracy", "dataset"],
    "nlp": ["text", "token", "embedding", "language", "sentence", "nlp", "bert", "gpt", "transformer"],
    "retrieval": ["retrieval", "search", "query", "index", "similarity", "vector", "rag", "fetch"],
    "data_engineering": ["pipeline", "ingestion", "chunk", "window", "document", "process", "extract"],
    "mathematics": ["equation", "theorem", "proof", "formula", "calculate", "algebra", "geometry"],
    "finance": ["revenue", "profit", "loss", "market", "stock", "investment", "budget", "cost"],
    "healthcare": ["patient", "disease", "treatment", "medicine", "clinical", "hospital", "diagnosis"],
    "legal": ["law", "contract", "rights", "court", "clause", "regulation", "compliance", "statute"],
    "cybersecurity": ["attack", "vulnerability", "encryption", "firewall", "malware", "threat", "breach"],
    "cloud": ["aws", "azure", "docker", "kubernetes", "deployment", "server", "microservice"],
    "general": []
}

def classify_chunk(text: str) -> str:
    # Rule-based category detection using keyword frequency
    text_lower = text.lower()
    scores = {
        category: sum(text_lower.count(kw) for kw in keywords)
        for category, keywords in CATEGORY_RULES.items()
        if keywords
    }

    if not scores or max(scores.values()) == 0:
        return "general"

    return max(scores, key=scores.get)


def distill_keywords(text: str, top_n: int = 10) -> List[str]:
    # TF-IDF based top-N keyword extraction
    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        max_features=top_n,
        token_pattern=r'\b[a-zA-Z]{4,}\b'  # only words 4+ chars
    )

    try:
        vectorizer.fit([text])
        return vectorizer.get_feature_names_out().tolist()
    except ValueError:
        return text.split()[:top_n]


def build_pyramid(chunk: Dict[str, Any]) -> Dict[str, Any]:
    # Build all 4 layers for a single chunk
    text = chunk["text"]
    return {
        "id": chunk["id"],
        "layer_0_raw": text,
        "layer_1_summary": summarize_chunk(text),
        "layer_2_category": classify_chunk(text),
        "layer_3_keywords": distill_keywords(text),
    }

def retrieve(query: str, pyramid_index: List[Dict[str, Any]], top_k: int = 3,) -> List[Dict[str, Any]]:
    # Rank chunks by cosine similarity to query
    corpus = []
    for pyramid in pyramid_index:
        combined = " ".join([
            pyramid["layer_0_raw"],
            pyramid["layer_1_summary"],
            " ".join(pyramid["layer_3_keywords"]),
        ])
        corpus.append(combined)

    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))
    all_texts = corpus + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    scores = cosine_similarity(query_vec, doc_vecs).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        pyramid = pyramid_index[idx]
        results.append({
            "score": round(float(scores[idx]), 4),
            "chunk_id": pyramid["id"],
            "category": pyramid["layer_2_category"],
            "summary": summarize_chunk(pyramid["layer_0_raw"], query=query),
            "keywords": pyramid["layer_3_keywords"],
            "raw_snippet": pyramid["layer_0_raw"][:300] + "...",
        })

    return results

def ingest_document(text: str) -> List[Dict[str, Any]]:

    chunks = sliding_window_chunks(text)

    pyramid_index = []
    for chunk in chunks:
        pyramid = build_pyramid(chunk)
        pyramid_index.append(pyramid)
    return pyramid_index



### MAIN EXECUTION

if __name__ == "__main__":
    # Usage: python part1_ingestion.py document.pdf

    if len(sys.argv) < 2:
        print("[Info] No document provided.")
        print("[Info] Usage: python part1_ingestion.py document.pdf")
        sys.exit(0)

    filepath = sys.argv[1]

    if not filepath.lower().endswith(".pdf"):
        print(f"[Warning] Unsupported file format: '{filepath}'")
        print("[Warning] Please provide a PDF file only.")
        print("[Warning] Usage: python part1_ingestion.py document.pdf")
        sys.exit(0)

    text = load_pdf(filepath)

    # Ingest document
    index = ingest_document(text)

    # Query loop — keep asking until user types 'exit'
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = retrieve(query, index, top_k=2)

        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score={r['score']}) ---")
            print(f"  Category : {r['category']}")
            print(f"  Keywords : {r['keywords']}")
            print(f"  Summary  : {r['summary']}")
            print(f"  Snippet  : {r['raw_snippet']}\n")