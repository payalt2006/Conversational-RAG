#!/usr/bin/env python3
"""
Evaluation Script for Conversational-RAG
=========================================
Tests and validates chatbot retrieval hit rate and answer correctness over a sample PDF
using 15 to 20 benchmark question-answer pairs.

Features:
- Document chunking & retrieval evaluation (Top-1, Top-3, Top-k Hit Rate, MRR).
- Dual mode:
    1. Live LLM Generation & Scoring via Groq (if GROQ_API_KEY is available)
    2. Deterministic Offline Lexical & Semantic Evaluation (F1 Score, Key Fact Recall)
- Generates JSON metrics artifact and formatted Markdown report.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure UTF-8 console compatibility on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Optional dotenv loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# PDF text extraction
try:
    import pypdf
except ImportError:
    pypdf = None

# Retrieval dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    np = None

# Groq client
try:
    from groq import Groq
except ImportError:
    Groq = None


# ==============================================================================
# 1. Text Extraction & Chunking
# ==============================================================================

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text and page numbers from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    if pypdf is None:
        raise ImportError("pypdf is required to extract PDF text. Install it with: pip install pypdf")
    
    reader = pypdf.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def recursive_chunk_text(pages: List[Dict[str, Any]], chunk_size: int = 1500, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Chunks document pages into overlapping text segments with page metadata.
    Emulates RecursiveCharacterTextSplitter behavior.
    """
    chunks = []
    chunk_id = 1
    
    for p in pages:
        text = p["text"]
        page_no = p["page"]
        if not text.strip():
            continue
            
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_content = text[start:end].strip()
            if len(chunk_content) > 40:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_no,
                    "text": chunk_content
                })
                chunk_id += 1
            if end >= len(text):
                break
            start += chunk_size - chunk_overlap
            
    return chunks


# ==============================================================================
# 2. Retriever Engine
# ==============================================================================

class RAGRetriever:
    """Fast, local TF-IDF and Vector retrieval engine for evaluation."""
    
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=15000,
                sublinear_tf=True
            )
            corpus = [c["text"] for c in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks relevant to query."""
        if not self.chunks:
            return []
            
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices, 1):
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(sims[idx])
                chunk["rank"] = rank
                results.append(chunk)
            return results
        else:
            # Fallback keyword overlap ranking
            query_words = set(re.findall(r"\w+", query.lower()))
            scored = []
            for c in self.chunks:
                c_words = set(re.findall(r"\w+", c["text"].lower()))
                overlap = len(query_words & c_words)
                scored.append((overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for rank, (score, c) in enumerate(scored[:top_k], 1):
                chunk = dict(c)
                chunk["score"] = float(score)
                chunk["rank"] = rank
                results.append(chunk)
            return results


# ==============================================================================
# 3. Answer Generation Engine
# ==============================================================================

def resolve_best_groq_model(client: Any, requested_model: Optional[str] = None) -> str:
    """Detects available Groq models and picks the highest-performing available model."""
    preferred_order = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b"
    ]
    try:
        available_models = [m.id for m in client.models.list().data]
        if requested_model and requested_model != "auto" and requested_model in available_models:
            return requested_model
        for model in preferred_order:
            if model in available_models:
                return model
        return available_models[0] if available_models else "qwen/qwen3.8-27b"
    except Exception:
        return requested_model if (requested_model and requested_model != "auto") else "qwen/qwen3.8-27b"


def generate_rag_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    groq_api_key: Optional[str] = None,
    model_name: str = "auto",
    client: Optional[Any] = None
) -> Tuple[str, float]:
    """
    Generates answer using Groq LLM if API key is provided, or extractive RAG fallback.
    Returns (answer_text, latency_seconds).
    """
    context = "\n\n---\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in retrieved_chunks])
    t0 = time.time()
    
    if (groq_api_key or client) and Groq is not None:
        try:
            active_client = client or Groq(api_key=groq_api_key)
            actual_model = model_name if model_name != "auto" else resolve_best_groq_model(active_client)
            system_prompt = (
                "You are an assistant for answering questions. "
                "Use the retrieved context to provide accurate responses. "
                "If you don't know the answer, say so. "
                "Keep answers concise (max 3 sentences).\n\n"
                f"Context:\n{context}"
            )
            chat_completion = active_client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=256
            )
            answer = chat_completion.choices[0].message.content.strip()
            latency = time.time() - t0
            return answer, latency
        except Exception as e:
            print(f" [!] Groq generation warning: {e}", flush=True)

    # Extractive offline fallback: extract most relevant sentences from top chunk
    latency = time.time() - t0
    if not retrieved_chunks:
        return "No relevant context found.", latency
        
    top_text = retrieved_chunks[0]["text"]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_text) if len(s.strip()) > 15]
    q_words = set(re.findall(r"\w+", query.lower()))
    
    scored_sentences = []
    for s in sentences:
        s_words = set(re.findall(r"\w+", s.lower()))
        score = len(q_words & s_words)
        scored_sentences.append((score, s))
        
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best_sentences = [s for _, s in scored_sentences[:2] if _ > 0]
    
    if best_sentences:
        answer = " ".join(best_sentences)
    else:
        answer = top_text[:250] + "..."
        
    return answer, latency


# ==============================================================================
# 4. Evaluation Metrics
# ==============================================================================

def normalize_text(s: str) -> List[str]:
    """Tokenize and normalize text."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction)
    gt_tokens = normalize_text(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(gt_tokens))
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def compute_key_phrase_recall(prediction: str, key_phrases: List[str]) -> float:
    """Calculates proportion of key ground-truth phrases present in prediction."""
    if not key_phrases:
        return 1.0
    pred_lower = prediction.lower()
    matches = sum(1 for phrase in key_phrases if phrase.lower() in pred_lower)
    return matches / len(key_phrases)


def evaluate_retrieval_hit(
    retrieved_chunks: List[Dict[str, Any]],
    key_phrases: List[str],
    expected_page: Optional[int]
) -> Tuple[bool, int, Optional[int]]:
    """
    Checks if retrieved chunks contain relevant ground truth facts.
    Returns (hit_found, first_hit_rank, hit_page).
    """
    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_text = chunk["text"].lower()
        phrase_matches = sum(1 for p in key_phrases if p.lower() in chunk_text)
        page_matches = (expected_page is not None and chunk.get("page") == expected_page)
        
        if phrase_matches >= 1 or page_matches:
            return True, rank, chunk.get("page")
            
    return False, 0, None


# ==============================================================================
# 5. Main Benchmark Runner
# ==============================================================================

def run_evaluation(
    pdf_path: str = "./temp.pdf",
    qa_path: str = "./eval_qa_pairs.json",
    top_k: int = 3,
    groq_api_key: Optional[str] = None,
    model_name: str = "llama-3.1-8b-instant",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    output_json: str = "eval_results.json",
    output_report: str = "eval_report.md"
) -> Dict[str, Any]:
    """Runs the full evaluation benchmark and produces metrics and reports."""

    print("=" * 80)
    print("CONVERSATIONAL-RAG: ACCURACY & RETRIEVAL EVALUATION BENCHMARK")
    print("=" * 80)
    
    # 1. Load QA pairs
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"QA dataset not found: {qa_path}")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"[*] Loaded {len(qa_pairs)} question-answer pairs from: {qa_path}")

    # 2. Extract PDF and chunk
    print(f"[*] Extracting text from PDF: {pdf_path}")
    pages = extract_pdf_pages(pdf_path)
    print(f"[*] Extracted {len(pages)} pages.")
    
    chunks = recursive_chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[*] Created {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap}).")

    # 3. Build retriever
    print(f"[*] Indexing chunks for retrieval...")
    retriever = RAGRetriever(chunks)
    
    # 4. Detect mode
    effective_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    active_model = model_name
    groq_client = None
    if effective_api_key and Groq is not None:
        try:
            groq_client = Groq(api_key=effective_api_key)
            active_model = resolve_best_groq_model(groq_client, model_name)
            mode_str = f"Live LLM Generation (Groq - {active_model})"
        except Exception:
            mode_str = f"Live LLM Generation (Groq - {model_name})"
    else:
        mode_str = "Offline Extractive / Lexical Validation"
    print(f"[*] Evaluation Mode: {mode_str}", flush=True)
    print("-" * 80, flush=True)

    # 5. Execute Evaluation
    results = []
    hits_top1 = 0
    hits_top3 = 0
    hits_top_k = 0
    reciprocal_ranks = []
    f1_scores = []
    concept_recalls = []
    latencies = []
    correct_count = 0

    print(f"{'ID':<3} | {'Top-1':<5} | {'Top-3':<5} | {'Page':<5} | {'F1':<5} | {'Recall':<6} | {'Status':<7} | Question", flush=True)
    print("-" * 80, flush=True)

    for item in qa_pairs:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth_answer"]
        key_phrases = item.get("key_phrases", [])
        expected_page = item.get("expected_page")

        # Retrieval
        retrieved = retriever.retrieve(question, top_k=max(top_k, 5))
        top_k_chunks = retrieved[:top_k]

        is_hit, hit_rank, hit_page = evaluate_retrieval_hit(top_k_chunks, key_phrases, expected_page)
        
        # Rank tracking
        if hit_rank == 1:
            hits_top1 += 1
        if 1 <= hit_rank <= 3:
            hits_top3 += 1
        if is_hit:
            hits_top_k += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Answer generation
        answer, latency = generate_rag_answer(
            query=question,
            retrieved_chunks=top_k_chunks,
            groq_api_key=effective_api_key,
            model_name=active_model,
            client=groq_client
        )
        latencies.append(latency)

        # Accuracy metrics
        f1 = compute_f1_score(answer, ground_truth)
        recall = compute_key_phrase_recall(answer, key_phrases)
        f1_scores.append(f1)
        concept_recalls.append(recall)

        # Correctness threshold (either high concept recall or decent F1 token match)
        is_correct = (recall >= 0.5) or (f1 >= 0.35)
        if is_correct:
            correct_count += 1

        top1_sym = "YES" if hit_rank == 1 else "NO"
        top3_sym = "YES" if 1 <= hit_rank <= 3 else "NO"
        page_sym = str(hit_page) if hit_page else "-"
        status_sym = "CORRECT" if is_correct else "PARTIAL"

        print(f"{qid:<3} | {top1_sym:<5} | {top3_sym:<5} | {page_sym:<5} | {f1:.2f}  | {recall*100:>5.1f}% | {status_sym:<7} | {question[:38]}...", flush=True)

        results.append({
            "id": qid,
            "category": item.get("category", "General"),
            "question": question,
            "ground_truth": ground_truth,
            "key_phrases": key_phrases,
            "expected_page": expected_page,
            "retrieval_hit": is_hit,
            "first_hit_rank": hit_rank,
            "retrieved_page": hit_page,
            "generated_answer": answer,
            "token_f1": round(f1, 4),
            "concept_recall": round(recall, 4),
            "is_correct": is_correct,
            "latency_sec": round(latency, 3)
        })

    # Summary calculations
    total = len(qa_pairs)
    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_pdf": os.path.basename(pdf_path),
        "total_questions": total,
        "evaluation_mode": mode_str,
        "top_1_hit_rate": round(hits_top1 / total * 100, 2),
        "top_3_hit_rate": round(hits_top3 / total * 100, 2),
        f"top_{top_k}_hit_rate": round(hits_top_k / total * 100, 2),
        "mean_reciprocal_rank": round(float(np.mean(reciprocal_ranks) if np else sum(reciprocal_ranks) / total), 4),
        "average_token_f1": round(float(np.mean(f1_scores) if np else sum(f1_scores) / total), 4),
        "average_concept_recall": round(float(np.mean(concept_recalls) if np else sum(concept_recalls) / total) * 100, 2),
        "overall_answer_accuracy": round(correct_count / total * 100, 2),
        "average_latency_sec": round(float(np.mean(latencies) if np else sum(latencies) / total), 3)
    }

    print("=" * 80)
    print("EVALUATION SUMMARY METRICS")
    print("=" * 80)
    print(f"Total Test Questions       : {total}")
    print(f"Top-1 Retrieval Hit Rate   : {summary['top_1_hit_rate']}% ({hits_top1}/{total})")
    print(f"Top-3 Retrieval Hit Rate   : {summary['top_3_hit_rate']}% ({hits_top3}/{total})")
    print(f"Top-{top_k} Retrieval Hit Rate   : {summary[f'top_{top_k}_hit_rate']}% ({hits_top_k}/{total})")
    print(f"Mean Reciprocal Rank (MRR) : {summary['mean_reciprocal_rank']}")
    print(f"Average Token F1 Score     : {summary['average_token_f1']}")
    print(f"Average Key Concept Recall : {summary['average_concept_recall']}%")
    print(f"Overall Answer Accuracy    : {summary['overall_answer_accuracy']}% ({correct_count}/{total})")
    print(f"Average Latency            : {summary['average_latency_sec']}s")
    print("=" * 80)

    # 6. Save JSON Results
    output_data = {
        "summary": summary,
        "results": results
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[OK] Saved JSON results to: {output_json}")

    # 7. Save Markdown Report
    generate_markdown_report(output_data, output_report)
    print(f"[OK] Saved Markdown report to: {output_report}")

    return output_data


def generate_markdown_report(output_data: Dict[str, Any], report_path: str):
    """Generates a GitHub-flavored Markdown evaluation report."""
    summary = output_data["summary"]
    results = output_data["results"]

    md = []
    md.append("# 🧪 Conversational-RAG Evaluation Report\n")
    md.append(f"**Date:** {summary['benchmark_timestamp']}  ")
    md.append(f"**Sample Document:** `{summary['sample_pdf']}`  ")
    md.append(f"**Evaluation Mode:** `{summary['evaluation_mode']}`  \n")
    md.append("## 📈 Summary Metrics\n")
    md.append("| Metric | Result | Benchmark Target |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Top-1 Hit Rate** | **{summary['top_1_hit_rate']}%** | > 80% |")
    md.append(f"| **Top-3 Hit Rate** | **{summary['top_3_hit_rate']}%** | > 90% |")
    md.append(f"| **Mean Reciprocal Rank (MRR)** | **{summary['mean_reciprocal_rank']}** | > 0.85 |")
    md.append(f"| **Average Token F1** | **{summary['average_token_f1']}** | > 0.40 |")
    md.append(f"| **Average Concept Recall** | **{summary['average_concept_recall']}%** | > 75% |")
    md.append(f"| **Overall Answer Accuracy** | **{summary['overall_answer_accuracy']}%** | > 85% |")
    md.append(f"| **Average Query Latency** | `{summary['average_latency_sec']}s` | < 2.0s |\n")

    md.append("## 📋 Detailed Results per Question\n")
    md.append("| # | Category | Question | Top-1 | Top-3 | Page | F1 | Recall | Status |")
    md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        top1 = "✅" if r["first_hit_rank"] == 1 else "❌"
        top3 = "✅" if 1 <= r["first_hit_rank"] <= 3 else "❌"
        page = r["retrieved_page"] or "-"
        status = "✅ Correct" if r["is_correct"] else "⚠️ Partial"
        q_text = r["question"].replace("|", "\\|")
        md.append(f"| {r['id']} | {r['category']} | {q_text} | {top1} | {top3} | {page} | {r['token_f1']} | {r['concept_recall']*100:.0f}% | {status} |")

    md.append("\n---\n*Report generated automatically by `evaluate_rag.py`.*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Conversational-RAG retrieval hit rate and answer accuracy."
    )
    parser.add_argument("--pdf", default="./temp.pdf", help="Path to sample PDF file (default: ./temp.pdf)")
    parser.add_argument("--qa_pairs", default="./eval_qa_pairs.json", help="Path to QA pairs JSON (default: ./eval_qa_pairs.json)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved chunks (default: 3)")
    parser.add_argument("--groq_api_key", default=None, help="Groq API key (defaults to GROQ_API_KEY env var)")
    parser.add_argument("--model", default="auto", help="Groq model (default: auto - selects highest-performing active model)")
    parser.add_argument("--chunk_size", type=int, default=1500, help="Chunk size in characters (default: 1500)")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap in characters (default: 200)")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON path (default: eval_results.json)")
    parser.add_argument("--report", default="eval_report.md", help="Output Markdown report path (default: eval_report.md)")

    args = parser.parse_args()

    run_evaluation(
        pdf_path=args.pdf,
        qa_path=args.qa_pairs,
        top_k=args.top_k,
        groq_api_key=args.groq_api_key,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_json=args.output,
        output_report=args.report
    )


if __name__ == "__main__":
    main()
