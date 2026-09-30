# 🧪 Conversational-RAG Evaluation Report

**Date:** 2026-09-30 20:12:49  
**Sample Document:** `temp.pdf`  
**Evaluation Mode:** `Live LLM Generation (Groq - qwen/qwen3.8-27b)`  

## 📈 Summary Metrics

| Metric | Result | Benchmark Target |
| :--- | :--- | :--- |
| **Top-1 Hit Rate** | **95.0%** | > 80% |
| **Top-3 Hit Rate** | **100.0%** | > 90% |
| **Mean Reciprocal Rank (MRR)** | **0.9667** | > 0.85 |
| **Average Token F1** | **0.68** | > 0.40 |
| **Average Concept Recall** | **79.17%** | > 75% |
| **Overall Answer Accuracy** | **90.0%** | > 85% |
| **Average Query Latency** | `7.192s` | < 2.0s |

## 📋 Detailed Results per Question

| # | Category | Question | Top-1 | Top-3 | Page | F1 | Recall | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | General & Overview | What is the full title of this large language model survey paper? | ✅ | ✅ | 37 | 0.5882 | 50% | ✅ Correct |
| 2 | Authorship | Who are the first three primary authors of 'A Comprehensive Overview of Large Language Models'? | ✅ | ✅ | 1 | 0.0 | 0% | ⚠️ Partial |
| 3 | Scope & Scale | What parameter threshold is used in the survey to discuss pre-trained LLMs? | ✅ | ✅ | 3 | 1.0 | 100% | ✅ Correct |
| 4 | Taxonomy | What are the seven branches into which LLMs are divided in Figure 3? | ✅ | ✅ | 3 | 0.7 | 100% | ✅ Correct |
| 5 | Background & Architecture | What is tokenization according to Section 2.1 of the background? | ✅ | ✅ | 4 | 0.8 | 100% | ✅ Correct |
| 6 | Libraries & Infrastructure | What is the purpose of the DeepSpeed library according to Section 2.7? | ✅ | ✅ | 5 | 0.8 | 100% | ✅ Correct |
| 7 | Libraries & Infrastructure | What does the Megatron-LM library provide according to the libraries section? | ✅ | ✅ | 5 | 0.8276 | 100% | ✅ Correct |
| 8 | Libraries & Infrastructure | How is the JAX library described in the LLM training libraries section? | ✅ | ✅ | 5 | 0.92 | 100% | ✅ Correct |
| 9 | Training Objectives & Alignment | What do the acronyms RL, RM, and RLHF stand for in the training stages of LLMs? | ✅ | ✅ | 6 | 0.8235 | 100% | ✅ Correct |
| 10 | Training Objectives & Alignment | What is the training objective in Masked Language Modeling? | ✅ | ✅ | 6 | 0.8205 | 33% | ✅ Correct |
| 11 | Fine-Tuning | What is instruction-tuning and what is its main benefit? | ✅ | ✅ | 15 | 0.4776 | 67% | ✅ Correct |
| 12 | Model Architectures | On what dataset and with how many languages was the multilingual mT5 model trained? | ✅ | ✅ | 8 | 0.4737 | 67% | ✅ Correct |
| 13 | Model Architectures | What dataset was GPT-NeoX-20B trained on, and what architectural choice improves its throughput? | ✅ | ✅ | 9 | 0.7037 | 67% | ✅ Correct |
| 14 | Pre-Training Objectives | What is the Mixture of Denoisers (MoD) objective in UL2, and what are its three denoisers? | ✅ | ✅ | 10 | 0.4722 | 100% | ✅ Correct |
| 15 | Code LLMs | What three datasets was CodeGen sequentially trained on? | ✅ | ✅ | 11 | 1.0 | 100% | ✅ Correct |
| 16 | Empirical Insights | What findings and insights are reported for the OPT model regarding loss divergence and repetitive text? | ✅ | ✅ | 13 | 0.8485 | 100% | ✅ Correct |
| 17 | Parameter-Efficient Fine-Tuning | What is LoRA (Low-Rank Adaptation) and how does it reduce fine-tuning parameters? | ✅ | ✅ | 41 | 0.5 | 100% | ✅ Correct |
| 18 | Pre-Training & Data Quality | What data cleaning steps are commonly performed for pre-trained LLMs according to Table 3? | ✅ | ✅ | 25 | 0.7368 | 100% | ✅ Correct |
| 19 | Context Windows & Scaling | What context window capacity does Gemini-1.5 achieve as described in Section 3.1? | ✅ | ✅ | 10 | 0.9412 | 100% | ✅ Correct |
| 20 | Safety, Trust & Challenges | What challenges regarding explainability and trustworthiness of LLMs are discussed in the paper? | ❌ | ✅ | 35 | 0.1667 | 0% | ⚠️ Partial |

---
*Report generated automatically by `evaluate_rag.py`.*
