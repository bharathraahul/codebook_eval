# codebook_eval

Reproducing and extending [Hu et al. (ACL 2024)](https://arxiv.org/pdf/2308.07876) — zero-shot political event classification on the PLOVER ontology using NLI baselines and LLM-based methods.

Built on the original [Zero-Shot-PLOVER](https://github.com/snowood1/Zero-Shot-PLOVER) codebase.

## Repository Structure

```
codebook_eval/
├── PLOVER_Experiments.py      # Main experiment script (rootcode classification)
├── PLOVER_Bin_Quad.py         # Direct binary and quadcode classification
├── PLOVER_rag.py              # RAG v1 (FAISS + sentence-transformers)
├── PLOVER_rag_v2.py           # RAG v2 (flat and hierarchical retrieval)
├── parse_cameo_to_json.py     # Parses CAMEO manual PDF into plover_codebook.json
├── main_script.py             # Original ZSP paper code (NLI baselines)
├── utils.py                   # Utility functions for ZSP
├── datasets/
│   ├── PLV_test.tsv           # PLOVER test set (1033 examples)
│   └── AW_test.tsv            # A/W dataset
├── codebooks/                 # CAMEO and PLOVER ontology codebooks
└── prompts/
    ├── Tree.txt               # NLI prompt (tree structure)
    ├── Tiny.txt               # NLI prompt (tiny)
    └── Full.txt               # NLI prompt (full)
```

## Setup

```bash
# Python dependencies
pip install transformers torch pandas scikit-learn requests

# For RAG experiments (optional)
pip install sentence-transformers faiss-cpu

# Install Ollama (https://ollama.com)
# Then pull models:
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

## Experiments

### PLOVER_Experiments.py (Rootcode Classification)

The main script supports 7+ methods via `--step`:

| --step | Description |
|--------|-------------|
| tree | ZSP Tree NLI baseline (online, needs GPU) |
| tiny | ZSP Tiny NLI baseline |
| full | ZSP Full NLI baseline |
| llm_no_cb | LLM, no codebook (label list only) |
| llm_cb | LLM + original codebook (v1) |
| llm_cb_v2 | LLM + enriched codebook v2 (best results) |
| llm_json_cb | LLM + JSON-structured codebook (requires plover_codebook.json) |
| llm_cot | LLM + chain-of-thought |
| llm_icl | LLM + in-context learning (8 examples) |
| table | Print comparison table from saved outputs |

```bash
# Run with default model (gemma2:9b)
python3 PLOVER_Experiments.py --step llm_cb_v2

# Quick test (5 examples)
python3 PLOVER_Experiments.py --step llm_cb_v2 --limit 5

# Run all LLM experiments
python3 PLOVER_Experiments.py --step llm_all

# Print results table
python3 PLOVER_Experiments.py --step table
```

### PLOVER_Bin_Quad.py (Direct Binary and Quadcode)

Classifies directly at binary (COOPERATION/CONFLICT) and quadcode (4 classes) levels instead of deriving them from rootcode predictions.

| --step | Description |
|--------|-------------|
| llm_bin_no_cb | Binary classification, no codebook |
| llm_bin_cb | Binary classification, with codebook |
| llm_bin_cot | Binary classification, chain-of-thought |
| llm_bin_icl | Binary classification, in-context learning |
| llm_quad_no_cb | Quadcode classification, no codebook |
| llm_quad_cb | Quadcode classification, with codebook |
| llm_quad_cot | Quadcode classification, chain-of-thought |
| llm_quad_icl | Quadcode classification, in-context learning |

```bash
python3 PLOVER_Bin_Quad.py --step llm_bin_cb
python3 PLOVER_Bin_Quad.py --step llm_quad_cb
```

### RAG Experiments

**PLOVER_rag.py (v1)** uses FAISS with three retrieval strategies:

| --step | Description |
|--------|-------------|
| rag_cb | Retrieve codebook definitions only |
| rag_cb_ex | Retrieve codebook + labeled examples |
| rag_noisy | Retrieve with noise preprocessing |

```bash
python3 PLOVER_rag.py --step rag_cb
python3 PLOVER_rag.py --step rag_noisy
```

**PLOVER_rag_v2.py** adds flat and hierarchical retrieval strategies:

| --step | Description |
|--------|-------------|
| rag_flat_cb | Flat retrieval, codebook only (top-k definitions) |
| rag_flat_cb_ex | Flat retrieval, codebook + examples |
| rag_hier | Hierarchical (predict quadcode first, then rootcode) |

```bash
python3 PLOVER_rag_v2.py --step rag_hier
```

### JSON Codebook (optional)

The `--step llm_json_cb` in PLOVER_Experiments.py requires `plover_codebook.json`. Generate it first:

```bash
python3 parse_cameo_to_json.py
```

This parses the CAMEO manual PDF from `codebooks/` into a structured JSON with label, definition, clarification, negative clarification, positive example, and negative example per rootcode.

## Results

All results on PLV_test (1033 examples) using Gemma2:9b via Ollama.

### Rootcode Classification (main results)

| Method | Binary F1 | Quad F1 | Root F1 | Avg |
|--------|-----------|---------|---------|-----|
| LLM CB v2 (enriched) | 95.8 | 86.8 | 71.4 | 84.6 |
| LLM CB v1 (original) | 92.8 | 80.3 | 63.8 | 79.0 |
| LLM JSON CB (CAMEO) | 91.6 | 73.3 | 55.7 | 73.6 |
| LLM No Codebook | 91.0 | 65.7 | 52.1 | 69.6 |
| LLM ICL | — | — | — | — |
| LLM CoT | — | — | — | — |
| [Paper] GPT-4 | 93.4 | 76.7 | 61.5 | 77.2 |
| [Paper] GPT-3.5 | 90.1 | 66.2 | 40.9 | 65.7 |
| [Paper] ZSP Tree | 96.4 | 89.6 | 82.4 | 89.5 |

### Codebook Comparison Across Models

| Method | Binary F1 | Quad F1 | Root F1 | Avg |
|--------|-----------|---------|---------|-----|
| Gemma2:9b CB v2 | 95.8 | 86.8 | 71.4 | 84.6 |
| Gemma2:9b CB v1 | 92.8 | 80.3 | 63.8 | 79.0 |
| Qwen2.5:7b CB v2 | 90.1 | 79.6 | 60.8 | 76.8 |
| Qwen2.5:7b CB v1 | 89.6 | 76.5 | 58.8 | 75.0 |

### Direct Binary and Quadcode Classification

| Method | Binary F1 | | Method | Quad F1 |
|--------|-----------|-|--------|---------|
| LLM Bin CB | 94.0 | | LLM Quad CB | 78.4 |
| LLM Bin No CB | 90.5 | | LLM Quad ICL | 76.2 |
| LLM Bin ICL | 90.5 | | LLM Quad No CB | 64.3 |
| LLM Bin CoT | 52.4 | | LLM Quad CoT | 19.2 |

### RAG Experiments

| Method | Binary F1 | Quad F1 | Root F1 | Avg |
|--------|-----------|---------|---------|-----|
| RAG v1 Noisy | 94.4 | 80.6 | 67.8 | 80.9 |
| RAG v1 CB + Examples | 93.5 | 78.5 | 66.3 | 79.4 |
| RAG v1 CB Only | 92.0 | 78.1 | 65.4 | 78.5 |
| RAG v2 Hierarchical | 93.3 | 79.9 | 63.8 | 79.0 |
| RAG v2 Flat (CB+Ex) | 61.3 | 62.5 | 57.4 | 60.4 |
| RAG v2 Flat (CB) | 61.0 | 61.3 | 59.3 | 60.5 |

### ZSP NLI Baselines (reproduced)

| Method | Binary F1 | Quad F1 | Root F1 |
|--------|-----------|---------|---------|
| ZSP Tree (L1) | 90.3 | 72.2 | 28.9 |
| ZSP Tree (L2) | 90.0 | 74.2 | 30.8 |

## Key Findings

- Enriched codebook v2 (with examples and disambiguation rules) significantly outperforms the original codebook at all three levels (+3.0 Binary, +6.5 Quad, +7.6 Root F1)
- Gemma2:9b + CB v2 beats GPT-4 at all three levels (95.8 vs 93.4 Binary, 86.8 vs 76.7 Quad, 71.4 vs 61.5 Root)
- Text format codebook outperforms structured JSON codebook (84.6 vs 73.6 avg F1)
- Gemma2:9b consistently outperforms Qwen2.5:7b across all conditions
- RAG v1 Noisy beats GPT-4 and CB v1 but does not surpass CB v2, meaning enriched static definitions outperform dynamic retrieval
- RAG v2 flat retrieval collapses because excluding definitions hurts more than focused context helps
- CoT degrades sharply when prompts are long, collapsing at both binary and rootcode levels
- Codebook consistently improves performance; ICL is competitive with codebook; CoT is unreliable

## Codebook Variants

The enriched codebook v2 (`CODEBOOK_V2` in PLOVER_Experiments.py) adds the following over the original v1:
- Concrete examples for every rootcode (e.g. "agreed to cooperate," "offered to negotiate")
- Disambiguation rules for common confusion patterns (future tense, negated cooperation, peacekeeping, CONSULT narrowing)
- Quadcode grouping headers (Q1-Verbal Cooperation, Q2-Material Cooperation, Q3-Verbal Conflict, Q4-Material Conflict)

## Citation

Based on:
```
@inproceedings{hu2024leveraging,
  title={Leveraging Codebook Knowledge with NLI and ChatGPT for Zero-Shot Political Relation Classification},
  author={Hu, Yibo and Parolin, Erick Skorupa and Khan, Latifur and Brandt, Patrick and Osorio, Javier and D'Orazio, Vito},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={583--603},
  year={2024}
}
```
