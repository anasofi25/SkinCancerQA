# SkinCancerQA

A multi-agent question answering system for skin cancer education, combining a formally structured OWL ontology with locally-hosted LLMs.

## System Overview

User questions are processed through a three-agent pipeline:

1. **Agent 1** — translates natural language into SPARQL queries using few-shot prompted LLaMA 3.2
2. **Agent 2** — validates query results using a hybrid rule-based and LLM-based approach
3. **Agent 3** — explains ontology results in plain English (four interchangeable variants)

All inference runs locally via Ollama or LM Studio — no API keys required.

## Agent 3 Variants

| Variant | Model | Fine-tuned | Backend |
|---|---|---|---|
| A | LLaMA 3.2 | No | Ollama |
| B | MedGemma | No | Ollama |
| C | LLaMA 3.2 | Yes (LoRA) | Ollama |
| D | MedGemma | Yes (LoRA) | LM Studio |

## Requirements

- Python 3.13
- [Ollama](https://ollama.com) with `llama3.2`, `medgemma`, `skincancer-llama`
- [LM Studio](https://lmstudio.ai) with `medgemma-skincancer` (for Variant D)
- [GraphDB Free](https://graphdb.ontotext.com) with `skin_cancer_expanded` repository

## Installation

```bash
pip install -r requirements.txt
```

Load the ontology into GraphDB:
1. Create repository named `skin_cancer_expanded`
2. Import `skinCancerExpanded.ttl`

## Running

```bash
streamlit run VariantA/app.py
```

## Evaluation

```bash
python VariantA/run_all_testing_pipeline2.py   # 25 clinical + edge questions
python VariantA/run_basic_q.py                  # 15 basic ontology questions
python VariantA/full_evaluation.py              # metrics + plots
python VariantA/evaluate_agent2.py              # Agent 2 confusion matrix
```

Results saved to `Results/testing_pipeline2.csv`.


```
skin_cancer_qaV@/
├── VariantA/
│   ├── agent_gen_query.py          # Agent 1 — SPARQL generator
│   ├── agent_validator.py          # Agent 2 — result validator
│   ├── agent_explainer2.py         # Agent 3 — answer explainer (4 variants)
│   ├── pipeline_v2.py              # Pipeline orchestrator
│   ├── app.py                      # Streamlit web interface
│   ├── run_all_testing_pipeline2.py # Evaluation — 25 questions
│   ├── run_basic_q.py              # Evaluation — 15 basic questions
│   ├── full_evaluation.py          # Metrics + plots
│   ├── evaluate_agent2.py          # Agent 2 confusion matrix
│   └── evaluation_plots/           # Generated evaluation figures
├── Dataset/
│   ├── generate_dataset.py         # QA pair generation with Gemini
│   ├── prep.py                     # Format to Alpaca JSONL
│   ├── training_data_gemini.jsonl  # Training dataset (1,000 entries)
│   └── training_data_gemini.json
├── Results/
│   ├── testing_pipeline2.csv       # Full evaluation results (160 runs)
│   ├── testing_pipeline2_enriched.csv
│   ├── agent2_classifier_results.csv
│   └── evaluation_report.txt
└── README.md
```
## Dataset

Training data generated from [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) using Gemini 2.5 Flash-Lite. 1,000 Alpaca-format QA pairs in `Dataset/training_data_gemini.jsonl`.

## Tech Stack

Python · PyTorch · HuggingFace Transformers · PEFT/LoRA · GraphDB · SPARQL · OWL · Streamlit · Ollama · LM Studio
