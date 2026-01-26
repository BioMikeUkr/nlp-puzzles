# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NLP/ML learning curriculum with 16 modules covering practical skills for ML engineering roles. The repository contains:
- Learning notebooks with explanations
- Livecoding tasks (empty cells + asserts for validation)
- Solution notebooks (fully solved versions)
- System design problems (50 end-to-end scenarios)

## Repository Structure

```
nlp-puzzles/
├── modules/                      # 16 learning modules
│   └── XX_module_name/
│       ├── README.md             # Theory + 30 deep Q&A
│       ├── requirements.txt      # Module-specific dependencies
│       ├── fixtures/input/       # Sample data (CSV, JSON, Parquet)
│       ├── learning/             # Tutorial notebooks
│       ├── tasks/                # Empty livecoding tasks
│       └── solutions/            # Solved versions of tasks
├── datasets/                     # Shared datasets
│   └── download_datasets.py      # HuggingFace downloader
└── system_design_problems/       # 50 architecture scenarios
```

## Commands

### Download datasets from HuggingFace
```bash
cd datasets
python download_datasets.py                    # Download all (small)
python download_datasets.py --dataset tickets  # Specific dataset
python download_datasets.py --size full        # Full size (large)
```

### Install module dependencies
```bash
pip install -r modules/01_python_data_fundamentals/requirements.txt
```

### Run Jupyter notebooks
```bash
jupyter notebook modules/01_python_data_fundamentals/learning/
```

## Key Patterns

### Livecoding Task Structure
Tasks have empty cells for solutions with assert statements for validation:
```python
# YOUR CODE HERE

# TEST - Do not modify
assert len(df) == 10, f"Expected 10, got {len(df)}"
```

### Module README Q&A Format
Each module README contains 30 deep questions with detailed answers covering:
- Architecture & Design (10 questions)
- Implementation & Coding (10 questions)
- Debugging & Troubleshooting (5 questions)
- Trade-offs & Decisions (5 questions)

### Fixtures Convention
```
fixtures/
├── input/       # Raw input data
├── expected/    # Expected outputs for validation
└── edge_cases/  # Tricky examples
```

## Module Topics (in recommended order)

1. Python & Data Fundamentals (pandas, regex, JSON)
2. Text Embeddings & Semantic Search (sentence-transformers)
3. Vector Databases (FAISS, ChromaDB)
4. Cross-Encoders & NLI (reranking, zero-shot)
5. RAG (retrieval-augmented generation)
6. LLM APIs (OpenAI, Pydantic)
7. Fine-tuning (sentence-transformers)
8. ML Metrics (precision, recall, F1)
9. MLOps (MLflow, Docker)
10. Document Automation (PDF, OCR)
11. Data Formats & SQL (Parquet, PostgreSQL)
12. FastAPI for ML
13. Testing & Code Quality (pytest)
14. Spark Basics
15. Prompt Engineering
16. LangChain & Orchestration
