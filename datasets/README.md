# Datasets

Shared datasets for all modules. Mix of synthetic data and real datasets from HuggingFace.

## Quick Start

```bash
# Download all datasets from HuggingFace
python download_datasets.py

# Download specific dataset
python download_datasets.py --dataset tickets
```

## Available Datasets

### Local (Included)

| Dataset | Location | Size | Description |
|---------|----------|------|-------------|
| Tickets Sample | `tickets/sample.csv` | 100 rows | Support tickets for learning |

### HuggingFace (Download Required)

| Dataset | HF Path | Size | Use Case |
|---------|---------|------|----------|
| Support Tickets | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | 27K | Classification, search |
| Banking77 | `PolyAI/banking77` | 13K | Intent classification |
| Financial News | `ashraq/financial-news` | 300K | Sentiment, NER |
| Emails | `aeslc` | 18K | Summarization |
| Stack Overflow | `koutch/stackoverflow_python` | 2M | Q&A, search |

## Directory Structure

```
datasets/
├── README.md
├── download_datasets.py
├── tickets/
│   ├── sample.csv          # Small sample (included)
│   ├── data.parquet        # Full dataset (after download)
│   └── data.jsonl
├── documents/
│   ├── sample_pdfs/        # Sample PDFs for testing
│   └── contracts/
├── embeddings/
│   ├── tickets_gte_base.npy    # Pre-computed embeddings
│   └── tickets_ids.json
└── sql/
    ├── schema.sql          # Database schema
    └── tickets.db          # SQLite database
```

## Format Variants

Each dataset is available in multiple formats:

- **CSV** - Human readable, pandas compatible
- **Parquet** - Columnar, efficient for large data
- **JSONL** - One JSON object per line, streaming
- **SQLite** - For SQL practice

## Usage Examples

### Pandas
```python
import pandas as pd

# CSV
df = pd.read_csv("datasets/tickets/sample.csv")

# Parquet (faster for large files)
df = pd.read_parquet("datasets/tickets/data.parquet")

# JSONL
df = pd.read_json("datasets/tickets/data.jsonl", lines=True)
```

### SQL
```python
import sqlite3

conn = sqlite3.connect("datasets/sql/tickets.db")
df = pd.read_sql("SELECT * FROM tickets WHERE status = 'resolved'", conn)
```

### Pre-computed Embeddings
```python
import numpy as np
import json

embeddings = np.load("datasets/embeddings/tickets_gte_base.npy")
with open("datasets/embeddings/tickets_ids.json") as f:
    ids = json.load(f)
```

## Data Sizes

| Size | Rows | Purpose |
|------|------|---------|
| Tiny | 100 | Unit tests, quick iteration |
| Small | 1K | Learning notebooks |
| Medium | 10K | Livecoding tasks |
| Full | 100K+ | Performance testing |

Use `--size` flag with download script:
```bash
python download_datasets.py --size small  # 1K rows
python download_datasets.py --size full   # All data
```
