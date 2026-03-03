# Module 16: Text Classification with GLiClass

## Overview

GLiClass (Generalist and Lightweight Text Classifier) is a zero-shot text classification model that classifies texts into arbitrary label sets without any task-specific fine-tuning. Unlike traditional classifiers, GLiClass uses a single forward pass to score all labels simultaneously, making it fast and flexible.

In this module you will use `knowledgator/gliclass-edge-v3.0` — a compact edge-optimized model that runs efficiently on CPU/GPU.

---

## Architecture

GLiClass uses a transformer encoder augmented with label-aware attention:

1. **Input**: Text + all candidate labels are concatenated with special delimiters
2. **Encoder**: Bidirectional transformer processes text and labels jointly
3. **Scoring heads**: Separate classification heads for multi-label vs single-label mode
4. **Output**: Probability score for each label in a single forward pass

This differs from NLI-based classifiers (which run N separate forward passes for N labels) and embedding similarity (which requires training on the target domain).

---

## Installation

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
gliclass>=0.1.16
transformers>=4.48.0
torch>=2.0.0
```

---

## Quick Start

### Multi-Label Classification

```python
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch

model = GLiClassModel.from_pretrained("knowledgator/gliclass-edge-v3.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-edge-v3.0", add_prefix_space=True)

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

results = pipeline(
    "Apple unveiled the new M3 MacBook Pro at a special event.",
    ["technology", "sports", "finance", "science"],
    threshold=0.4
)
# results: [[{"label": "technology", "score": 0.92}, ...]]
```

### Single-Label Classification

```python
pipeline_sl = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='single-label',
    device=device
)

result = pipeline_sl("I love this product!", ["positive", "negative", "neutral"])[0]
# result[0]: {"label": "positive", "score": 0.98}
```

### Batch Processing

```python
texts = ["Article 1...", "Article 2...", "Article 3..."]
labels = ["technology", "sports", "finance"]

# All texts in a single call
results = pipeline(texts, labels, threshold=0.4, batch_size=8)
# results: [[labels for text1], [labels for text2], ...]
```

### Fine-Tuning

GLiClass can be fine-tuned on labeled data using `gliclass.training`:

```python
from gliclass.data_processing import GLiClassDataset, DataCollatorWithPadding, AugmentationConfig
from gliclass.training import TrainingArguments, Trainer

# Training data format: {text, all_labels, true_labels}
train_data = [
    {"text": "Apple launched M4 MacBook...", "all_labels": ["technology", "sports"], "true_labels": ["technology"]},
]

augment_config = AugmentationConfig(enabled=False)
train_dataset = GLiClassDataset(
    train_data, tokenizer, augment_config,
    problem_type='multi_label_classification',  # required for proper collation
    architecture_type='uni-encoder', prompt_first=True
)
training_args = TrainingArguments(output_dir="./output", max_steps=150, learning_rate=3e-5, others_lr=1e-4)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=DataCollatorWithPadding())
trainer.train()
```

---

## Key Concepts

### classification_type

| Mode | Use Case | Output |
|------|----------|--------|
| `'multi-label'` | Multiple topics can apply | All labels above threshold |
| `'single-label'` | Mutually exclusive classes | Best label only |

### threshold

- Only relevant in `multi-label` mode
- Labels with score < threshold are dropped
- Default: `0.5`; use `0.0` to get scores for all labels

### Pipeline Output Format

```python
# pipeline(text, labels) returns list of lists (one inner list per input text)
results = pipeline(["text1", "text2"], labels)
# results[0] = [{label: "tech", score: 0.9}, {label: "science", score: 0.7}]
# results[1] = [{label: "sports", score: 0.8}]

# For single text input, still returns a list of lists:
result = pipeline("text", labels)
# result[0] = [{label: "tech", score: 0.9}, ...]
```

---

## Tasks

| Task | Description | Key Concept |
|------|-------------|-------------|
| `task_01_topic_classification.ipynb` | Multi-label + single-label topic classification on news | `multi-label` mode, batch processing |
| `task_02_sentiment_analysis.ipynb` | Sentiment on product reviews, ≥70% accuracy | `single-label` mode, `analyze_sentiment()` |
| `task_03_finetuning.ipynb` | Fine-tune on 50 labeled examples, measure improvement | `GLiClassDataset`, `Trainer`, before/after eval |

---

## Fixtures

| File | Description |
|------|-------------|
| `fixtures/input/news_articles.json` | 10 articles with `domain`, `text`, `expected_topics` |
| `fixtures/input/product_reviews.json` | 15 reviews with `text`, `sentiment` ground truth |
| `fixtures/input/classification_training_data.json` | 50 examples with `text`, `all_labels`, `true_labels` for fine-tuning |

---

## Accuracy Targets

- **Topic classification** (single-label): ≥ 50%
- **Sentiment analysis**: ≥ 70%
- **Fine-tuning**: ≥ 70% after fine-tuning, and ≥10pp improvement over zero-shot (or ≥90% absolute)

---

## Tips

- Always use `add_prefix_space=True` for the tokenizer
- Use `batch_size=4–8` for throughput vs memory trade-off
- GLiClass processes all labels in one forward pass — adding more labels is cheap
- Use `problem_type='multi_label_classification'` in `GLiClassDataset` even for single-label tasks (required for collation)
- For very long texts, consider truncating or chunking before passing to pipeline
