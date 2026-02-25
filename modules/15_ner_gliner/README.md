# Module 15: NER with GLiNER

## Overview

GLiNER (Generalist and Lightweight Named Entity Recognition) is a bidirectional transformer that
extracts entities of *any* type specified at runtime — no retraining required. You pass a list of
entity type descriptions and the model finds matching spans.

This module covers:
- Zero-shot NER with `gliner-bi-edge-v2.0` (bi-encoder architecture)
- PII detection and anonymization with `gliner-pii-edge-v1.0`
- Fine-tuning GLiNER on domain-specific data

### Learning Objectives
- Extract named entities using GLiNER's zero-shot interface
- Pre-compute label embeddings for constant-time batch inference
- Build a PII anonymization pipeline with char-level offset replacement
- Convert character-level annotations to GLiNER's token-level training format
- Fine-tune a bi-encoder model with `model.train_model()`

---

## 1. GLiNER Architecture

### Uni-Encoder vs. Bi-Encoder

The original GLiNER uses a **uni-encoder**: text and entity labels are concatenated and processed
together. This is accurate but slow at scale — with 1000 labels, each forward pass processes the
text + all 1000 label descriptions.

The **bi-encoder** (`gliner-bi-*`) separates the two:

```
Text Encoder (ModernBERT)  →  text embeddings
Label Encoder (MiniLM)     →  label embeddings  ← can be pre-computed
                              ↘ nearest-neighbor matching
```

This enables:
- **Label embedding caching** — compute once, reuse across millions of documents
- **Constant inference time** regardless of entity type count
- **130× throughput** vs. uni-encoder at 1024 entity types

### Entity Output Format

```python
{
    "text":  "Elon Musk",   # matched span text
    "label": "person",      # matched label
    "start": 0,             # char offset start (inclusive)
    "end":   9,             # char offset end (exclusive)
    "score": 0.94           # confidence score [0, 1]
}
```

---

## 2. Zero-Shot Inference

### Single document

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-bi-edge-v2.0")

text = "Tim Cook unveiled Apple Vision Pro in Cupertino on June 5, 2023."
labels = ["person", "organization", "product", "location", "date"]

entities = model.predict_entities(text, labels, threshold=0.3)
for e in entities:
    print(e["text"], "=>", e["label"], f"({e['score']:.2f})")
```

### Batch inference

```python
texts = ["...", "...", "..."]
results = model.inference(texts, labels, threshold=0.3, batch_size=8)
# → list of entity lists, one per text
```

### Pre-computed label embeddings

```python
# Encode labels once
entity_embeddings = model.encode_labels(labels, batch_size=8)

# Reuse for all documents
results = model.batch_predict_with_embeds(texts, entity_embeddings, labels, threshold=0.3)
```

### Threshold guidance

| Use case              | Threshold |
|-----------------------|-----------|
| High precision (demos)| 0.5–0.7  |
| General NER           | 0.3       |
| High recall (compliance) | 0.2   |

---

## 3. PII Detection and Anonymization

### Load PII model

```python
pii_model = GLiNER.from_pretrained("knowledgator/gliner-pii-edge-v1.0")
```

### PII label taxonomy (60+ categories)

| Category     | Labels                                               |
|--------------|------------------------------------------------------|
| Personal     | name, first name, last name, dob, age, gender        |
| Contact      | email address, phone number, ip address, url, location address |
| Financial    | credit card, ssn, account number, routing number, cvv |
| Healthcare   | condition, drug, organization medical facility, name medical professional |
| Identity     | passport number, driver license, username, password, vehicle id |

### Anonymization pattern

```python
def anonymize_text(model, text, labels, threshold=0.3):
    entities = model.predict_entities(text, labels, threshold=threshold)
    # Critical: sort end-to-start to preserve char offsets
    entities.sort(key=lambda e: e["start"], reverse=True)
    result = text
    for e in entities:
        placeholder = "<" + e["label"].upper().replace(" ", "_") + ">"
        result = result[:e["start"]] + placeholder + result[e["end"]:]
    return result
```

**Why end-to-start?** Each replacement changes the string length. Replacing from the right
keeps earlier offsets valid for subsequent replacements.

---

## 4. Fine-Tuning

### Training data format

```python
{
    "tokenized_text": ["Bill", "Gates", "founded", "Microsoft", "in", "1975"],
    "ner": [
        [0, 1, "person"],        # tokens 0–1 = "Bill Gates"
        [3, 3, "organization"],  # token 3 = "Microsoft"
        [5, 5, "date"]           # token 5 = "1975"
    ]
}
```

- `tokenized_text`: **word-level** list (NOT subword tokens)
- `ner` indices are **token-level** (into `tokenized_text`) and **inclusive**
- For bi-encoders, optionally add `ner_labels` (positive) and `ner_negatives` (hard negatives)

### Converting character-level spans

```python
import re

def from_char_annotations(text, char_entities):
    token_matches = list(re.finditer(r'\S+', text))
    tokens = [m.group() for m in token_matches]
    token_spans = [(m.start(), m.end()) for m in token_matches]
    ner = []
    for ent in char_entities:
        toks = [i for i, (ts, te) in enumerate(token_spans)
                if ts >= ent["start"] and te <= ent["end"]]
        if toks:
            ner.append([toks[0], toks[-1], ent["label"]])
    return {"tokenized_text": tokens, "ner": ner}
```

### Training call

```python
train_split = train_data[:int(len(train_data) * 0.8)]
eval_split  = train_data[int(len(train_data) * 0.8):]

trainer = model.train_model(
    train_split,
    eval_split,
    output_dir="./my_model",
    max_steps=2000,
    learning_rate=1e-5,    # lower for text encoder
    others_lr=3e-5,        # higher for non-encoder params
    per_device_train_batch_size=8,
    negatives=1.5,         # bi-encoder: higher negative sampling
    warmup_ratio=0.1,
    save_steps=500,
    save_total_limit=2,
)
trainer.model.save_pretrained("./my_model")  # save final weights (not just checkpoint dir)
```

### Key training hyperparameters

| Parameter | Purpose | Bi-encoder guidance |
|-----------|---------|---------------------|
| `learning_rate` | Encoder LR | `1e-5` |
| `others_lr` | Non-encoder LR | `3e-5` |
| `negatives` | Negative sampling ratio | `1.5` (higher than default) |
| `negatives` | Negative sampling ratio | `1.5` |
| `max_steps` | Total steps | 2000+ for production |
| `per_device_train_batch_size` | Batch size | 8 (or 4 with limited memory) |
| `freeze_components` | Freeze encoder | `["text_encoder"]` for fast adaptation |

---

## 5. Models in This Module

| Model | Params | Use case |
|-------|--------|----------|
| `knowledgator/gliner-bi-edge-v2.0` | 60M | General NER, production (fast) |
| `knowledgator/gliner-pii-edge-v1.0` | — | PII/PHI/PCI detection |
