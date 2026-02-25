# Module 16: GLiClass — 30 Deep Q&A

---

## Architecture & Design (10 Questions)

**Q1. How does GLiClass differ from a traditional NLI-based zero-shot classifier?**

A traditional NLI classifier (e.g., `facebook/bart-large-mnli`) runs a separate forward pass for each label, making it O(N) in the number of labels. GLiClass concatenates the text with all candidate labels and processes them jointly in a single forward pass. This makes inference O(1) regardless of label count — dramatically faster when classifying into many categories.

---

**Q2. How does GLiClass handle multi-label vs single-label classification internally?**

The model has a shared encoder but different output heads:
- **Multi-label**: A sigmoid head is applied to each label independently. Labels above `threshold` are returned.
- **Single-label**: A softmax head normalizes scores across all labels. Only the highest-scoring label is returned.
The `classification_type` parameter to `ZeroShotClassificationPipeline` controls which head is used.

---

**Q3. What is the role of `add_prefix_space=True` in the tokenizer?**

Some tokenizers (BPE-based like GPT-2/RoBERTa-style) treat the first token of a word differently depending on whether it follows a space. Without `add_prefix_space=True`, label tokens at the start of the label sequence may be tokenized differently than when they appear mid-sentence, leading to suboptimal representations. GLiClass requires this flag for correct label tokenization.

---

**Q4. What is the training data format for GLiClass fine-tuning?**

Each training example must have three fields:
```json
{
  "text": "Apple unveiled a new MacBook Pro with M4 chip...",
  "all_labels": ["technology", "finance", "sports", "science", "politics"],
  "true_labels": ["technology"]
}
```
- `text`: the document to classify
- `all_labels`: all candidate labels the model should score (the full label set)
- `true_labels`: the correct labels (one for single-label, multiple for multi-label)

The model is trained to score correct labels higher than incorrect ones within each example.

---

**Q5. Why use `problem_type='multi_label_classification'` even for single-label fine-tuning?**

When using `GLiClassDataset` with `problem_type='single_label_classification'`, the labels tensor is a 0-dimensional scalar (the class index). The built-in `DataCollatorWithPadding` only handles 1D and 2D tensors, so it silently skips these batches — resulting in `loss=0` and no gradient updates.

With `problem_type='multi_label_classification'`, labels are stored as a one-hot vector of shape `[num_labels]`, which the collator handles correctly. Even with one true label, this works fine: the sigmoid outputs for all labels are compared against the one-hot ground truth.

---

**Q6. What is `threshold` and how should you tune it for multi-label classification?**

`threshold` is the minimum confidence score for a label to be included in multi-label output. Too high → labels missed (low recall). Too low → false positives (low precision). Strategy:
- Start at `0.5` (default)
- If too few labels are returned, lower to `0.3–0.4`
- If too many irrelevant labels appear, raise to `0.6–0.7`
- Use held-out validation data to find the optimal threshold for your specific label set and domain

---

**Q7. How does label wording affect GLiClass predictions?**

Label wording matters significantly. GLiClass was trained on natural-language label descriptions, so:
- `"positive sentiment"` often outperforms `"positive"` because it's more specific
- `"artificial intelligence and machine learning"` may outperform `"AI"`
- Ambiguous or very short labels (1-2 chars) tend to underperform
- Descriptive, domain-specific labels tend to work better than abstract category names

---

**Q8. What is the role of `others_lr` in GLiClass `TrainingArguments`?**

GLiClass has two parameter groups with separate learning rates:
- **Encoder** (`learning_rate`): The transformer backbone (DeBERTa, BERT, etc.) — lower LR (e.g., `1e-5` to `5e-5`) to avoid catastrophic forgetting of pre-trained knowledge
- **Others** (`others_lr`): The classification head, scorer, and adapter layers — higher LR (e.g., `1e-4`) since these are randomly initialized and need to learn from scratch

Using a single LR for both tends to either under-train the head or over-train/corrupt the encoder. The two-group learning rate is a form of discriminative fine-tuning that preserves pre-trained representations while adapting the task-specific components.

---

**Q9. What are the limitations of GLiClass for text classification?**

1. **Label count**: Very large label sets (100+) may cause truncation or degraded performance since all labels are concatenated into the input
2. **Text length**: Long texts may be truncated — consider chunking
3. **Domain specificity**: Zero-shot performance drops on highly specialized domains (medical, legal) not well represented in training data
4. **Label ambiguity**: If labels are semantically overlapping (e.g., "finance" vs "economics"), single-label mode may be inconsistent
5. **Threshold sensitivity**: Optimal threshold varies by domain and requires calibration

---

**Q10. How does fine-tuning GLiClass compare to training a classification head from scratch?**

| | Fine-tune GLiClass | New classifier head |
|---|---|---|
| Requires labeled data? | Yes (but fewer examples needed) | Yes (more needed) |
| Zero-shot capability? | Kept (transfers to unseen labels) | Lost (fixed output size) |
| Generalization | High (open-vocabulary labels) | Lower (closed label set) |
| Inference speed | Slower (processes all labels) | Faster (linear head) |
| Data efficiency | High (few-shot fine-tuning works) | Lower |

Fine-tuning GLiClass is preferred when: (1) the label set may change, (2) you have < 1000 labeled examples, (3) you need to classify into new labels at inference time. A fixed classifier head is better for high-throughput inference with a stable, known label set.

---

## Implementation & Coding (10 Questions)

**Q11. How do you load GLiClass and create both pipeline types?**

```python
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch

model = GLiClassModel.from_pretrained("knowledgator/gliclass-edge-v3.0")
tokenizer = AutoTokenizer.from_pretrained(
    "knowledgator/gliclass-edge-v3.0", add_prefix_space=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline_ml = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device=device
)
pipeline_sl = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='single-label', device=device
)
```

---

**Q12. What is the exact output format of the pipeline?**

```python
results = pipeline(texts_or_text, labels, threshold=0.5, batch_size=8)
# Always returns: list[list[dict]]
# Outer list: one element per input text
# Inner list: scored labels (filtered by threshold for multi-label)
# Each dict: {"label": str, "score": float}

# Examples:
pipeline("text", ["a", "b"])        # → [[{"label": "a", "score": 0.9}]]
pipeline(["t1", "t2"], ["a", "b"]) # → [[...], [...]]
```

---

**Q13. How do you implement `analyze_sentiment` for batch processing?**

```python
def analyze_sentiment(pipeline, texts, labels):
    results = pipeline(texts, labels, batch_size=8)
    return [
        {"text": text, "label": res[0]['label'], "score": res[0]['score']}
        for text, res in zip(texts, results)
    ]
# Note: res[0] is correct for single-label mode (best label is always first)
```

---

**Q14. How do you implement a `rerank` function using GLiClass?**

```python
def rerank(pipeline, query, passages, top_k=None):
    results = pipeline(query, passages, threshold=0.0)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    if top_k is not None:
        sorted_results = sorted_results[:top_k]
    return [
        {"text": r['label'], "score": r['score'], "rank": i + 1}
        for i, r in enumerate(sorted_results)
    ]
# Key: threshold=0.0 to get scores for ALL passages, not just confident ones
# Key: r['label'] contains the passage text (labels = passages in NLI mode)
```

---

**Q15. How do you compute MRR@K from reranker output?**

```python
def compute_mrr(pipeline, queries, passages, top_k=5):
    passage_texts = [p["text"] for p in passages]
    passage_id_map = {p["text"]: p["id"] for p in passages}
    reciprocal_ranks = []
    for q in queries:
        relevant_ids = set(q["relevant_passage_ids"])
        ranked = rerank(pipeline, q["query"], passage_texts, top_k=top_k)
        rr = 0.0
        for item in ranked:
            if passage_id_map.get(item["text"]) in relevant_ids:
                rr = 1.0 / item["rank"]
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

---

**Q16. How do you implement `classify_topics` for multi-label batch classification?**

```python
def classify_topics(pipeline, texts, labels, threshold=0.4):
    all_results = pipeline(texts, labels, threshold=threshold, batch_size=4)
    return [[r['label'] for r in res] for res in all_results]
# Each text returns a list of label names (strings), not dicts
# Empty list = no labels passed threshold (valid for multi-label)
```

---

**Q17. What is the correct way to compute single-label accuracy?**

```python
results = pipeline_sl(texts, labels, batch_size=4)
correct = sum(
    1 for article, res in zip(articles, results)
    if res[0]['label'] in article['expected_topics']
)
accuracy = correct / len(articles)
# res[0] is the top (and only) prediction in single-label mode
# Use `in expected_topics` when ground truth allows multiple valid answers
```

---

**Q18. How would you handle a very large label set (500+ classes) with GLiClass?**

Chunk the labels into batches and aggregate:
```python
def classify_large_labelset(pipeline, text, all_labels, chunk_size=50, threshold=0.4):
    all_scores = {}
    for i in range(0, len(all_labels), chunk_size):
        chunk = all_labels[i:i + chunk_size]
        res = pipeline(text, chunk, threshold=0.0)[0]
        for r in res:
            all_scores[r['label']] = r['score']
    return [label for label, score in all_scores.items() if score >= threshold]
```

---

**Q19. How would you build an aspect-based sentiment analyzer with GLiClass?**

```python
aspects = ["battery life", "screen quality", "camera", "price", "build quality"]
text = "The camera is excellent but the battery life is terrible."

# For each aspect, classify sentiment separately
for aspect in aspects:
    labels = [f"positive {aspect}", f"negative {aspect}", f"neutral {aspect}"]
    result = pipeline_sl(text, labels)[0][0]
    print(f"{aspect}: {result['label']} ({result['score']:.2f})")
# Output: "camera: positive camera (0.91)"
#         "battery life: negative battery life (0.87)"
```

---

**Q20. How do you implement `compute_metrics` for GLiClass training evaluation?**

For `multi_label_classification` mode, predictions and labels are flattened before computing metrics:

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds, labels = p
    # preds shape: [batch, num_labels]; labels shape: [batch, num_labels]
    preds_flat = (preds.reshape(-1) > 0.5).astype(int)   # threshold at 0.5
    labels_flat = (labels.reshape(-1) > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels_flat, preds_flat)),
        "f1": float(f1_score(labels_flat, preds_flat, average='weighted', zero_division=0)),
    }
```

For single-label evaluation via the pipeline (after training), compute accuracy differently:
```python
results = pipeline(eval_texts, LABELS, batch_size=4)
preds = [r[0]['label'] for r in results]
accuracy = sum(p == t for p, t in zip(preds, true_labels)) / len(true_labels)
```

---

## Debugging & Troubleshooting (5 Questions)

**Q21. Why does `pipeline(text, labels)` return an empty list for all texts?**

Most likely causes:
1. **Threshold too high**: All scores below threshold → empty inner list. Fix: lower `threshold` or use `threshold=0.0` to debug and inspect raw scores.
2. **Wrong classification type**: `multi-label` returns empty if nothing passes threshold; `single-label` always returns at least one item. Check `classification_type`.
3. **Labels too generic/short**: One-character labels or empty strings may score near zero. Use descriptive label phrases.

---

**Q22. Why does the pipeline output have wrong keys (no 'label' or 'score')?**

This can happen if:
- You're accessing `results` instead of `results[0]` — the outer list is per-text, not per-label
- You upgraded `gliclass` and the output format changed — check `gliclass.__version__`
- You're iterating over the string `results[0]` (if accidentally passing a single character)

Debug with: `print(type(results), type(results[0]), results[0][:2])`

---

**Q23. Why is reranking accuracy poor even though multi-label classification works?**

Common issues in NLI/reranker mode:
1. **Using `classification_type='single-label'`**: NLI mode should use `multi-label` with `threshold=0.0` so all passages are scored.
2. **Missing `threshold=0.0`**: Default threshold filters out low-confidence passages, losing ranking signal.
3. **Wrong input order**: Text = query, Labels = passages. Reversed order gives poor results.
4. **Very short passages**: GLiClass struggles with 1–2 word passages; works best with full sentences.

---

**Q24. Why does sentiment analysis get less than 70% accuracy?**

Troubleshooting steps:
1. Inspect misclassified reviews: are they genuinely ambiguous or clearly wrong?
2. Try more descriptive labels: `"very positive"` / `"very negative"` instead of `"positive"` / `"negative"`
3. Check for neutral reviews misclassified: neutral is often confused with positive
4. Try `single-label` with `classification_type='single-label'` if using multi-label
5. Inspect the score distribution: if scores cluster near 0.5, the model is uncertain

---

**Q25. Why does the model fail with `CUDA out of memory`?**

Reduce batch size or label count:
```python
# Instead of batch_size=16 with 50 labels:
pipeline(texts, labels, batch_size=4, threshold=0.4)  # Smaller batches
# Or use CPU for edge model:
device = torch.device("cpu")  # Edge models are designed for CPU
```
The edge model (`gliclass-edge-v3.0`) is designed to run on CPU. Large batch sizes × many labels × long texts cause OOM on GPU with small VRAM.

---

## Trade-offs & Decisions (5 Questions)

**Q26. When should you use GLiClass vs a fine-tuned classifier?**

| Scenario | Recommendation |
|----------|----------------|
| Label set changes frequently | GLiClass (zero-shot) |
| < 1000 labeled training examples | GLiClass |
| High accuracy required, fixed labels | Fine-tuned classifier |
| Latency-critical, large label set | Fine-tuned classifier |
| Prototyping / quick iteration | GLiClass |
| Production at scale | Fine-tuned classifier |

GLiClass is ideal for exploration, quickly changing requirements, and low-data scenarios. Fine-tuned models win when you have labeled data and stable label sets.

---

**Q27. When should you use multi-label vs single-label mode?**

- **Multi-label**: When categories are not mutually exclusive (a news article can be both "technology" and "business"); when you need to return all applicable categories; when the question is "what topics apply?" rather than "what is the best topic?"
- **Single-label**: When only one category applies (positive/negative/neutral sentiment); when you need a definitive single answer; when categories are mutually exclusive by definition.

---

**Q28. How do you use `AugmentationConfig` to improve fine-tuning?**

`AugmentationConfig` enables data augmentation during training to improve robustness:

```python
from gliclass.data_processing import AugmentationConfig

augment_config = AugmentationConfig(
    enabled=True,
    random_label_removal_prob=0.05,   # randomly drop some labels from all_labels
    random_label_addition_prob=0.05,  # randomly add negative labels
    random_text_addition_prob=0.05,   # randomly append extra text
)
# Disable augmentation for eval:
eval_augment = AugmentationConfig(enabled=False)
```

Augmentation is most useful when: training data is small (< 500 examples), labels are few (< 20), or you want the model to be robust to different label sets. Keep probabilities low (0.05–0.1) to avoid corrupting the training signal.

---

**Q29. What `batch_size` should you use for the pipeline?**

Trade-off between throughput and memory:
- `batch_size=1`: Slowest, lowest memory — use for very long texts or tight memory
- `batch_size=4`: Good default for CPU inference with edge model
- `batch_size=8–16`: Good for GPU with moderate text lengths
- `batch_size=32+`: Only for short texts on high-memory GPU

For the edge model on CPU, `batch_size=4–8` is typically optimal. Larger batches don't always help because the edge model is memory-bandwidth-limited, not compute-limited.

---

**Q30. How would you evaluate whether GLiClass is good enough for your use case?**

Systematic evaluation approach:
1. **Create a gold-standard test set**: 50–200 examples with human-verified labels
2. **Measure relevant metrics**: Accuracy (single-label), F1/precision/recall (multi-label), MRR/NDCG (reranker)
3. **Compare baselines**: Zero-shot embedding similarity, keyword matching, fine-tuned classifier
4. **Analyze failure modes**: Which label pairs are most confused? Are errors semantic or random?
5. **Calibration check**: Are confidence scores well-calibrated? (high score → high precision)
6. **Latency profiling**: Measure p50/p99 latency under expected load

A simple rule of thumb: if GLiClass achieves ≥ 80% of fine-tuned model accuracy with zero training data, it's likely worth using for its flexibility benefits.
