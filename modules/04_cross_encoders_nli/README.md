# Module 4: Cross-Encoders & NLI

> Reranking and zero-shot classification for production NLP systems

## Why This Matters

Bi-encoders (sentence-transformers) are fast but not always accurate. Cross-encoders examine both texts together for superior accuracy. NLI models enable zero-shot classification without training data. In production, you typically use bi-encoders for retrieval (fast) + cross-encoders for reranking (accurate).

## Key Concepts

### Bi-Encoder vs Cross-Encoder

**Bi-Encoder (sentence-transformers):**
```python
# Encode separately, compare with cosine similarity
query_emb = model.encode(query)
doc_emb = model.encode(doc)
score = cosine_similarity(query_emb, doc_emb)
# Fast: Can precompute doc embeddings
# Less accurate: Texts don't interact
```

**Cross-Encoder:**
```python
# Encode together, direct relevance score
score = cross_encoder.predict([(query, doc)])
# Accurate: Full attention between texts
# Slow: Must recompute for every pair
```

### Production Pipeline

```
User Query
    ↓
Bi-Encoder (retrieve top 100)  ← Fast retrieval
    ↓
Cross-Encoder (rerank to top 10)  ← Accurate reranking
    ↓
Final Results
```

### NLI (Natural Language Inference)

NLI models predict relationship between premise and hypothesis:
- **Entailment**: Hypothesis follows from premise
- **Contradiction**: Hypothesis contradicts premise
- **Neutral**: No clear relationship

**Zero-shot classification:**
```python
premise = "I love this movie, best film ever!"
hypothesis = "This text expresses positive sentiment"
# Model predicts: Entailment (high score)
```

## Common Models

| Model | Type | Use Case | Speed |
|-------|------|----------|-------|
| `ms-marco-MiniLM-L-12-v2` | Cross-Encoder | Passage reranking | Medium |
| `cross-encoder/nli-deberta-v3-base` | NLI | Zero-shot classification | Slow |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Cross-Encoder | Fast reranking | Fast |

## Documentation & Resources

- [Cross-Encoders Guide](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [NLI Zero-Shot](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [Reranking Tutorial](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)

## Self-Assessment Checklist

- [ ] I understand the difference between bi-encoders and cross-encoders
- [ ] I can implement a reranking pipeline
- [ ] I know when to use cross-encoders vs bi-encoders
- [ ] I can use NLI models for zero-shot classification
- [ ] I understand entailment/contradiction/neutral
- [ ] I can measure reranking quality (MRR, NDCG)

---

## Deep Dive Q&A (30 Questions)

### Architecture & Design (1-10)

#### Q1: What is the fundamental difference between bi-encoders and cross-encoders?

**Answer:**

**Bi-Encoder (Dual Encoder):**
- Encodes query and document **separately**
- Creates fixed-size embeddings for each
- Compares embeddings with cosine similarity or dot product
- Can precompute and cache document embeddings
- Fast at inference (vector similarity)

**Cross-Encoder:**
- Encodes query and document **together** as a single input
- Uses full cross-attention between all tokens
- Outputs direct relevance score (0-1)
- Cannot cache - must recompute for every pair
- Slow but more accurate

**Architecture comparison:**
```python
# Bi-Encoder
[CLS] query [SEP] → Embedding A
[CLS] document [SEP] → Embedding B
score = dot(A, B)

# Cross-Encoder
[CLS] query [SEP] document [SEP] → score
# Full attention between query and document tokens
```

**When to use:**
- Bi-encoder: First-stage retrieval (millions of candidates)
- Cross-encoder: Reranking (hundreds of candidates)

---

#### Q2: How does the retrieve-rerank pipeline work in production?

**Answer:**

Two-stage approach combining speed and accuracy:

**Stage 1: Retrieval (Bi-Encoder)**
```python
# Fast retrieval from millions of documents
query_emb = bi_encoder.encode(query)
# Search FAISS index
top_100_docs = faiss_index.search(query_emb, k=100)
# Takes ~50ms for 1M documents
```

**Stage 2: Reranking (Cross-Encoder)**
```python
# Accurate scoring of top candidates
pairs = [(query, doc) for doc in top_100_docs]
scores = cross_encoder.predict(pairs)
top_10 = sorted(zip(top_100_docs, scores),
                key=lambda x: x[1], reverse=True)[:10]
# Takes ~200ms for 100 pairs
```

**Why this works:**
- Bi-encoder filters 1M → 100 quickly (high recall)
- Cross-encoder refines 100 → 10 accurately (high precision)
- Total: ~250ms vs 50 seconds if using cross-encoder on all

**Production metrics:**
```python
# Before reranking (bi-encoder only)
MRR@10: 0.65
NDCG@10: 0.70

# After reranking (+ cross-encoder)
MRR@10: 0.82  # +26% improvement
NDCG@10: 0.85  # +21% improvement
```

---

#### Q3: What is NLI and how does it enable zero-shot classification?

**Answer:**

**Natural Language Inference (NLI):**

Task: Given premise and hypothesis, predict relationship:
- **Entailment**: Hypothesis logically follows from premise
- **Contradiction**: Hypothesis contradicts premise
- **Neutral**: No clear relationship

**Training data example (MNLI dataset):**
```python
{
  "premise": "A soccer game with multiple males playing.",
  "hypothesis": "Some men are playing a sport.",
  "label": "entailment"
}

{
  "premise": "An older and younger man smiling.",
  "hypothesis": "Two men are smiling and laughing.",
  "label": "neutral"  # laughing not mentioned
}
```

**Zero-shot classification:**

Convert classification to NLI by framing labels as hypotheses:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")

text = "I absolutely loved this movie! Best I've seen all year."

labels = ["positive", "negative", "neutral"]

# Framework converts to NLI:
# Premise: "I absolutely loved this movie! Best I've seen all year."
# Hypothesis: "This text expresses positive sentiment"
# Model predicts: Entailment (high)

result = classifier(text, labels)
# {'labels': ['positive', 'negative', 'neutral'],
#  'scores': [0.97, 0.02, 0.01]}
```

**How it works:**
1. Create hypothesis template: "This text is about {label}"
2. For each label, compute entailment probability
3. Return label with highest entailment score

**Advantages:**
- No training data needed for new categories
- Flexible - just change label list
- Works across many domains

---

#### Q4: Why can't you use cross-encoders for initial retrieval?

**Answer:**

**Computational infeasibility:**

```python
# Scenario: 1M documents, 1 query

# Cross-encoder approach
pairs = [(query, doc) for doc in all_docs]  # 1M pairs
scores = cross_encoder.predict(pairs)
# Time: 1M pairs × 50ms = 13.9 HOURS

# Bi-encoder + FAISS approach
query_emb = bi_encoder.encode(query)  # 10ms
top_k = faiss_index.search(query_emb, 100)  # 40ms
# Time: 50ms total
```

**Why cross-encoders are slow:**

1. **No caching**: Must process every (query, document) pair together
2. **Full forward pass**: Complete transformer computation for each pair
3. **No indexing**: Can't use vector search algorithms (FAISS, HNSW)
4. **Sequential processing**: Even with batch processing, scales linearly

**Memory requirements:**
```python
# Cross-encoder input
input_ids = tokenize(f"{query} [SEP] {document}")
# Length: len(query) + len(doc) + 3 special tokens
# For 1M docs: Need to fit 1M sequences in memory or batch

# Bi-encoder
query_emb = model.encode(query)  # Shape: (384,)
# FAISS index: 1M × 384 × 4 bytes = 1.5 GB
# Can memory-map, stays in RAM
```

**Solution: Two-stage pipeline:**
- Bi-encoder: Fast retrieval (1M → 100 candidates)
- Cross-encoder: Accurate reranking (100 → 10 results)

---

#### Q5: How do you measure reranking quality?

**Answer:**

**Mean Reciprocal Rank (MRR):**

Measures rank of first relevant result:

```python
def mrr(rankings):
    """
    rankings: List of lists, each inner list is doc IDs in ranked order
    relevant_doc_id: ID of correct document
    """
    reciprocal_ranks = []
    for ranking, relevant_id in zip(rankings, relevant_ids):
        try:
            rank = ranking.index(relevant_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)

# Example
rankings = [
    [5, 3, 1, 7],  # relevant is 3 (rank=2) → 1/2 = 0.5
    [2, 8, 6, 4],  # relevant is 2 (rank=1) → 1/1 = 1.0
    [9, 1, 3, 5],  # relevant is 5 (rank=4) → 1/4 = 0.25
]
# MRR = (0.5 + 1.0 + 0.25) / 3 = 0.583
```

**Normalized Discounted Cumulative Gain (NDCG):**

Accounts for multiple relevant documents and their positions:

```python
def dcg_at_k(relevances, k):
    """relevances: List of relevance scores (0 or 1)"""
    relevances = np.asarray(relevances)[:k]
    return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))

def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0

# Example: Retrieved docs with relevance [0, 1, 1, 0, 1]
ndcg_at_k([0, 1, 1, 0, 1], k=5)
# DCG = 0/log2(2) + 1/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)
#     = 0 + 0.631 + 0.5 + 0 + 0.387 = 1.518
# IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) + 0/log2(5) + 0/log2(6)
#      = 1 + 0.631 + 0.5 + 0 + 0 = 2.131
# NDCG = 1.518 / 2.131 = 0.712
```

**Reranking improvement:**
```python
# Before reranking (bi-encoder)
MRR@10: 0.65
NDCG@10: 0.70
Precision@10: 0.35

# After reranking (cross-encoder)
MRR@10: 0.82 (+26%)
NDCG@10: 0.85 (+21%)
Precision@10: 0.51 (+46%)
```

---

#### Q6: When should you use NLI models vs training a classifier?

**Answer:**

**Use NLI (Zero-Shot) when:**

✅ No training data available
```python
# New category detection with zero examples
categories = ["billing", "technical", "feature_request", "bug"]
classifier.predict(text, candidate_labels=categories)
```

✅ Labels change frequently
```python
# Support ticket routing - new teams added weekly
teams = ["payments", "auth", "api", "mobile", "web"]
# No retraining needed
```

✅ Many classes, few examples per class
```python
# 100 product categories, 5-10 tickets each
# Not enough data to train classifier
```

✅ Multi-label classification
```python
# Ticket can have multiple tags
tags = ["urgent", "customer_facing", "data_issue", "security"]
result = classifier(text, tags, multi_label=True)
# Can belong to multiple categories
```

**Use Trained Classifier when:**

✅ Fixed set of classes with abundant training data
```python
# Spam detection: 100K labeled emails
# Train custom BERT classifier
```

✅ Need maximum accuracy on specific domain
```python
# Medical diagnosis: Fine-tuned on domain data
# NLI models lack medical knowledge
```

✅ Strict latency requirements
```python
# NLI: ~200ms per classification (3 classes)
# Fine-tuned: ~20ms per classification
# 10x faster for high-throughput
```

✅ Class definitions are complex/nuanced
```python
# NLI struggles with:
# "Technical debt", "Code smell", "Architectural issue"
# Better to train on examples
```

**Hybrid approach:**
```python
# Use NLI to bootstrap, then collect data and fine-tune
# 1. Deploy NLI model initially
# 2. Collect user corrections/feedback
# 3. Fine-tune custom classifier on collected data
# 4. Replace NLI with fine-tuned model
```

---

#### Q7: What are the trade-offs of different cross-encoder sizes?

**Answer:**

| Model | Params | Speed | Accuracy | Use Case |
|-------|--------|-------|----------|----------|
| TinyBERT-L-2 | 14M | 5ms/pair | 85% | Real-time, high throughput |
| MiniLM-L-6 | 22M | 15ms/pair | 90% | Balanced |
| MiniLM-L-12 | 33M | 30ms/pair | 93% | High quality |
| DeBERTa-v3-base | 86M | 80ms/pair | 95% | Maximum accuracy |

**Detailed comparison:**

```python
# Scenario: Rerank 100 candidates

# TinyBERT-L-2
batch_size = 32
batches = 100 // 32 = 4
time_per_batch = 15ms
total = 4 × 15 = 60ms
accuracy = 85%

# MiniLM-L-12
batch_size = 16
batches = 100 // 16 = 7
time_per_batch = 40ms
total = 7 × 40 = 280ms
accuracy = 93%

# DeBERTa-v3-base
batch_size = 8
batches = 100 // 8 = 13
time_per_batch = 80ms
total = 13 × 80 = 1040ms
accuracy = 95%
```

**Production decision tree:**

```python
# <100ms latency requirement
→ Use TinyBERT-L-2

# 100-500ms acceptable, need quality
→ Use MiniLM-L-6 or MiniLM-L-12

# Offline processing, maximum quality
→ Use DeBERTa-v3-base

# Extremely high throughput (>1000 QPS)
→ Consider staying with bi-encoder only
```

**GPU vs CPU:**
```python
# MiniLM-L-12 on 100 pairs

# CPU (16 cores):
# Batch size: 16
# Time: 280ms

# GPU (T4):
# Batch size: 64
# Time: 45ms
# 6x speedup

# GPU (A100):
# Batch size: 128
# Time: 15ms
# 18x speedup
```

---

#### Q8: How do you handle long documents with cross-encoders?

**Answer:**

**Problem:**

Cross-encoders have token limits (typically 512 tokens):
```python
query = "How do I reset my password?"  # ~10 tokens
document = long_doc  # 2000 tokens
# query + document + special tokens = 2010 > 512
# Model truncates, loses information
```

**Solution 1: Passage-level scoring**

```python
def split_into_passages(doc, max_length=400):
    """Split document into overlapping passages"""
    passages = []
    sentences = doc.split('.')
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < max_length:
            current += sent + "."
        else:
            passages.append(current)
            current = sent + "."

    if current:
        passages.append(current)
    return passages

# Score each passage separately
doc_passages = split_into_passages(document)
pairs = [(query, passage) for passage in doc_passages]
scores = cross_encoder.predict(pairs)

# Aggregate scores
final_score = max(scores)  # or mean, or weighted
```

**Solution 2: MaxP (Maximum Passage)**

Most commonly used in research:
```python
# Score document by its best-matching passage
passages = split_document(doc, stride=200)
scores = [cross_encoder.predict([(query, p)])[0]
          for p in passages]
document_score = max(scores)
```

**Solution 3: Hierarchical approach**

```python
# Step 1: Retrieve passages with bi-encoder
query_emb = bi_encoder.encode(query)
top_passages = faiss_index.search(query_emb, k=50)

# Step 2: Rerank passages with cross-encoder
scores = cross_encoder.predict(
    [(query, p) for p in top_passages]
)

# Step 3: Group by document and aggregate
doc_scores = defaultdict(list)
for passage, score in zip(top_passages, scores):
    doc_scores[passage.doc_id].append(score)

final_scores = {doc_id: max(scores)
                for doc_id, scores in doc_scores.items()}
```

**Solution 4: Sliding window**

```python
def score_with_window(query, doc, window_size=400, stride=200):
    scores = []
    for i in range(0, len(doc), stride):
        chunk = doc[i:i+window_size]
        score = cross_encoder.predict([(query, chunk)])[0]
        scores.append(score)
    return max(scores)
```

**Best practices:**
- Use passage-level retrieval from the start
- Store passages with document metadata
- Aggregate with `max()` for best results
- Consider overlap between passages (stride)

---

#### Q9: What is the relationship between NLI and semantic similarity?

**Answer:**

**Key difference:**

**Semantic Similarity (bi-encoders):**
- Measures how similar texts are in meaning
- Symmetric: sim(A, B) = sim(B, A)
- Based on embedding distance

**NLI (cross-encoders):**
- Measures directional relationship (entailment)
- Asymmetric: entail(A, B) ≠ entail(B, A)
- Based on logical inference

**Examples showing asymmetry:**

```python
text_a = "I have a dog"
text_b = "I have a pet"

# Similarity (symmetric)
similarity = cosine_sim(embed(text_a), embed(text_b))
# → 0.75 (moderately similar)

# NLI (asymmetric)
nli_model.predict((text_a, text_b))
# → Entailment (dog → pet is true)

nli_model.predict((text_b, text_a))
# → Neutral (pet → dog not necessarily true)
```

**Another example:**
```python
premise = "All birds can fly"
hypothesis = "Penguins can fly"

# Similarity: High (both about birds/flying)
similarity_score = 0.82

# NLI: Entailment by premise logic
# (even though factually wrong!)
nli_score = 0.91 (entailment)
```

**When to use each:**

```python
# Use Similarity for:
# - Finding duplicate questions
question1 = "How to reset password?"
question2 = "Password reset procedure?"
# High similarity → duplicates

# Use NLI for:
# - Fact verification
claim = "The product costs $50"
evidence = "Pricing starts at $50 per month"
# Entailment → claim supported

# - Question answering
question = "What is the capital of France?"
passage = "Paris is the capital and largest city of France"
# Entailment → passage answers question

# - Text classification
text = "This movie was terrible"
hypothesis = "This text expresses negative sentiment"
# Entailment → negative classification
```

**Combining both:**
```python
# Hybrid retrieval-verification pipeline
# 1. Retrieve with similarity
candidates = retrieve_by_similarity(query, top_k=100)

# 2. Verify with NLI
verified = []
for candidate in candidates:
    entailment_score = nli_model.predict((candidate, query))
    if entailment_score > 0.7:
        verified.append(candidate)
```

---

#### Q10: How do you handle class imbalance in NLI zero-shot classification?

**Answer:**

**Problem:**

NLI models can be biased toward certain labels:

```python
text = "The meeting is scheduled for tomorrow"
labels = ["positive", "negative", "neutral", "scheduling"]

result = classifier(text, labels)
# {'scheduling': 0.89, 'neutral': 0.08, 'positive': 0.02, 'negative': 0.01}
# 'scheduling' dominates because it's most specific
```

**Solution 1: Hypothesis template engineering**

```python
# Poor templates (generic)
template = "This text is about {}"
# All labels treated equally generic

# Better templates (specific)
templates = {
    "positive": "This text expresses positive sentiment",
    "negative": "This text expresses negative sentiment",
    "neutral": "This text expresses neutral sentiment",
    "scheduling": "This text is about scheduling an event"
}

# Now all hypotheses are equally specific
```

**Solution 2: Multi-label with thresholds**

```python
# Don't force single label
result = classifier(text, labels, multi_label=True)

# Apply per-class thresholds
thresholds = {
    "positive": 0.5,
    "negative": 0.5,
    "neutral": 0.3,  # Lower threshold for neutral
    "scheduling": 0.7  # Higher threshold for specific class
}

predictions = [
    label for label, score in zip(result['labels'], result['scores'])
    if score > thresholds[label]
]
```

**Solution 3: Calibration with validation set**

```python
# Collect validation data
val_data = [
    ("I love this product!", ["positive"]),
    ("The system crashed again", ["negative", "technical"]),
    # ... more examples
]

# Measure per-class precision/recall
from sklearn.metrics import classification_report

# Tune thresholds to balance classes
def tune_thresholds(val_data, candidate_thresholds):
    best_thresholds = {}
    for label in labels:
        best_f1 = 0
        for t in candidate_thresholds:
            # Test threshold and measure F1
            f1 = compute_f1(val_data, label, threshold=t)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[label] = t
    return best_thresholds
```

**Solution 4: Hierarchical classification**

```python
# First classify into broad categories
level_1 = classifier(text, ["positive", "negative", "neutral"])
top_sentiment = level_1['labels'][0]

# Then classify into specific topics
if top_sentiment == "negative":
    level_2 = classifier(text, ["bug", "slow", "confusing", "expensive"])

# Prevents specific labels from dominating general ones
```

**Solution 5: Contrastive framing**

```python
# Instead of independent classification
# Frame as pairwise comparisons

def pairwise_classify(text, label_a, label_b):
    hypothesis_a = f"This text is {label_a}"
    hypothesis_b = f"This text is {label_b}"

    score_a = nli_model.predict((text, hypothesis_a))
    score_b = nli_model.predict((text, hypothesis_b))

    return label_a if score_a > score_b else label_b

# Tournament-style classification
# Compare pairs, winner advances
```

---

### Implementation & Coding (11-20)

#### Q11: Implement a production-ready reranking pipeline

**Answer:**

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple
import time

class RetrieveRerankPipeline:
    def __init__(
        self,
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_retrieval: int = 100,
        top_k_rerank: int = 10
    ):
        """Production reranking pipeline"""
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.index = None
        self.documents = None

    def index_documents(self, documents: List[str]):
        """Create FAISS index for documents"""
        print(f"Indexing {len(documents)} documents...")
        self.documents = documents

        # Generate embeddings
        embeddings = self.bi_encoder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64
        ).astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✓ Indexed {self.index.ntotal} documents")

    def search(
        self,
        query: str,
        return_scores: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search with retrieve + rerank pipeline

        Returns:
            List of (document, score) tuples
        """
        # Stage 1: Retrieve with bi-encoder
        t0 = time.time()
        query_emb = self.bi_encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')

        scores, indices = self.index.search(
            query_emb,
            self.top_k_retrieval
        )

        candidates = [self.documents[idx] for idx in indices[0]]
        retrieval_time = (time.time() - t0) * 1000

        # Stage 2: Rerank with cross-encoder
        t0 = time.time()
        pairs = [[query, doc] for doc in candidates]
        rerank_scores = self.cross_encoder.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False
        )

        rerank_time = (time.time() - t0) * 1000

        # Sort by rerank scores
        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k_rerank]

        print(f"Retrieval: {retrieval_time:.1f}ms | "
              f"Rerank: {rerank_time:.1f}ms")

        return ranked if return_scores else [doc for doc, _ in ranked]

    def batch_search(
        self,
        queries: List[str],
        batch_size: int = 8
    ) -> List[List[Tuple[str, float]]]:
        """Process multiple queries efficiently"""
        results = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_results = [self.search(q) for q in batch]
            results.extend(batch_results)

        return results


# Usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "How to reset your password: Go to settings, click forgot password",
        "Python tutorial for beginners: Learn basic syntax and data types",
        "Customer support contact: Email us at support@example.com",
        "Product pricing: $9.99 per month for standard plan",
        # ... more documents
    ]

    # Initialize pipeline
    pipeline = RetrieveRerankPipeline(
        top_k_retrieval=50,
        top_k_rerank=5
    )

    # Index documents
    pipeline.index_documents(documents)

    # Search
    results = pipeline.search("password reset help")

    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.3f}] {doc[:80]}...")
```

---

#### Q12: Implement zero-shot classification with custom templates

**Answer:**

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class ZeroShotClassifier:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Custom zero-shot classifier with template support"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        self.device = device

        # Label mapping: entailment=support for hypothesis
        self.label2id = self.model.config.label2id
        self.entailment_id = self.label2id.get("entailment",
                                                self.label2id.get("ENTAILMENT", 0))

    def predict(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template: str = "This text is about {}.",
        multi_label: bool = False,
        threshold: float = 0.5
    ) -> dict:
        """
        Zero-shot classification with custom templates

        Args:
            text: Input text to classify
            candidate_labels: List of possible labels
            hypothesis_template: Template for hypothesis (use {} for label)
            multi_label: Allow multiple labels
            threshold: Minimum score for multi-label

        Returns:
            Dict with 'labels' and 'scores'
        """
        # Create hypotheses from template
        hypotheses = [
            hypothesis_template.format(label)
            for label in candidate_labels
        ]

        # Compute entailment scores
        scores = []
        for hypothesis in hypotheses:
            # Tokenize premise + hypothesis
            inputs = self.tokenizer(
                text,
                hypothesis,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                entailment_score = probs[0, self.entailment_id].item()

            scores.append(entailment_score)

        # Normalize scores to sum to 1 (if single-label)
        if not multi_label:
            scores = np.array(scores)
            scores = scores / scores.sum()

            # Sort by score
            sorted_indices = np.argsort(scores)[::-1]

            return {
                'labels': [candidate_labels[i] for i in sorted_indices],
                'scores': [scores[i] for i in sorted_indices]
            }
        else:
            # Multi-label: return all above threshold
            results = [
                (label, score)
                for label, score in zip(candidate_labels, scores)
                if score >= threshold
            ]
            results.sort(key=lambda x: x[1], reverse=True)

            return {
                'labels': [label for label, _ in results],
                'scores': [score for _, score in results]
            }

    def predict_with_custom_hypotheses(
        self,
        text: str,
        label_hypotheses: dict
    ) -> dict:
        """
        Use completely custom hypotheses per label

        Args:
            text: Input text
            label_hypotheses: Dict mapping label -> custom hypothesis

        Example:
            label_hypotheses = {
                'positive': 'The author of this text is satisfied',
                'negative': 'The author of this text is dissatisfied',
                'neutral': 'The author expresses no strong opinion'
            }
        """
        scores = {}

        for label, hypothesis in label_hypotheses.items():
            inputs = self.tokenizer(
                text,
                hypothesis,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                scores[label] = probs[0, self.entailment_id].item()

        # Sort by score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'labels': [label for label, _ in sorted_items],
            'scores': [score for _, score in sorted_items]
        }


# Usage examples
if __name__ == "__main__":
    classifier = ZeroShotClassifier()

    # Example 1: Standard classification
    text = "I absolutely loved this product! Best purchase ever."
    labels = ["positive", "negative", "neutral"]

    result = classifier.predict(text, labels)
    print("Standard:", result)
    # {'labels': ['positive', 'negative', 'neutral'],
    #  'scores': [0.95, 0.03, 0.02]}

    # Example 2: Custom template
    text = "The new iPhone 15 has a better camera than iPhone 14"
    labels = ["comparison", "review", "technical_specs"]

    result = classifier.predict(
        text,
        labels,
        hypothesis_template="This is a {} article."
    )
    print("Custom template:", result)

    # Example 3: Multi-label with threshold
    text = "URGENT: Payment system is down, affecting all customers!"
    labels = ["urgent", "technical", "customer_facing", "billing"]

    result = classifier.predict(
        text,
        labels,
        multi_label=True,
        threshold=0.6
    )
    print("Multi-label:", result)
    # {'labels': ['urgent', 'technical', 'customer_facing', 'billing'],
    #  'scores': [0.91, 0.84, 0.78, 0.68]}

    # Example 4: Completely custom hypotheses
    text = "This movie was boring and predictable"

    custom_hypotheses = {
        'positive': 'The reviewer enjoyed and recommends this movie',
        'negative': 'The reviewer disliked this movie and does not recommend it',
        'mixed': 'The reviewer had both positive and negative opinions'
    }

    result = classifier.predict_with_custom_hypotheses(text, custom_hypotheses)
    print("Custom hypotheses:", result)
```

---

[Q13-Q20 would continue with implementation questions about batch processing, caching, error handling, monitoring, A/B testing, etc.]

### Debugging & Troubleshooting (21-25)

#### Q21: Cross-encoder returns unintuitive scores - how to debug?

**Answer:**

**Common issues:**

**Issue 1: Input format wrong**
```python
# WRONG - separate inputs
score = cross_encoder.predict(query, document)

# CORRECT - list of pairs
score = cross_encoder.predict([(query, document)])

# WRONG - reversed order for some models
score = cross_encoder.predict([(document, query)])

# CORRECT - query first (check model card!)
score = cross_encoder.predict([(query, document)])
```

**Issue 2: Input too long (truncated)**
```python
# Check if truncation happening
inputs = tokenizer(query, document, return_tensors="pt")
print(f"Input length: {inputs['input_ids'].shape[1]}")
# If > 512, truncation occurred

# Solution: split document
def score_long_doc(query, doc, max_length=400):
    passages = split_doc(doc, max_length)
    scores = cross_encoder.predict([(query, p) for p in passages])
    return max(scores)  # MaxP strategy
```

**Issue 3: Model trained on different task**
```python
# ms-marco models: trained on passage ranking
# Good for: "query" vs "passage" pairs
# Bad for: sentence similarity, paraphrase detection

# NLI models: trained on entailment
# Good for: classification, QA verification
# Bad for: passage ranking

# Check model card for intended use!
```

**Debugging script:**
```python
def debug_cross_encoder(model, query, document):
    """Comprehensive debugging"""
    print("="*50)
    print("DEBUGGING CROSS-ENCODER")
    print("="*50)

    # 1. Check inputs
    print(f"\nQuery ({len(query)} chars): {query[:100]}...")
    print(f"Document ({len(document)} chars): {document[:100]}...")

    # 2. Check tokenization
    inputs = model.tokenizer(query, document, return_tensors="pt")
    input_len = inputs['input_ids'].shape[1]
    print(f"\nTokenized length: {input_len}")
    if input_len >= 512:
        print("⚠️  WARNING: Input truncated!")

    # 3. Get raw score
    score = model.predict([(query, document)])[0]
    print(f"\nRaw score: {score:.4f}")

    # 4. Compare with reversed order
    score_reversed = model.predict([(document, query)])[0]
    print(f"Reversed score: {score_reversed:.4f}")
    if abs(score - score_reversed) > 0.1:
        print("⚠️  WARNING: Order matters! Check model documentation.")

    # 5. Test with known good/bad pairs
    good_pair = ("weather today", "The weather forecast shows sunny skies")
    bad_pair = ("weather today", "Python programming tutorial")

    good_score = model.predict([good_pair])[0]
    bad_score = model.predict([bad_pair])[0]

    print(f"\nSanity check:")
    print(f"  Good pair score: {good_score:.4f}")
    print(f"  Bad pair score: {bad_score:.4f}")

    if good_score < bad_score:
        print("⚠️  ERROR: Good pair scored lower than bad pair!")
        print("   Model may be wrong for this use case.")

    return {
        'score': score,
        'truncated': input_len >= 512,
        'order_sensitive': abs(score - score_reversed) > 0.1,
        'sanity_check_passed': good_score > bad_score
    }
```

---

[Q22-Q25 would continue with debugging questions]

### Trade-offs & Decisions (26-30)

#### Q26: Should you use cross-encoder or fine-tune bi-encoder?

**Answer:**

**Cross-Encoder (Reranking):**

**Pros:**
- Higher accuracy out-of-the-box
- No training data needed
- Works for rare/tail queries
- Faster to deploy

**Cons:**
- Slow (can't precompute)
- Need bi-encoder anyway for retrieval
- Two models to maintain

**When to use:**
- Limited training data (<1K examples)
- Need to deploy quickly
- Tail query distribution
- Budget for 2-stage pipeline

**Fine-Tuned Bi-Encoder:**

**Pros:**
- Faster (single-stage possible)
- Single model to maintain
- Can optimize for specific domain
- Better on common queries

**Cons:**
- Need training data (>5K pairs)
- Lower ceiling on accuracy
- May overfit to training distribution

**When to use:**
- Have abundant training data
- Queries are predictable
- Need single-stage simplicity
- Latency critical (<50ms)

**Hybrid approach (recommended):**
```python
# Combine both!
# 1. Fine-tune bi-encoder on your data
#    (better retrieval than generic model)
# 2. Use cross-encoder for reranking
#    (fixes bi-encoder mistakes)

# Best of both worlds
```

**Cost-benefit analysis:**

| Approach | Accuracy | Latency | Development | Maintenance |
|----------|----------|---------|-------------|-------------|
| Generic bi-encoder | 75% | 50ms | 1 day | Low |
| + Cross-encoder | 88% | 250ms | 3 days | Medium |
| Fine-tuned bi-encoder | 82% | 50ms | 2 weeks | Medium |
| Fine-tuned + Cross | 92% | 250ms | 3 weeks | High |

---

[Q27-Q30 would continue with trade-off questions about deployment, scaling, costs, etc.]

---

## Additional Resources

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [Cross-Encoder Documentation](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
