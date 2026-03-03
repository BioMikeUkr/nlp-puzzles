# Module 4: Cross-Encoders & NLI - Deep Dive Q&A

## Architecture & Design (1-10)

### Q1: What is the fundamental difference between bi-encoders and cross-encoders?

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

### Q2: How does the retrieve-rerank pipeline work in production?

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

### Q3: What is NLI and how does it enable zero-shot classification?

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

### Q4: Why can't you use cross-encoders for initial retrieval?

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

### Q5: How do you measure reranking quality?

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

### Q6: When should you use NLI models vs training a classifier?

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

### Q7: What are the trade-offs of different cross-encoder sizes?

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

### Q8: How do you handle long documents with cross-encoders?

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

### Q9: What is the relationship between NLI and semantic similarity?

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

### Q10: How do you handle class imbalance in NLI zero-shot classification?

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

## Implementation & Coding (11-20)

### Q11: Implement a production-ready reranking pipeline

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

### Q12: Implement zero-shot classification with custom templates

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

### Q13: How do you implement batch processing for cross-encoders?

**Answer:**

```python
import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import numpy as np

class BatchCrossEncoder:
    """Cross-encoder with optimized batch processing"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str = None
    ):
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def predict_pairs(
        self,
        pairs: List[Tuple[str, str]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Predict scores for list of (query, document) pairs

        Args:
            pairs: List of (text1, text2) tuples
            show_progress: Show progress bar

        Returns:
            Array of scores
        """
        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress
        )

    def rank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, str, float]]:
        """
        Rank documents for a single query

        Returns:
            List of (index, document, score) tuples sorted by score
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.predict_pairs(pairs, show_progress=False)

        # Combine and sort
        results = list(zip(range(len(documents)), documents, scores))
        results.sort(key=lambda x: x[2], reverse=True)

        if top_k:
            return results[:top_k]
        return results

    def rank_multi_query(
        self,
        queries: List[str],
        documents: List[str],
        top_k: int = 10
    ) -> dict:
        """
        Rank documents for multiple queries efficiently

        Returns:
            Dict mapping query_idx -> List[(doc_idx, score)]
        """
        # Create all pairs
        all_pairs = []
        query_map = []  # Track which query each pair belongs to

        for q_idx, query in enumerate(queries):
            for doc in documents:
                all_pairs.append([query, doc])
                query_map.append(q_idx)

        # Batch predict all pairs at once
        all_scores = self.predict_pairs(all_pairs)

        # Group results by query
        results = {i: [] for i in range(len(queries))}

        doc_idx = 0
        for pair_idx, (score, q_idx) in enumerate(zip(all_scores, query_map)):
            # Calculate which document this is
            d_idx = pair_idx % len(documents)
            results[q_idx].append((d_idx, score))

            # Move to next doc
            if (pair_idx + 1) % len(documents) == 0:
                doc_idx = 0
            else:
                doc_idx += 1

        # Sort each query's results and take top-k
        for q_idx in results:
            results[q_idx].sort(key=lambda x: x[1], reverse=True)
            results[q_idx] = results[q_idx][:top_k]

        return results


# Usage
if __name__ == "__main__":
    encoder = BatchCrossEncoder(batch_size=32)

    # Example 1: Rank documents for single query
    query = "How to reset password?"
    documents = [
        "Password reset instructions: Go to settings...",
        "Product pricing information...",
        "Contact customer support...",
        "Login troubleshooting guide..."
    ]

    results = encoder.rank_documents(query, documents, top_k=3)

    print("Top 3 results:")
    for idx, doc, score in results:
        print(f"  {score:.3f}: {doc[:50]}...")

    # Example 2: Multiple queries
    queries = [
        "password reset",
        "pricing info",
        "contact support"
    ]

    multi_results = encoder.rank_multi_query(queries, documents, top_k=2)

    for q_idx, query in enumerate(queries):
        print(f"\nQuery: {query}")
        for doc_idx, score in multi_results[q_idx]:
            print(f"  {score:.3f}: {documents[doc_idx][:40]}...")
```

---

### Q14: How do you implement caching for cross-encoder predictions?

**Answer:**

```python
import hashlib
import json
from functools import lru_cache
from typing import List, Tuple
import sqlite3
from sentence_transformers import CrossEncoder

class CachedCrossEncoder:
    """Cross-encoder with persistent caching"""

    def __init__(
        self,
        model_name: str,
        cache_db: str = "cross_encoder_cache.db",
        memory_cache_size: int = 10000
    ):
        self.model = CrossEncoder(model_name)
        self.cache_db = cache_db
        self.memory_cache_size = memory_cache_size

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Create cache database"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                pair_hash TEXT PRIMARY KEY,
                text1 TEXT,
                text2 TEXT,
                score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _get_pair_hash(self, text1: str, text2: str) -> str:
        """Generate hash for text pair"""
        pair_str = f"{text1}|||{text2}"
        return hashlib.md5(pair_str.encode()).hexdigest()

    @lru_cache(maxsize=10000)
    def _predict_cached_memory(self, pair_hash: str, text1: str, text2: str) -> float:
        """Memory cache layer"""
        return self._predict_cached_db(pair_hash, text1, text2)

    def _predict_cached_db(self, pair_hash: str, text1: str, text2: str) -> float:
        """Database cache layer"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Check cache
        cursor.execute(
            "SELECT score FROM cache WHERE pair_hash = ?",
            (pair_hash,)
        )
        result = cursor.fetchone()

        if result:
            conn.close()
            return result[0]

        # Not in cache, compute
        score = self.model.predict([[text1, text2]])[0]

        # Store in cache
        cursor.execute(
            "INSERT INTO cache (pair_hash, text1, text2, score) VALUES (?, ?, ?, ?)",
            (pair_hash, text1, text2, float(score))
        )
        conn.commit()
        conn.close()

        return float(score)

    def predict(
        self,
        pairs: List[Tuple[str, str]],
        use_cache: bool = True
    ) -> List[float]:
        """
        Predict with caching

        Args:
            pairs: List of (text1, text2) tuples
            use_cache: Whether to use cache
        """
        if not use_cache:
            return self.model.predict(pairs).tolist()

        scores = []
        uncached_pairs = []
        uncached_indices = []

        # Check cache for each pair
        for i, (text1, text2) in enumerate(pairs):
            pair_hash = self._get_pair_hash(text1, text2)

            try:
                # Try memory cache first
                score = self._predict_cached_memory(pair_hash, text1, text2)
                scores.append((i, score))
            except:
                # Need to compute
                uncached_pairs.append([text1, text2])
                uncached_indices.append(i)

        # Batch compute uncached pairs
        if uncached_pairs:
            new_scores = self.model.predict(uncached_pairs)

            # Add to results and cache
            for idx, pair, score in zip(uncached_indices, uncached_pairs, new_scores):
                scores.append((idx, float(score)))

                # Cache the result
                pair_hash = self._get_pair_hash(pair[0], pair[1])
                self._predict_cached_memory(pair_hash, pair[0], pair[1])

        # Sort by original index and return scores
        scores.sort(key=lambda x: x[0])
        return [score for _, score in scores]

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cache")
        total_entries = cursor.fetchone()[0]

        # Get memory cache info
        cache_info = self._predict_cached_memory.cache_info()

        conn.close()

        return {
            "db_entries": total_entries,
            "memory_hits": cache_info.hits,
            "memory_misses": cache_info.misses,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses)
                       if (cache_info.hits + cache_info.misses) > 0 else 0
        }

    def clear_cache(self, older_than_days: int = None):
        """Clear cache (optionally only old entries)"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        if older_than_days:
            cursor.execute(
                "DELETE FROM cache WHERE timestamp < datetime('now', ? || ' days')",
                (f"-{older_than_days}",)
            )
        else:
            cursor.execute("DELETE FROM cache")

        conn.commit()
        deleted = cursor.rowcount
        conn.close()

        # Clear memory cache
        self._predict_cached_memory.cache_clear()

        return deleted


# Usage
if __name__ == "__main__":
    encoder = CachedCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    query = "password reset"
    docs = [
        "How to reset your password",
        "Pricing information",
        "Contact support"
    ]

    # First call: compute all
    pairs = [(query, doc) for doc in docs]
    scores1 = encoder.predict(pairs)
    print("First call scores:", scores1)

    # Second call: cached (instant)
    scores2 = encoder.predict(pairs)
    print("Second call scores:", scores2)

    # Stats
    stats = encoder.get_cache_stats()
    print(f"Cache stats: {stats}")
```

---

### Q15: How do you handle model versioning and A/B testing for cross-encoders?

**Answer:**

```python
import random
from typing import Dict, List
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import json
from datetime import datetime

@dataclass
class ModelVersion:
    """Model version metadata"""
    name: str
    model_path: str
    traffic_percentage: float
    enabled: bool = True

class ABTestingCrossEncoder:
    """Cross-encoder with A/B testing support"""

    def __init__(self, config_path: str = "model_config.json"):
        self.models: Dict[str, CrossEncoder] = {}
        self.config_path = config_path
        self.metrics = {}

        # Load configuration
        self.load_config()

    def load_config(self):
        """Load model versions from config"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Default config
            config = {
                "versions": [
                    {
                        "name": "baseline",
                        "model_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        "traffic_percentage": 100.0,
                        "enabled": True
                    }
                ]
            }

        # Load models
        for version_config in config["versions"]:
            if version_config["enabled"]:
                version = ModelVersion(**version_config)
                self.models[version.name] = CrossEncoder(version.model_path)
                self.metrics[version.name] = {
                    "requests": 0,
                    "total_score": 0.0,
                    "latency_ms": []
                }

    def select_model(self) -> str:
        """Select model based on traffic distribution"""
        # Get enabled models with traffic
        enabled = [
            (name, model) for name, model in self.models.items()
        ]

        if len(enabled) == 1:
            return enabled[0][0]

        # Weighted random selection
        rand = random.random() * 100
        cumulative = 0

        for name in self.models.keys():
            # Get traffic percentage from config
            cumulative += self._get_traffic_percentage(name)
            if rand <= cumulative:
                return name

        # Fallback to first model
        return list(self.models.keys())[0]

    def _get_traffic_percentage(self, model_name: str) -> float:
        """Get traffic percentage from config"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        for version in config["versions"]:
            if version["name"] == model_name:
                return version["traffic_percentage"]
        return 0.0

    def predict(
        self,
        pairs: List[List[str]],
        model_name: str = None
    ) -> Dict:
        """
        Predict with model selection and metric tracking

        Args:
            pairs: List of [text1, text2] pairs
            model_name: Specific model to use (None = auto-select)

        Returns:
            {
                'scores': List[float],
                'model_used': str,
                'latency_ms': float
            }
        """
        import time

        # Select model
        if model_name is None:
            model_name = self.select_model()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        # Predict with timing
        start = time.time()
        scores = model.predict(pairs, show_progress_bar=False)
        latency_ms = (time.time() - start) * 1000

        # Track metrics
        self.metrics[model_name]["requests"] += 1
        self.metrics[model_name]["total_score"] += float(scores.mean())
        self.metrics[model_name]["latency_ms"].append(latency_ms)

        return {
            'scores': scores.tolist(),
            'model_used': model_name,
            'latency_ms': latency_ms
        }

    def get_metrics(self) -> Dict:
        """Get performance metrics for all models"""
        import numpy as np

        result = {}
        for name, metrics in self.metrics.items():
            if metrics["requests"] > 0:
                result[name] = {
                    "requests": metrics["requests"],
                    "avg_score": metrics["total_score"] / metrics["requests"],
                    "avg_latency_ms": np.mean(metrics["latency_ms"]),
                    "p95_latency_ms": np.percentile(metrics["latency_ms"], 95)
                                      if len(metrics["latency_ms"]) > 0 else 0,
                    "p99_latency_ms": np.percentile(metrics["latency_ms"], 99)
                                      if len(metrics["latency_ms"]) > 0 else 0
                }

        return result

    def update_traffic(self, traffic_distribution: Dict[str, float]):
        """
        Update traffic distribution

        Args:
            traffic_distribution: Dict mapping model_name -> percentage
            Example: {"baseline": 50, "new_model": 50}
        """
        # Validate percentages sum to 100
        total = sum(traffic_distribution.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100, got {total}")

        # Update config
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        for version in config["versions"]:
            if version["name"] in traffic_distribution:
                version["traffic_percentage"] = traffic_distribution[version["name"]]

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Updated traffic distribution: {traffic_distribution}")


# Usage
if __name__ == "__main__":
    # Initialize with default config
    ab_encoder = ABTestingCrossEncoder()

    # Example queries
    pairs = [
        ["password reset", "How to reset your password"],
        ["pricing info", "Product pricing details"]
    ]

    # Make predictions (auto-selects model based on traffic)
    for _ in range(100):
        result = ab_encoder.predict(pairs)
        # In production, track user engagement here

    # View metrics
    metrics = ab_encoder.get_metrics()
    print("Model Performance:")
    for model_name, stats in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Requests: {stats['requests']}")
        print(f"  Avg Score: {stats['avg_score']:.3f}")
        print(f"  Avg Latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"  P95 Latency: {stats['p95_latency_ms']:.1f}ms")

    # Update traffic (e.g., after validation)
    # ab_encoder.update_traffic({"baseline": 50, "new_model": 50})
```

---

[Continue with Q16-Q20 for implementation questions about monitoring, error handling, production deployment, etc.]

### Q16: How do you monitor cross-encoder performance in production?

**Answer:**

```python
import time
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from sentence_transformers import CrossEncoder

@dataclass
class PredictionMetrics:
    """Metrics for a single prediction"""
    timestamp: str
    query: str
    num_candidates: int
    latency_ms: float
    scores: List[float]
    top_score: float
    score_variance: float
    model_version: str

class MonitoredCrossEncoder:
    """Cross-encoder with comprehensive monitoring"""

    def __init__(
        self,
        model_name: str,
        log_file: str = "cross_encoder_metrics.jsonl",
        alert_threshold_ms: float = 500.0
    ):
        self.model = CrossEncoder(model_name)
        self.model_version = model_name
        self.log_file = log_file
        self.alert_threshold_ms = alert_threshold_ms

        # In-memory metrics
        self.metrics_buffer = []
        self.buffer_size = 1000

    def predict_with_monitoring(
        self,
        query: str,
        candidates: List[str],
        log_to_file: bool = True
    ) -> Dict:
        """
        Predict with full monitoring

        Returns:
            {
                'scores': List[float],
                'metrics': PredictionMetrics,
                'alerts': List[str]
            }
        """
        import numpy as np

        # Create pairs
        pairs = [[query, cand] for cand in candidates]

        # Time the prediction
        start = time.time()
        scores = self.model.predict(pairs, show_progress_bar=False)
        latency_ms = (time.time() - start) * 1000

        # Calculate metrics
        metrics = PredictionMetrics(
            timestamp=datetime.now().isoformat(),
            query=query[:100],  # Truncate for logging
            num_candidates=len(candidates),
            latency_ms=latency_ms,
            scores=scores.tolist(),
            top_score=float(scores.max()),
            score_variance=float(scores.var()),
            model_version=self.model_version
        )

        # Check for alerts
        alerts = self._check_alerts(metrics)

        # Log metrics
        if log_to_file:
            self._log_metrics(metrics)

        self.metrics_buffer.append(metrics)
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer.pop(0)

        return {
            'scores': scores.tolist(),
            'metrics': metrics,
            'alerts': alerts
        }

    def _check_alerts(self, metrics: PredictionMetrics) -> List[str]:
        """Check for alert conditions"""
        alerts = []

        # Latency alert
        if metrics.latency_ms > self.alert_threshold_ms:
            alerts.append(
                f"HIGH_LATENCY: {metrics.latency_ms:.1f}ms > {self.alert_threshold_ms}ms"
            )

        # Low confidence alert
        if metrics.top_score < 0.3:
            alerts.append(
                f"LOW_CONFIDENCE: Top score {metrics.top_score:.3f} < 0.3"
            )

        # Low variance alert (all scores similar)
        if metrics.score_variance < 0.01 and metrics.num_candidates > 3:
            alerts.append(
                f"LOW_VARIANCE: Score variance {metrics.score_variance:.4f} < 0.01"
            )

        return alerts

    def _log_metrics(self, metrics: PredictionMetrics):
        """Log metrics to file"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')

    def get_summary_stats(self, last_n: int = None) -> Dict:
        """
        Get summary statistics

        Args:
            last_n: Use only last N predictions (None = all buffered)
        """
        import numpy as np

        if last_n:
            metrics_to_analyze = self.metrics_buffer[-last_n:]
        else:
            metrics_to_analyze = self.metrics_buffer

        if not metrics_to_analyze:
            return {}

        latencies = [m.latency_ms for m in metrics_to_analyze]
        top_scores = [m.top_score for m in metrics_to_analyze]
        num_candidates = [m.num_candidates for m in metrics_to_analyze]

        return {
            "total_requests": len(metrics_to_analyze),
            "latency": {
                "mean_ms": np.mean(latencies),
                "median_ms": np.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "max_ms": np.max(latencies)
            },
            "scores": {
                "mean_top_score": np.mean(top_scores),
                "median_top_score": np.median(top_scores),
                "low_confidence_rate": sum(1 for s in top_scores if s < 0.3) / len(top_scores)
            },
            "candidates": {
                "mean": np.mean(num_candidates),
                "median": np.median(num_candidates),
                "max": np.max(num_candidates)
            }
        }

    def export_metrics(self, output_path: str, format: str = "json"):
        """Export metrics to file"""
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump([asdict(m) for m in self.metrics_buffer], f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if self.metrics_buffer:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.metrics_buffer[0]).keys())
                    writer.writeheader()
                    for metrics in self.metrics_buffer:
                        writer.writerow(asdict(metrics))


# Usage
if __name__ == "__main__":
    encoder = MonitoredCrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alert_threshold_ms=300.0
    )

    # Make predictions
    query = "password reset instructions"
    candidates = [
        "How to reset your password",
        "Pricing information",
        "Contact support",
        "Product features"
    ]

    result = encoder.predict_with_monitoring(query, candidates)

    print(f"Scores: {result['scores']}")
    print(f"Latency: {result['metrics'].latency_ms:.1f}ms")

    if result['alerts']:
        print("Alerts:")
        for alert in result['alerts']:
            print(f"  - {alert}")

    # After many predictions, get summary
    stats = encoder.get_summary_stats(last_n=100)
    print("\nSummary Statistics (last 100):")
    print(json.dumps(stats, indent=2))
```

---

[Continue with Q17-Q20...]

Due to length constraints, I'll now continue with the remaining questions for Module 4. Let me create Q17-Q20 and then the Debugging & Trade-offs sections.
### Q17: How do you implement ensemble cross-encoders?

**Answer:**

```python
from sentence_transformers import CrossEncoder
from typing import List
import numpy as np

class EnsembleCrossEncoder:
    """Ensemble of multiple cross-encoders for better accuracy"""
    
    def __init__(self, model_names: List[str], weights: List[float] = None):
        """
        Initialize ensemble
        
        Args:
            model_names: List of cross-encoder model names
            weights: Optional weights for each model (default: equal)
        """
        self.models = [CrossEncoder(name) for name in model_names]
        
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        """
        Predict using weighted ensemble
        
        Returns:
            Weighted average of predictions from all models
        """
        all_predictions = []
        
        for model in self.models:
            preds = model.predict(pairs, show_progress_bar=False)
            all_predictions.append(preds)
        
        # Weighted average
        ensemble_scores = np.zeros(len(pairs))
        for preds, weight in zip(all_predictions, self.weights):
            ensemble_scores += preds * weight
        
        return ensemble_scores
    
    def predict_with_variance(self, pairs: List[List[str]]) -> dict:
        """
        Predict with confidence estimates based on model agreement
        
        Returns:
            {
                'scores': ensemble scores,
                'variance': variance across models,
                'confidence': confidence based on agreement
            }
        """
        all_predictions = []
        
        for model in self.models:
            preds = model.predict(pairs, show_progress_bar=False)
            all_predictions.append(preds)
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        ensemble_scores = np.average(all_predictions, axis=0, weights=self.weights)
        variance = np.var(all_predictions, axis=0)
        
        # Confidence: inverse of variance (high agreement = high confidence)
        confidence = 1.0 / (1.0 + variance)
        
        return {
            'scores': ensemble_scores,
            'variance': variance,
            'confidence': confidence
        }


# Usage
if __name__ == "__main__":
    # Create ensemble
    models = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    ]
    
    ensemble = EnsembleCrossEncoder(
        models,
        weights=[0.4, 0.4, 0.2]  # Give more weight to larger models
    )
    
    pairs = [
        ["password reset", "How to reset your password"],
        ["pricing info", "Technical documentation"]
    ]
    
    # Standard prediction
    scores = ensemble.predict(pairs)
    print("Ensemble scores:", scores)
    
    # With variance
    result = ensemble.predict_with_variance(pairs)
    for i, pair in enumerate(pairs):
        print(f"\nPair: {pair[0]} | {pair[1]}")
        print(f"  Score: {result['scores'][i]:.3f}")
        print(f"  Variance: {result['variance'][i]:.4f}")
        print(f"  Confidence: {result['confidence'][i]:.3f}")
```

---

### Q18: How do you implement cross-encoder with dynamic temperature scaling?

**Answer:**

```python
import torch
import numpy as np
from sentence_transformers import CrossEncoder

class TemperatureScaledCrossEncoder:
    """Cross-encoder with temperature scaling for calibration"""
    
    def __init__(self, model_name: str, temperature: float = 1.0):
        self.model = CrossEncoder(model_name)
        self.temperature = temperature
        
    def predict(
        self,
        pairs: List[List[str]],
        temperature: float = None
    ) -> np.ndarray:
        """
        Predict with temperature scaling
        
        Args:
            pairs: List of [text1, text2] pairs
            temperature: Override default temperature
                        > 1.0 = softer probabilities
                        < 1.0 = sharper probabilities
        """
        if temperature is None:
            temperature = self.temperature
        
        # Get raw logits
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Apply temperature scaling
        scaled_scores = raw_scores / temperature
        
        return scaled_scores
    
    def calibrate_temperature(
        self,
        val_pairs: List[List[str]],
        val_labels: List[float],
        temp_range: tuple = (0.1, 5.0),
        num_steps: int = 50
    ) -> float:
        """
        Find optimal temperature using validation set
        
        Args:
            val_pairs: Validation pairs
            val_labels: True labels (0 or 1)
            temp_range: Range to search
            num_steps: Number of temperatures to try
        
        Returns:
            Optimal temperature
        """
        from sklearn.metrics import log_loss
        
        # Get raw predictions
        raw_scores = self.model.predict(val_pairs, show_progress_bar=False)
        
        # Try different temperatures
        temperatures = np.linspace(temp_range[0], temp_range[1], num_steps)
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in temperatures:
            # Scale scores
            scaled_scores = 1 / (1 + np.exp(-raw_scores / temp))  # Sigmoid
            
            # Calculate log loss
            loss = log_loss(val_labels, scaled_scores)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        self.temperature = best_temp
        print(f"Optimal temperature: {best_temp:.3f} (log loss: {best_loss:.4f})")
        
        return best_temp


# Usage
if __name__ == "__main__":
    encoder = TemperatureScaledCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Example pairs
    pairs = [
        ["password reset", "How to reset your password"],
        ["pricing", "Product features and specifications"]
    ]
    
    # Predictions with different temperatures
    print("Default (T=1.0):", encoder.predict(pairs, temperature=1.0))
    print("Sharper (T=0.5):", encoder.predict(pairs, temperature=0.5))
    print("Softer (T=2.0):", encoder.predict(pairs, temperature=2.0))
    
    # Calibrate on validation set
    val_pairs = [
        ["query1", "relevant doc"],
        ["query2", "irrelevant doc"],
        # ... more validation pairs
    ]
    val_labels = [1, 0, ...]  # True labels
    
    optimal_temp = encoder.calibrate_temperature(val_pairs, val_labels)
```

---

### Q19: How do you implement query expansion for cross-encoder retrieval?

**Answer:**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Set
import numpy as np

class QueryExpandedRetrieval:
    """Retrieval with query expansion and cross-encoder reranking"""
    
    def __init__(
        self,
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
    
    def expand_query(
        self,
        query: str,
        expansion_method: str = "synonyms",
        num_expansions: int = 3
    ) -> List[str]:
        """
        Expand query with related terms
        
        Methods:
            - synonyms: Add synonyms for key words
            - reformulation: Rephrase the query
            - related_terms: Add semantically related terms
        """
        expansions = [query]  # Always include original
        
        if expansion_method == "synonyms":
            # Simple synonym expansion
            synonyms_map = {
                "password": ["passcode", "credentials", "login"],
                "reset": ["restore", "recover", "change"],
                "pricing": ["cost", "price", "fees"],
                "contact": ["reach", "support", "help"]
            }
            
            words = query.lower().split()
            for word in words:
                if word in synonyms_map:
                    for synonym in synonyms_map[word][:num_expansions]:
                        expanded = query.lower().replace(word, synonym)
                        expansions.append(expanded)
        
        elif expansion_method == "reformulation":
            # Reformulate query in different ways
            templates = [
                f"How to {query}?",
                f"Guide for {query}",
                f"{query} instructions",
                f"Help with {query}"
            ]
            expansions.extend(templates[:num_expansions])
        
        return list(set(expansions))  # Remove duplicates
    
    def retrieve_with_expansion(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 10,
        expansion_method: str = "synonyms",
        fusion_method: str = "rrf"  # reciprocal rank fusion
    ) -> List[tuple]:
        """
        Retrieve using query expansion
        
        Args:
            query: Original query
            corpus: Document corpus
            top_k: Number of results
            expansion_method: How to expand query
            fusion_method: How to combine results (rrf or max)
        
        Returns:
            List of (doc, score) tuples
        """
        # Expand query
        expanded_queries = self.expand_query(query, expansion_method)
        
        # Retrieve with each expanded query
        all_candidates = {}  # doc_id -> scores from different queries
        
        for exp_query in expanded_queries:
            # Embed and retrieve
            query_emb = self.bi_encoder.encode(exp_query)
            corpus_embs = self.bi_encoder.encode(corpus)
            
            # Calculate similarities
            similarities = np.dot(corpus_embs, query_emb)
            
            # Get top candidates
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]
            
            for rank, idx in enumerate(top_indices):
                if idx not in all_candidates:
                    all_candidates[idx] = []
                
                # Store score and rank
                all_candidates[idx].append({
                    'score': similarities[idx],
                    'rank': rank + 1
                })
        
        # Fusion: combine scores from different queries
        final_scores = {}
        
        for doc_id, score_list in all_candidates.items():
            if fusion_method == "rrf":
                # Reciprocal Rank Fusion
                k = 60  # RRF constant
                rrf_score = sum(1 / (k + s['rank']) for s in score_list)
                final_scores[doc_id] = rrf_score
            
            elif fusion_method == "max":
                # Maximum score across queries
                final_scores[doc_id] = max(s['score'] for s in score_list)
        
        # Get top candidates
        top_candidates = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k * 2]
        
        # Rerank with cross-encoder
        candidate_docs = [corpus[idx] for idx, _ in top_candidates]
        pairs = [[query, doc] for doc in candidate_docs]  # Use original query
        rerank_scores = self.cross_encoder.predict(pairs)
        
        # Final ranking
        results = sorted(
            zip(candidate_docs, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return results


# Usage
if __name__ == "__main__":
    retriever = QueryExpandedRetrieval()
    
    corpus = [
        "Password reset instructions: Go to settings...",
        "How to change your password securely",
        "Product pricing and plans",
        "Contact customer support team"
    ]
    
    query = "reset password"
    
    results = retriever.retrieve_with_expansion(
        query,
        corpus,
        top_k=3,
        expansion_method="synonyms"
    )
    
    print(f"Query: {query}")
    print("\nResults:")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc}")
```

---

### Q20: How do you implement cross-encoder with hard negative mining?

**Answer:**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from typing import List, Tuple

class HardNegativeMiner:
    """Mine hard negatives for better cross-encoder training"""
    
    def __init__(
        self,
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
    
    def mine_hard_negatives(
        self,
        query: str,
        positive_doc: str,
        corpus: List[str],
        num_hard_negatives: int = 5,
        strategy: str = "bi_encoder"
    ) -> List[str]:
        """
        Mine hard negatives for a query-positive pair
        
        Args:
            query: Query text
            positive_doc: Known relevant document
            corpus: All documents
            num_hard_negatives: How many to return
            strategy: "bi_encoder", "cross_encoder", or "both"
        
        Returns:
            List of hard negative documents
        """
        if strategy == "bi_encoder":
            return self._mine_with_bi_encoder(
                query, positive_doc, corpus, num_hard_negatives
            )
        elif strategy == "cross_encoder":
            return self._mine_with_cross_encoder(
                query, positive_doc, corpus, num_hard_negatives
            )
        elif strategy == "both":
            # Combine both strategies
            bi_negatives = self._mine_with_bi_encoder(
                query, positive_doc, corpus, num_hard_negatives * 2
            )
            return self._mine_with_cross_encoder(
                query, positive_doc, bi_negatives, num_hard_negatives
            )
    
    def _mine_with_bi_encoder(
        self,
        query: str,
        positive_doc: str,
        corpus: List[str],
        num_hard_negatives: int
    ) -> List[str]:
        """
        Use bi-encoder to find documents similar to query
        but not the positive (confusing negatives)
        """
        # Encode query and corpus
        query_emb = self.bi_encoder.encode(query)
        corpus_embs = self.bi_encoder.encode(corpus)
        
        # Calculate similarities
        similarities = np.dot(corpus_embs, query_emb)
        
        # Sort by similarity (high similarity = hard negative)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top similar docs that aren't the positive
        hard_negatives = []
        for idx in sorted_indices:
            doc = corpus[idx]
            if doc != positive_doc:  # Not the positive
                hard_negatives.append(doc)
                if len(hard_negatives) >= num_hard_negatives:
                    break
        
        return hard_negatives
    
    def _mine_with_cross_encoder(
        self,
        query: str,
        positive_doc: str,
        corpus: List[str],
        num_hard_negatives: int
    ) -> List[str]:
        """
        Use cross-encoder to find documents with high scores
        but not the positive (even harder negatives)
        """
        # Create pairs
        pairs = [[query, doc] for doc in corpus if doc != positive_doc]
        candidate_docs = [doc for doc in corpus if doc != positive_doc]
        
        # Score with cross-encoder
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        # Sort by score (high score = hard negative)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Return top scored docs
        hard_negatives = [candidate_docs[i] for i in sorted_indices[:num_hard_negatives]]
        
        return hard_negatives
    
    def create_training_triplets(
        self,
        queries: List[str],
        positive_docs: List[str],
        corpus: List[str],
        hard_negatives_per_query: int = 3
    ) -> List[Tuple[str, str, str]]:
        """
        Create (query, positive, negative) triplets with hard negatives
        
        Returns:
            List of (query, positive, hard_negative) triplets
        """
        triplets = []
        
        for query, positive in zip(queries, positive_docs):
            # Mine hard negatives
            hard_negatives = self.mine_hard_negatives(
                query,
                positive,
                corpus,
                num_hard_negatives=hard_negatives_per_query,
                strategy="both"
            )
            
            # Create triplets
            for negative in hard_negatives:
                triplets.append((query, positive, negative))
        
        return triplets


# Usage
if __name__ == "__main__":
    miner = HardNegativeMiner()
    
    # Example data
    query = "password reset instructions"
    positive_doc = "How to reset your password: Go to settings and click forgot password"
    
    corpus = [
        positive_doc,
        "How to change your account password",  # Hard negative (similar topic)
        "Login troubleshooting guide",  # Hard negative (related)
        "Product pricing information",  # Easy negative (different topic)
        "Customer support contact details",  # Easy negative
        "Password security best practices",  # Hard negative (contains "password")
    ]
    
    # Mine hard negatives
    hard_negatives = miner.mine_hard_negatives(
        query,
        positive_doc,
        corpus,
        num_hard_negatives=3,
        strategy="both"
    )
    
    print(f"Query: {query}\n")
    print(f"Positive: {positive_doc}\n")
    print("Hard Negatives:")
    for i, neg in enumerate(hard_negatives, 1):
        print(f"  {i}. {neg}")
    
    # Create training triplets
    queries = [query]
    positives = [positive_doc]
    
    triplets = miner.create_training_triplets(
        queries,
        positives,
        corpus,
        hard_negatives_per_query=2
    )
    
    print(f"\nCreated {len(triplets)} training triplets")
```

---

## Debugging & Troubleshooting (21-25)

### Q21: Cross-encoder returns unintuitive scores - how to debug?

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

### Q22: How do you handle out-of-memory errors with cross-encoders?

**Answer:**

**Problem:**
```python
# Large batch causes OOM
pairs = [(query, doc) for doc in million_docs]
scores = cross_encoder.predict(pairs)  # OOM!
```

**Solution 1: Reduce batch size**
```python
from sentence_transformers import CrossEncoder

# Default batch size might be too large
model = CrossEncoder("model-name", max_length=512)

# Reduce batch size
scores = model.predict(
    pairs,
    batch_size=8,  # Reduce from default 32
    show_progress_bar=True
)
```

**Solution 2: Process in chunks**
```python
def predict_in_chunks(model, pairs, chunk_size=100):
    """Process pairs in manageable chunks"""
    all_scores = []
    
    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i:i+chunk_size]
        scores = model.predict(chunk, batch_size=8)
        all_scores.extend(scores)
    
    return np.array(all_scores)

# Usage
scores = predict_in_chunks(cross_encoder, large_pair_list)
```

**Solution 3: Use mixed precision**
```python
import torch
from sentence_transformers import CrossEncoder

# Enable automatic mixed precision
model = CrossEncoder("model-name")

# Wrap predictions with autocast
with torch.cuda.amp.autocast():
    scores = model.predict(pairs, batch_size=16)

# Saves ~40% memory with minimal accuracy loss
```

**Solution 4: Gradient accumulation (training)**
```python
# When training, accumulate gradients
from sentence_transformers import CrossEncoder

model = CrossEncoder("model-name")

# Train with gradient accumulation
model.fit(
    train_dataloader=train_dl,
    epochs=1,
    warmup_steps=100,
    evaluation_steps=500,
    # Effective batch size = batch_size * gradient_accumulation_steps
    gradient_accumulation_steps=4  # Process 4 batches before update
)
```

**Solution 5: Use CPU for very large batches**
```python
# Move model to CPU if GPU OOM
model_cpu = CrossEncoder("model-name", device="cpu")

# Slower but won't crash
scores = model_cpu.predict(pairs, batch_size=32)
```

**Monitoring memory:**
```python
import torch

def predict_with_memory_monitoring(model, pairs):
    """Monitor memory usage during prediction"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    scores = model.predict(pairs, batch_size=16)
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak GPU memory: {peak_memory:.1f} MB")
        
        # Warning if approaching limit
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        if peak_memory > 0.9 * total_memory:
            print("⚠️  WARNING: Approaching GPU memory limit!")
    
    return scores
```

---

### Q23: NLI model gives inconsistent results - how to fix?

**Answer:**

**Problem: Hypothesis phrasing matters**
```python
text = "This product is expensive"

# Bad hypothesis
hypothesis1 = "negative"
score1 = nli_model.predict((text, hypothesis1))  # Low score

# Better hypothesis
hypothesis2 = "This text is negative"
score2 = nli_model.predict((text, hypothesis2))  # Better

# Best hypothesis
hypothesis3 = "This text expresses negative sentiment about pricing"
score3 = nli_model.predict((text, hypothesis3))  # Best
```

**Solution 1: Template consistency**
```python
class ConsistentNLIClassifier:
    """NLI classifier with consistent templates"""
    
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def predict(
        self,
        text: str,
        labels: List[str],
        template: str = "This example is {label}."
    ) -> dict:
        """Use consistent template for all labels"""
        scores = []
        
        for label in labels:
            hypothesis = template.format(label=label)
            
            inputs = self.tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True
            )
            
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            entailment_score = probs[0, 2].item()  # Entailment class
            
            scores.append(entailment_score)
        
        # Normalize
        scores = np.array(scores) / np.sum(scores)
        
        return {
            'labels': labels,
            'scores': scores.tolist()
        }
```

**Solution 2: Calibration**
```python
def calibrate_nli_thresholds(model, val_data):
    """Find optimal thresholds per class"""
    from sklearn.metrics import precision_recall_curve
    
    thresholds = {}
    
    for label in labels:
        # Get predictions for this label
        preds = []
        ground_truth = []
        
        for text, true_labels in val_data:
            score = model.predict(text, [label])
            preds.append(score)
            ground_truth.append(1 if label in true_labels else 0)
        
        # Find optimal threshold
        precision, recall, thresh = precision_recall_curve(
            ground_truth,
            preds
        )
        
        # Choose threshold with best F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        
        thresholds[label] = thresh[best_idx]
    
    return thresholds
```

**Solution 3: Ensemble NLI models**
```python
class EnsembleNLI:
    """Ensemble of NLI models for stability"""
    
    def __init__(self, model_names: List[str]):
        self.models = [
            pipeline("zero-shot-classification", model=name)
            for name in model_names
        ]
    
    def predict(self, text: str, labels: List[str]) -> dict:
        """Aggregate predictions from multiple models"""
        all_scores = []
        
        for model in self.models:
            result = model(text, labels)
            all_scores.append(result['scores'])
        
        # Average scores
        avg_scores = np.mean(all_scores, axis=0)
        
        return {
            'labels': labels,
            'scores': avg_scores.tolist(),
            'std': np.std(all_scores, axis=0).tolist()  # Uncertainty
        }
```

---

### Q24: How do you debug slow cross-encoder inference?

**Answer:**

**Profiling script:**
```python
import time
import torch
from sentence_transformers import CrossEncoder

def profile_cross_encoder(model, pairs):
    """Profile cross-encoder to find bottlenecks"""
    
    print("=== CROSS-ENCODER PROFILING ===\n")
    
    # 1. Check device
    device = next(model.model.parameters()).device
    print(f"Device: {device}")
    
    if device.type == "cpu":
        print("⚠️  WARNING: Running on CPU (slow!)")
        print("   Consider moving to GPU with model.to('cuda')\n")
    
    # 2. Measure tokenization time
    t0 = time.time()
    inputs = model.tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    tokenization_time = (time.time() - t0) * 1000
    print(f"Tokenization: {tokenization_time:.1f}ms")
    
    # 3. Measure model inference time
    t0 = time.time()
    with torch.no_grad():
        outputs = model.model(**inputs.to(device))
    inference_time = (time.time() - t0) * 1000
    print(f"Model inference: {inference_time:.1f}ms")
    
    # 4. Check input length
    input_len = inputs['input_ids'].shape[1]
    print(f"Input length: {input_len} tokens")
    
    if input_len > 128:
        print(f"⚠️  Long inputs detected! Consider truncating to speed up.\n")
    
    # 5. Batch size recommendation
    num_pairs = len(pairs)
    print(f"\nProcessing {num_pairs} pairs")
    
    time_per_pair = inference_time / num_pairs
    print(f"Time per pair: {time_per_pair:.2f}ms")
    
    if time_per_pair > 10:
        print("⚠️  Slow! Try:")
        print("   1. Reduce input length (truncate to 256 tokens)")
        print("   2. Use smaller model (TinyBERT instead of DeBERTa)")
        print("   3. Move to GPU")
        print("   4. Use mixed precision (amp)")


# Usage
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [["query", "document"] for _ in range(32)]

profile_cross_encoder(model, pairs)
```

**Optimization checklist:**
```python
# 1. Use GPU
model = CrossEncoder("model-name", device="cuda")

# 2. Increase batch size
scores = model.predict(pairs, batch_size=64)  # Default is 32

# 3. Truncate long inputs
scores = model.predict(
    pairs,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=False
)

# 4. Use mixed precision
with torch.cuda.amp.autocast():
    scores = model.predict(pairs)

# 5. Use smaller model
# TinyBERT-L-2: 5ms/pair
# MiniLM-L-6: 15ms/pair  
# MiniLM-L-12: 30ms/pair
# DeBERTa-base: 80ms/pair

# 6. Cache predictions
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_predict(query: str, doc: str):
    return model.predict([(query, doc)])[0]
```

---

### Q25: How do you handle multilingual text with cross-encoders?

**Answer:**

**Problem:**
```python
# English cross-encoder on non-English text
query_ru = "Как сбросить пароль?"  # Russian
doc_ru = "Инструкции по сбросу пароля"

# English model fails
eng_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
score = eng_model.predict([(query_ru, doc_ru)])  # Poor results
```

**Solution 1: Use multilingual models**
```python
# Multilingual cross-encoders
multilingual_models = [
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # 100+ languages
    "cross-encoder/mmarco-mBERT-L12-v2",  # BERT multilingual
]

model = CrossEncoder(multilingual_models[0])

# Works on any language
scores = model.predict([
    (query_ru, doc_ru),  # Russian
    ("Comment réinitialiser?", "Instructions de réinitialisation"),  # French
    ("如何重置密码？", "密码重置说明")  # Chinese
])
```

**Solution 2: Translation-based approach**
```python
from transformers import pipeline

class TranslatingCrossEncoder:
    """Cross-encoder with automatic translation"""
    
    def __init__(
        self,
        cross_encoder_name: str,
        translator_name: str = "Helsinki-NLP/opus-mt-mul-en"
    ):
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.translator = pipeline("translation", model=translator_name)
    
    def predict(
        self,
        pairs: List[List[str]],
        source_lang: str = "auto"
    ) -> np.ndarray:
        """Predict with automatic translation to English"""
        
        # Translate all texts to English
        all_texts = []
        for query, doc in pairs:
            all_texts.extend([query, doc])
        
        # Translate in batch
        translations = self.translator(
            all_texts,
            src_lang=source_lang,
            tgt_lang="en"
        )
        
        # Reconstruct pairs
        translated_pairs = []
        for i in range(0, len(translations), 2):
            query_en = translations[i]['translation_text']
            doc_en = translations[i+1]['translation_text']
            translated_pairs.append([query_en, doc_en])
        
        # Score with English cross-encoder
        return self.cross_encoder.predict(translated_pairs)
```

**Solution 3: Language-specific routing**
```python
class MultilingualRouter:
    """Route to language-specific cross-encoders"""
    
    def __init__(self):
        self.models = {
            'en': CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
            'multilingual': CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        }
        
    def detect_language(self, text: str) -> str:
        """Detect language (use fasttext or langdetect)"""
        from langdetect import detect
        try:
            return detect(text)
        except:
            return "en"
    
    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        """Route to appropriate model"""
        
        # Detect language from first query
        lang = self.detect_language(pairs[0][0])
        
        # Use English model for English, multilingual for others
        if lang == "en":
            return self.models['en'].predict(pairs)
        else:
            return self.models['multilingual'].predict(pairs)
```

**Best practices:**
```python
# 1. Use multilingual models for non-English
model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# 2. Don't mix languages in same pair
# BAD: query in English, doc in Russian
# GOOD: both in same language

# 3. For critical applications, validate on each target language
val_data = {
    'en': english_test_pairs,
    'ru': russian_test_pairs,
    'zh': chinese_test_pairs
}

for lang, test_pairs in val_data.items():
    score = evaluate(model, test_pairs)
    print(f"{lang}: {score:.3f}")
```

---

## Trade-offs & Decisions (26-30)

### Q26: Should you use cross-encoder or fine-tune bi-encoder?

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

### Q27: When should you use NLI vs fine-tuned classifier?

**Answer:**

**Decision Matrix:**

| Factor | NLI Zero-Shot | Fine-tuned Classifier |
|--------|--------------|----------------------|
| **Training data** | 0 examples needed | >1K examples needed |
| **Setup time** | Minutes | Days/weeks |
| **Accuracy** | 70-85% | 85-95% |
| **Label changes** | Instant (just modify list) | Requires retraining |
| **Latency** | ~200ms | ~20ms |
| **Domain specificity** | Generic understanding | Domain-optimized |

**Use NLI when:**

```python
# 1. No training data
labels = ["urgent", "not_urgent"]
# Can classify immediately

# 2. Labels change frequently
# This week
labels = ["team_a", "team_b", "team_c"]
# Next week (team_d added)
labels = ["team_a", "team_b", "team_c", "team_d"]
# No retraining!

# 3. Many classes, few examples each
# 50 product categories, 10 examples each
# Not enough to train classifier

# 4. Exploratory phase
# Still figuring out the right labels
# Test different categorization schemes quickly
```

**Use fine-tuned classifier when:**

```python
# 1. Fixed classes with lots of data
# Spam detection: 100K labeled emails, 2 classes
# High accuracy requirements

# 2. Latency critical
# Need to classify 10K items/second
# NLI too slow

# 3. Domain-specific nuances
# Medical: "acute" vs "chronic"
# Legal: "negligence" vs "breach of duty"
# NLI lacks domain knowledge

# 4. Maximum accuracy needed
# Production system, errors are expensive
# Fine-tuned: 94% vs NLI: 82%
```

**Hybrid approach:**

```python
# Phase 1: Start with NLI
nli_classifier = pipeline("zero-shot-classification")

# Phase 2: Collect data
for text in production_data:
    nli_result = nli_classifier(text, labels)
    # Collect user feedback on results
    if user_corrects_label:
        training_data.append((text, correct_label))

# Phase 3: Fine-tune when enough data
if len(training_data) > 5000:
    fine_tuned_model = train_classifier(training_data)
    # Switch to fine-tuned model
```

---

### Q28: How do you choose the right cross-encoder model size?

**Answer:**

**Model Size Trade-offs:**

| Size | Latency (100 pairs) | Accuracy | Memory | Use Case |
|------|---------------------|----------|--------|----------|
| **Tiny (14M)** | 60ms | 85% | 200MB | Real-time, high QPS |
| **Small (22M)** | 120ms | 88% | 300MB | Balanced |
| **Base (66M)** | 280ms | 92% | 500MB | High quality |
| **Large (86M)** | 400ms | 94% | 800MB | Offline, maximum accuracy |

**Decision Framework:**

```python
def choose_cross_encoder(
    latency_requirement_ms: int,
    accuracy_requirement: float,
    queries_per_second: int,
    gpu_available: bool
) -> str:
    """
    Choose appropriate cross-encoder model
    
    Args:
        latency_requirement_ms: Max acceptable latency
        accuracy_requirement: Min accuracy needed (0-1)
        queries_per_second: Expected traffic
        gpu_available: Whether GPU available
    
    Returns:
        Model name
    """
    
    # Very strict latency (<100ms)
    if latency_requirement_ms < 100:
        if accuracy_requirement < 0.87:
            return "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        else:
            return "Consider bi-encoder only (cross-encoder too slow)"
    
    # Moderate latency (100-300ms)
    elif latency_requirement_ms < 300:
        if gpu_available:
            return "cross-encoder/ms-marco-MiniLM-L-6-v2"
        else:
            return "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    
    # Relaxed latency (300-1000ms)
    elif latency_requirement_ms < 1000:
        if accuracy_requirement > 0.92:
            return "cross-encoder/ms-marco-MiniLM-L-12-v2"
        else:
            return "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # No strict latency requirement
    else:
        if accuracy_requirement > 0.93:
            return "cross-encoder/nli-deberta-v3-base"
        else:
            return "cross-encoder/ms-marco-MiniLM-L-12-v2"


# Usage
model_name = choose_cross_encoder(
    latency_requirement_ms=250,
    accuracy_requirement=0.90,
    queries_per_second=100,
    gpu_available=True
)
print(f"Recommended model: {model_name}")
```

**Practical benchmarks:**

```python
# Real-world scenario: Rerank 100 candidates

# TinyBERT (CPU):
# Latency: 60ms
# Accuracy: 85%
# Cost: $0 (CPU)
# Good for: Chatbots, real-time search

# MiniLM-L-6 (GPU T4):
# Latency: 45ms  
# Accuracy: 90%
# Cost: ~$0.50/month
# Good for: Production search, moderate traffic

# MiniLM-L-12 (GPU T4):
# Latency: 95ms
# Accuracy: 93%
# Cost: ~$0.50/month
# Good for: High-quality search

# DeBERTa-base (GPU A100):
# Latency: 40ms
# Accuracy: 95%
# Cost: ~$3/month
# Good for: Critical applications, offline processing
```

**Optimization tips:**

```python
# 1. Start small, scale up if needed
model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
# Measure accuracy on your data
# If < target, try larger model

# 2. Batch size tuning
# Small model: batch_size=64
# Large model: batch_size=16

# 3. Mixed precision for large models
with torch.cuda.amp.autocast():
    scores = large_model.predict(pairs)
# ~30% faster with minimal accuracy loss

# 4. Distillation: Train small model from large
# Deploy small model with accuracy close to large model
```

---

### Q29: Cross-encoder vs semantic similarity - when to use each?

**Answer:**

**Semantic Similarity (Bi-Encoder):**

**How it works:**
```python
# Encode separately, compare embeddings
query_emb = model.encode(query)
doc_emb = model.encode(doc)
score = cosine_similarity(query_emb, doc_emb)
```

**Use when:**
- Need to search millions of documents
- Can precompute document embeddings
- Latency critical (<50ms)
- Finding similar/duplicate content

**Examples:**
```python
# ✅ Duplicate detection
text1 = "How do I reset my password?"
text2 = "Password reset procedure?"
# High similarity → duplicates

# ✅ Document clustering
# Group similar documents together

# ✅ Semantic search (first stage)
# Retrieve top 100 from 1M documents
```

**Cross-Encoder:**

**How it works:**
```python
# Encode together, output relevance score
score = cross_encoder.predict([(query, doc)])
```

**Use when:**
- Need maximum accuracy
- Can afford higher latency (100-500ms)
- Limited candidates to compare (<1000)
- Ranking/reranking task

**Examples:**
```python
# ✅ Reranking search results
# Refine top 100 to top 10

# ✅ Question answering
# Find exact passage that answers question

# ✅ Relevance scoring
# "How relevant is this doc to this query?"
```

**Comparison:**

| Task | Semantic Similarity | Cross-Encoder | Winner |
|------|-------------------|---------------|---------|
| **Search 1M docs** | 50ms | 13 hours | Similarity |
| **Rerank 100 docs** | 82% accuracy | 92% accuracy | Cross-Encoder |
| **Find duplicates** | Very good | Overkill | Similarity |
| **QA (exact answer)** | Fair | Excellent | Cross-Encoder |
| **Clustering** | Perfect fit | Not applicable | Similarity |
| **Real-time (<50ms)** | ✅ | ❌ | Similarity |
| **Batch processing** | ✅ | ✅ | Either |

**Best practice: Use both!**

```python
# Two-stage pipeline
def search_and_rerank(query, corpus):
    # Stage 1: Fast similarity search
    query_emb = bi_encoder.encode(query)
    corpus_embs = bi_encoder.encode(corpus)  # Precomputed
    
    similarities = cosine_similarity([query_emb], corpus_embs)[0]
    top_100_idx = np.argsort(similarities)[::-1][:100]
    candidates = [corpus[i] for i in top_100_idx]
    
    # Stage 2: Accurate cross-encoder reranking
    pairs = [[query, doc] for doc in candidates]
    rerank_scores = cross_encoder.predict(pairs)
    
    final_ranking = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return final_ranking
```

---

### Q30: How do you decide between cross-encoder and LLM-based reranking?

**Answer:**

**Cross-Encoder:**

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
score = model.predict([("query", "document")])[0]
```

**Pros:**
- Fast (~15ms per pair)
- Cheap (one-time model download)
- Consistent outputs
- Easy to deploy
- Fine-tune on your data

**Cons:**
- Limited to training distribution
- May miss nuanced relationships
- Fixed maximum performance

**LLM-based Reranking:**

```python
from openai import OpenAI

client = OpenAI()

prompt = f"""On a scale of 0-10, how relevant is this document to the query?

Query: {query}
Document: {document}

Relevance score (0-10):"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
```

**Pros:**
- Can understand complex nuances
- Leverages world knowledge
- Flexible (change prompt anytime)
- Can explain reasoning

**Cons:**
- Slow (~500ms per pair)
- Expensive ($0.15 per 1M input tokens)
- Variable outputs (temperature)
- Requires API key/internet

**When to use each:**

```python
# Use Cross-Encoder when:
# ✅ High throughput (>100 QPS)
# ✅ Strict latency (<100ms)
# ✅ Cost sensitive
# ✅ Offline processing (millions of pairs)
# ✅ Have training data to fine-tune

# Use LLM when:
# ✅ Complex reasoning needed
# ✅ Few pairs to rerank (<20)
# ✅ Need explanations
# ✅ Domain requires world knowledge
# ✅ Queries are very diverse
```

**Cost comparison:**

```python
# Scenario: Rerank 100 candidates for 10,000 queries

# Cross-Encoder (MiniLM-L-6):
# Time: 10,000 * 100 * 0.015s = 4.2 hours
# Cost: $0 (one-time model download)
# Accuracy: 90%

# LLM (gpt-4o-mini):
# Time: 10,000 * 100 * 0.5s = 139 hours
# Cost: 10,000 * 100 * (50 tokens * $0.00000015) = $7.50
# Accuracy: 93%

# Winner: Cross-Encoder (33x faster, free, 3% accuracy difference)
```

**Hybrid approach:**

```python
def smart_reranking(query, candidates):
    """
    Use cross-encoder for bulk, LLM for edge cases
    """
    
    # Stage 1: Cross-encoder reranking
    pairs = [[query, c] for c in candidates]
    scores = cross_encoder.predict(pairs)
    
    top_results = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Stage 2: LLM for uncertain cases
    uncertain_results = [
        (doc, score) for doc, score in top_results
        if 0.4 < score < 0.6  # Uncertain range
    ]
    
    if uncertain_results:
        # Re-score uncertain ones with LLM
        for doc, _ in uncertain_results:
            llm_score = llm_rerank(query, doc)
            # Update score
    
    return top_results
```

**Recommendation:**

```python
# Default choice
use_cross_encoder = True

# Switch to LLM only if:
if (
    throughput < 10 and  # Low volume
    budget > 100 and  # Have budget
    accuracy_gain_worth_cost  # 3% gain worth $7.50
):
    use_llm = True
```

---

## Additional Resources

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [Cross-Encoder Documentation](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
- [NLI Zero-Shot Guide](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
