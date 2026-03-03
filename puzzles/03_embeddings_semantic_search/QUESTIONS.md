# Deep Dive Q&A - Text Embeddings & Semantic Search

> 30 questions covering embeddings fundamentals, implementation patterns, and production best practices

## Fundamentals (Q1-Q10)

### Q1: What are text embeddings and why are they useful?

**Answer:**

Text embeddings are dense numerical vector representations of text that capture semantic meaning.

**Key properties:**
- Fixed-size vectors (e.g., 384, 768, 1024 dimensions)
- Similar texts → similar vectors (close in vector space)
- Capture meaning, not just keywords: "dog" and "puppy" are close

**Why useful:**
```python
# Instead of keyword matching:
"How to install Python" vs "Python installation guide"  # Different words, same meaning

# Embeddings capture this:
embed1 = model.encode("How to install Python")
embed2 = model.encode("Python installation guide")
similarity = cosine_similarity(embed1, embed2)  # ~0.9 (very similar)
```

**Applications:**
- Semantic search
- Document clustering
- Duplicate detection
- RAG retrieval
- Classification

---

### Q2: What's the difference between word embeddings and sentence embeddings?

**Answer:**

| Aspect | Word Embeddings | Sentence Embeddings |
|--------|-----------------|---------------------|
| Unit | Single word | Sentence/paragraph |
| Examples | Word2Vec, GloVe | SBERT, E5, GTE |
| Context | Static (same vector always) | Contextual |
| Output | One vector per word | One vector per sentence |

**Word embeddings problem:**
```python
# "bank" has same embedding regardless of context
"river bank"  # bank = shore
"money bank"  # bank = financial institution
# Same vector for both!
```

**Sentence embeddings solution:**
```python
# Context is captured
embed1 = model.encode("I deposited money in the bank")
embed2 = model.encode("I walked along the river bank")
# Different vectors, different meanings captured
```

---

### Q3: How do you choose which embedding model to use?

**Answer:**

**Factors to consider:**

1. **Task type:**
   - Semantic search: `all-MiniLM-L6-v2`, `gte-base`
   - Retrieval: `e5-base-v2`, `bge-base`
   - Clustering: `all-mpnet-base-v2`

2. **Performance vs Speed:**
   | Model | Dimensions | Speed | Quality |
   |-------|------------|-------|---------|
   | `all-MiniLM-L6-v2` | 384 | Fast | Good |
   | `all-mpnet-base-v2` | 768 | Medium | Better |
   | `gte-large` | 1024 | Slow | Best |

3. **Domain:**
   - General: `all-MiniLM-L6-v2`
   - Financial: consider fine-tuning
   - Multilingual: `paraphrase-multilingual-MiniLM-L12-v2`

4. **Check MTEB leaderboard** for benchmarks

**Decision flow:**
```
Start → Need speed? → Yes → all-MiniLM-L6-v2
                   → No → Need best quality? → Yes → gte-large
                                             → No → gte-base
```

---

### Q4: What is cosine similarity and when do you use it?

**Answer:**

**Definition:**
Cosine similarity measures the angle between two vectors, ignoring magnitude.

```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

**Range:** [-1, 1]
- 1 = identical direction (most similar)
- 0 = orthogonal (unrelated)
- -1 = opposite direction (least similar)

**When to use:**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity
embed1 = model.encode("Python programming")
embed2 = model.encode("Python coding")
similarity = cosine_similarity([embed1], [embed2])[0][0]
# ~0.95 - very similar
```

**Advantages:**
- Invariant to vector magnitude (length)
- Good for comparing texts of different lengths
- Most common for NLP tasks

**When NOT to use:**
- When magnitude carries information
- When vectors are already normalized (use dot product - faster)

---

### Q5: What's the difference between cosine similarity and dot product?

**Answer:**

**Mathematical relationship:**
```
cosine(A, B) = dot(A, B) / (||A|| × ||B||)
```

If vectors are **L2 normalized** (||A|| = ||B|| = 1):
```
cosine(A, B) = dot(A, B)
```

**Practical comparison:**

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 4, 6])  # Same direction, different magnitude

# Dot product
dot = np.dot(a, b)  # 28

# Cosine similarity
cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # 1.0

# After normalization
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
np.dot(a_norm, b_norm)  # 1.0 - same as cosine!
```

**When to use which:**
- **Dot product:** Vectors already normalized (faster, no division)
- **Cosine:** Vectors not normalized, or need consistent scale

---

### Q6: How do you normalize embeddings and why?

**Answer:**

**L2 Normalization:**
```python
import numpy as np

def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm

# After normalization: ||vector|| = 1
```

**Why normalize:**

1. **Consistent similarity scores:**
```python
# Without normalization
embed1 = np.array([10, 20, 30])
embed2 = np.array([1, 2, 3])
np.dot(embed1, embed2)  # 140 (depends on magnitude)

# With normalization
e1_norm = normalize(embed1)
e2_norm = normalize(embed2)
np.dot(e1_norm, e2_norm)  # 1.0 (pure direction comparison)
```

2. **Faster computation:** Dot product = cosine similarity
3. **Better for vector DBs:** Many indexes assume normalized vectors

**Sentence Transformers:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences, normalize_embeddings=True)
```

---

### Q7: How do you measure the quality of embeddings?

**Answer:**

**Intrinsic evaluation:**

1. **Semantic Textual Similarity (STS):**
```python
# Compute correlation between predicted similarity and human scores
from scipy.stats import spearmanr

predicted_similarities = [cosine(e1, e2) for e1, e2 in pairs]
human_scores = [...]  # From STS benchmark
correlation = spearmanr(predicted_similarities, human_scores)
```

2. **Retrieval metrics:**
```python
# Recall@k: % of relevant docs in top-k
# MRR: Mean Reciprocal Rank
# NDCG: Normalized Discounted Cumulative Gain
```

**Extrinsic evaluation:**

Evaluate on downstream task:
```python
# If using for classification
accuracy = classifier.score(embeddings, labels)

# If using for search
recall_at_10 = evaluate_retrieval(queries, corpus, k=10)
```

**MTEB benchmark** covers multiple tasks:
- Retrieval
- Clustering
- Classification
- STS

---

### Q8: How do you batch process embeddings efficiently?

**Answer:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# BAD: One at a time
embeddings = []
for text in texts:
    embeddings.append(model.encode(text))  # Slow!

# GOOD: Batch processing
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# BETTER: With GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
embeddings = model.encode(texts, batch_size=128)
```

**For very large datasets:**
```python
# Process in chunks to manage memory
chunk_size = 10000
all_embeddings = []

for i in range(0, len(texts), chunk_size):
    chunk = texts[i:i+chunk_size]
    chunk_embeddings = model.encode(chunk, batch_size=64)
    all_embeddings.append(chunk_embeddings)

embeddings = np.vstack(all_embeddings)
```

---

### Q9: What is embedding dimensionality and how does it affect performance?

**Answer:**

**Common dimensions:**
- 384 (MiniLM-L6)
- 768 (BERT-base, MPNet)
- 1024 (Large models)
- 1536 (OpenAI ada-002)

**Trade-offs:**

| Dimension | Storage | Speed | Quality |
|-----------|---------|-------|---------|
| 384 | 1.5 KB/vec | Fast | Good |
| 768 | 3 KB/vec | Medium | Better |
| 1024 | 4 KB/vec | Slower | Best |

**Storage calculation:**
```python
# 1M documents, 768-dim, float32
memory = 1_000_000 * 768 * 4  # bytes
memory_gb = memory / 1e9  # ~3 GB
```

**Dimensionality reduction:**
```python
from sklearn.decomposition import PCA

# Reduce 768 → 256
pca = PCA(n_components=256)
reduced = pca.fit_transform(embeddings)

# Trade-off: faster search, some quality loss
```

---

### Q10: How do you handle long texts that exceed model's max length?

**Answer:**

Most models have max length (e.g., 512 tokens).

**Strategy 1: Truncation**
```python
# Simple but loses information
embedding = model.encode(text, truncate=True)
```

**Strategy 2: Chunking + Pooling**
```python
def embed_long_text(text, model, chunk_size=256, overlap=50):
    # Split into chunks
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])

    # Embed chunks
    chunk_embeddings = model.encode(chunks)

    # Pool (mean, max, or weighted)
    return np.mean(chunk_embeddings, axis=0)
```

**Strategy 3: Use long-context model**
```python
# Models designed for long text
model = SentenceTransformer('BAAI/bge-m3')  # 8192 tokens
```

**Strategy 4: Hierarchical**
```python
# Embed paragraphs separately, then combine
paragraph_embeddings = model.encode(paragraphs)
document_embedding = np.mean(paragraph_embeddings, axis=0)
```

---

## Implementation (Q11-Q20)

### Q11: How do you implement semantic search from scratch?

**Answer:**

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None

    def index(self, documents):
        """Index documents for search."""
        self.corpus = documents
        self.corpus_embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def search(self, query, top_k=5):
        """Find most similar documents."""
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        # Compute similarities (dot product since normalized)
        similarities = np.dot(self.corpus_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.corpus[idx],
                'score': similarities[idx]
            })

        return results

# Usage
searcher = SemanticSearch()
searcher.index(documents)
results = searcher.search("How to install Python?")
```

---

### Q12: How do you find duplicate or near-duplicate texts?

**Answer:**

```python
from sentence_transformers import SentenceTransformer, util

def find_duplicates(texts, threshold=0.9):
    """Find near-duplicate texts using embeddings."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Compute pairwise similarities
    cosine_scores = util.cos_sim(embeddings, embeddings)

    duplicates = []
    seen = set()

    for i in range(len(texts)):
        if i in seen:
            continue
        for j in range(i + 1, len(texts)):
            if cosine_scores[i][j] > threshold:
                duplicates.append({
                    'text1': texts[i],
                    'text2': texts[j],
                    'similarity': cosine_scores[i][j].item()
                })
                seen.add(j)

    return duplicates

# Usage
texts = [
    "How to install Python",
    "Python installation guide",
    "Best restaurants in NYC",
    "Installing Python on Windows"
]
duplicates = find_duplicates(texts, threshold=0.85)
```

---

### Q13: How do you cluster texts using embeddings?

**Answer:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

def cluster_texts(texts, n_clusters=5, method='kmeans'):
    # Get embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    if method == 'kmeans':
        # K-means (need to specify n_clusters)
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clustering.fit_predict(embeddings)

    elif method == 'dbscan':
        # DBSCAN (automatic cluster detection)
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)

    # Group texts by cluster
    clusters = {}
    for text, label in zip(texts, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)

    return clusters

# Usage
texts = ["Python tutorial", "Python guide", "NYC restaurants", "Food in New York"]
clusters = cluster_texts(texts, n_clusters=2)
# {0: ["Python tutorial", "Python guide"], 1: ["NYC restaurants", "Food in New York"]}
```

---

### Q14: How do you visualize embeddings?

**Answer:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings(texts, labels=None):
    # Get embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))

    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = [l == label for l in labels]
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=label,
                alpha=0.7
            )
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

    # Add text labels
    for i, text in enumerate(texts):
        plt.annotate(text[:30], (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title('Text Embeddings Visualization (t-SNE)')
    plt.tight_layout()
    plt.show()

# Usage
texts = ["cat", "dog", "car", "truck", "happy", "sad"]
labels = ["animal", "animal", "vehicle", "vehicle", "emotion", "emotion"]
visualize_embeddings(texts, labels)
```

---

### Q15: How do you save and load embeddings?

**Answer:**

```python
import numpy as np
import json

# Save embeddings
def save_embeddings(embeddings, ids, path):
    """Save embeddings and IDs to files."""
    np.save(f"{path}/embeddings.npy", embeddings)
    with open(f"{path}/ids.json", 'w') as f:
        json.dump(ids, f)

# Load embeddings
def load_embeddings(path):
    """Load embeddings and IDs from files."""
    embeddings = np.load(f"{path}/embeddings.npy")
    with open(f"{path}/ids.json") as f:
        ids = json.load(f)
    return embeddings, ids

# Usage
embeddings = model.encode(texts)
ids = ["doc_1", "doc_2", "doc_3"]

save_embeddings(embeddings, ids, "./cache")
loaded_embeddings, loaded_ids = load_embeddings("./cache")
```

**With metadata (pickle):**
```python
import pickle

data = {
    'embeddings': embeddings,
    'texts': texts,
    'metadata': metadata
}

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
```

---

### Q16: How do you compute similarity for a large corpus efficiently?

**Answer:**

```python
import numpy as np
from sentence_transformers import SentenceTransformer, util

def efficient_search(query_embedding, corpus_embeddings, top_k=10):
    """
    Efficient similarity search for large corpus.

    For very large corpus (>100K), use FAISS or other vector DBs.
    """
    # Method 1: NumPy (good for <100K docs)
    similarities = np.dot(corpus_embeddings, query_embedding)
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    return top_indices, similarities[top_indices]

# Method 2: sentence-transformers util (with batching)
def batch_search(queries, corpus, model, top_k=10, batch_size=100):
    """Search with query batching."""
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True)

    all_results = []
    for i in range(0, len(queries), batch_size):
        query_batch = queries[i:i+batch_size]
        query_embeddings = model.encode(query_batch, normalize_embeddings=True)

        # Compute similarities for batch
        similarities = util.cos_sim(query_embeddings, corpus_embeddings)

        for sim in similarities:
            top_results = torch.topk(sim, k=top_k)
            all_results.append({
                'indices': top_results.indices.tolist(),
                'scores': top_results.values.tolist()
            })

    return all_results
```

---

### Q17: How do you handle multilingual text?

**Answer:**

```python
from sentence_transformers import SentenceTransformer

# Use multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

texts = [
    "How to install Python",           # English
    "Comment installer Python",         # French
    "Как установить Python",            # Russian
    "如何安装Python"                     # Chinese
]

embeddings = model.encode(texts)

# All embeddings in same space - can compare across languages!
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
# High similarity between all (same meaning, different languages)
```

**Language-specific considerations:**
```python
# For best quality, use language-specific models if available
# Or use multilingual models like:
# - paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)
# - LaBSE (100+ languages)
# - BAAI/bge-m3 (multilingual, long context)
```

---

### Q18: How do you implement query expansion with embeddings?

**Answer:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def expand_query(query, model, synonym_corpus, top_k=3):
    """
    Expand query with semantically similar terms.
    """
    query_embedding = model.encode(query, normalize_embeddings=True)
    corpus_embeddings = model.encode(synonym_corpus, normalize_embeddings=True)

    similarities = np.dot(corpus_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    expanded_terms = [synonym_corpus[i] for i in top_indices]
    return expanded_terms

# Usage
model = SentenceTransformer('all-MiniLM-L6-v2')
synonym_corpus = ["install", "setup", "configure", "download", "Python", "programming"]

query = "How to install"
expanded = expand_query(query, model, synonym_corpus)
# ["install", "setup", "configure"]
```

---

### Q19: How do you implement semantic caching?

**Answer:**

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    """Cache responses based on semantic similarity of queries."""

    def __init__(self, threshold=0.95):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.cache = []  # [(embedding, query, response)]

    def get(self, query):
        """Get cached response if similar query exists."""
        if not self.cache:
            return None

        query_embedding = self.model.encode(query, normalize_embeddings=True)

        for cached_embedding, cached_query, response in self.cache:
            similarity = np.dot(query_embedding, cached_embedding)
            if similarity >= self.threshold:
                return response

        return None

    def set(self, query, response):
        """Cache a query-response pair."""
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        self.cache.append((query_embedding, query, response))

# Usage
cache = SemanticCache(threshold=0.9)
cache.set("How to install Python?", "Visit python.org...")

# Later, similar query hits cache
response = cache.get("Python installation instructions")
# Returns cached response!
```

---

### Q20: How do you benchmark embedding models?

**Answer:**

```python
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def benchmark_model(model_name, test_pairs, test_labels):
    """
    Benchmark embedding model on speed and quality.

    test_pairs: [(text1, text2), ...]
    test_labels: [similarity_score, ...] (0-1)
    """
    model = SentenceTransformer(model_name)

    # Speed benchmark
    texts = [t for pair in test_pairs for t in pair]
    start = time.time()
    embeddings = model.encode(texts, batch_size=32)
    encode_time = time.time() - start

    # Quality benchmark (correlation with human labels)
    similarities = []
    for i in range(0, len(embeddings), 2):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)

    from scipy.stats import spearmanr
    correlation = spearmanr(similarities, test_labels).correlation

    return {
        'model': model_name,
        'encode_time_sec': encode_time,
        'texts_per_sec': len(texts) / encode_time,
        'spearman_correlation': correlation,
        'dimensions': embeddings.shape[1]
    }

# Compare models
models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'thenlper/gte-base']
results = [benchmark_model(m, test_pairs, test_labels) for m in models]
```

---

## Debugging & Troubleshooting (Q21-Q25)

### Q21: Your semantic search returns irrelevant results. How do you debug?

**Answer:**

**Step 1: Check embeddings**
```python
# Are embeddings reasonable?
query_emb = model.encode("Python programming")
print(f"Shape: {query_emb.shape}")
print(f"Norm: {np.linalg.norm(query_emb)}")
print(f"Sample values: {query_emb[:5]}")
```

**Step 2: Check similarity distribution**
```python
# Plot similarity scores
import matplotlib.pyplot as plt
similarities = np.dot(corpus_embeddings, query_emb)
plt.hist(similarities, bins=50)
plt.title("Similarity Distribution")
plt.show()
# Should see some separation between relevant and irrelevant
```

**Step 3: Check top results manually**
```python
# Look at top-10 results
for idx in top_10_indices:
    print(f"Score: {similarities[idx]:.3f} | Text: {corpus[idx][:100]}")
```

**Step 4: Check model choice**
```python
# Try different models
models = ['all-MiniLM-L6-v2', 'thenlper/gte-base']
for m in models:
    model = SentenceTransformer(m)
    # Compare results
```

**Step 5: Check data quality**
- Empty strings?
- Special characters?
- Wrong language?

---

### Q22: Embeddings are too slow. How do you optimize?

**Answer:**

**1. Use GPU:**
```python
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

**2. Increase batch size:**
```python
embeddings = model.encode(texts, batch_size=128)  # Default is 32
```

**3. Use smaller model:**
```python
# Instead of all-mpnet-base-v2 (768-dim)
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, 5x faster
```

**4. Pre-compute and cache:**
```python
# Compute once, save to disk
embeddings = model.encode(corpus)
np.save('corpus_embeddings.npy', embeddings)

# Load for search
corpus_embeddings = np.load('corpus_embeddings.npy')
```

**5. Use ONNX/TensorRT:**
```python
# Export to ONNX for faster inference
# See optimum library
```

---

### Q23: Memory issues with large corpus. How do you handle?

**Answer:**

**1. Process in chunks:**
```python
chunk_size = 10000
all_embeddings = []

for i in range(0, len(texts), chunk_size):
    chunk = texts[i:i+chunk_size]
    emb = model.encode(chunk)
    all_embeddings.append(emb)

embeddings = np.vstack(all_embeddings)
```

**2. Use float16:**
```python
embeddings = model.encode(texts).astype(np.float16)
# 50% memory reduction
```

**3. Use memory-mapped files:**
```python
# Save to disk, load on-demand
np.save('embeddings.npy', embeddings)
embeddings = np.load('embeddings.npy', mmap_mode='r')
```

**4. Use vector database:**
```python
# FAISS, ChromaDB, etc. handle memory efficiently
import faiss
index = faiss.IndexFlatIP(768)  # Handles millions of vectors
```

---

### Q24: Similarity scores seem wrong. How do you debug?

**Answer:**

**1. Check normalization:**
```python
# Are vectors normalized?
norms = np.linalg.norm(embeddings, axis=1)
print(f"Norms: min={norms.min():.3f}, max={norms.max():.3f}")
# Should be ~1.0 if normalized
```

**2. Check for NaN/Inf:**
```python
print(f"NaN: {np.isnan(embeddings).any()}")
print(f"Inf: {np.isinf(embeddings).any()}")
```

**3. Verify similarity calculation:**
```python
# Manual check
a = embeddings[0]
b = embeddings[1]

# Should be equivalent if normalized
dot = np.dot(a, b)
cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Dot: {dot:.4f}, Cosine: {cosine:.4f}")
```

**4. Check input text:**
```python
# Empty or weird texts?
for i, text in enumerate(texts):
    if len(text) < 5 or not text.strip():
        print(f"Suspicious text at {i}: '{text}'")
```

---

### Q25: Model outputs different results each time. Why?

**Answer:**

**Usually NOT a problem:**
Sentence Transformer inference is deterministic. If you see different results:

**1. Check for randomness in your code:**
```python
# Set seeds everywhere
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

**2. Check batch processing order:**
```python
# Different batch sizes can give slightly different floating point results
# Use consistent batch_size
embeddings = model.encode(texts, batch_size=32)  # Always same
```

**3. Check GPU non-determinism:**
```python
# For exact reproducibility on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Trade-offs & Decisions (Q26-Q30)

### Q26: When would you use OpenAI embeddings vs open-source?

**Answer:**

**OpenAI embeddings (text-embedding-ada-002, text-embedding-3-*):**

Pros:
- Very high quality
- Easy API, no infrastructure
- Good for multilingual

Cons:
- Cost ($0.0001/1K tokens)
- Data sent to OpenAI (privacy)
- Rate limits
- Internet required

**Open-source (Sentence Transformers):**

Pros:
- Free
- Private (runs locally)
- No rate limits
- Customizable (fine-tuning)

Cons:
- Need GPU for speed
- Self-host infrastructure
- May need to choose right model

**Decision matrix:**

| Factor | OpenAI | Open-source |
|--------|--------|-------------|
| Privacy critical | No | Yes |
| High volume | No | Yes |
| Budget limited | No | Yes |
| No infrastructure | Yes | No |
| Need fine-tuning | No | Yes |

---

### Q27: Cosine similarity vs Euclidean distance - when to use which?

**Answer:**

**Cosine similarity:**
- Measures angle (direction)
- Ignores magnitude
- Range: [-1, 1]
- **Use when:** text length varies, care about semantic direction

**Euclidean distance:**
- Measures absolute distance
- Considers magnitude
- Range: [0, ∞)
- **Use when:** magnitude matters, clustering

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

a = np.array([1, 2, 3])
b = np.array([2, 4, 6])  # Same direction, different magnitude
c = np.array([3, 2, 1])  # Different direction

# Cosine: a and b are identical (same direction)
cosine_similarity([a], [b])  # 1.0
cosine_similarity([a], [c])  # 0.71

# Euclidean: a and b are different (different magnitude)
euclidean_distances([a], [b])  # 3.74
euclidean_distances([a], [c])  # 2.83
```

**For NLP:** Usually cosine similarity (direction = meaning)

---

### Q28: When should you fine-tune embeddings vs use pre-trained?

**Answer:**

**Use pre-trained when:**
- General domain (no specialized vocabulary)
- Limited training data (<1000 pairs)
- Quick prototype needed
- Good performance with general model

**Fine-tune when:**
- Domain-specific vocabulary (medical, legal, financial)
- Have labeled similarity data
- Pre-trained underperforms
- Need best possible quality

**Fine-tuning approaches:**
```python
# 1. Contrastive learning (pairs)
from sentence_transformers import SentenceTransformer, InputExample, losses

train_examples = [
    InputExample(texts=["query", "positive doc"]),
    # ...
]

model = SentenceTransformer('all-MiniLM-L6-v2')
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_loader, train_loss)])

# 2. Triplet loss (query, positive, negative)
# 3. SetFit for few-shot
```

---

### Q29: Single embedding per document vs multiple embeddings (chunking)?

**Answer:**

**Single embedding:**
- Simple
- Fast retrieval
- May miss details in long docs
- Works for short texts

**Multiple embeddings (chunking):**
- Better for long documents
- More storage needed
- More complex retrieval
- Better precision

**Decision:**
```
Document length < 256 tokens → Single embedding
Document length > 256 tokens → Consider chunking
```

**Chunking strategies:**
```python
# 1. Fixed size chunks
chunks = [text[i:i+256] for i in range(0, len(text), 256)]

# 2. Semantic chunks (by paragraph/section)
chunks = text.split('\n\n')

# 3. Overlapping chunks
chunk_size, overlap = 256, 50
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
```

---

### Q30: Real-time embedding vs batch pre-computation?

**Answer:**

**Real-time:**
```python
# Embed at query time
def search(query):
    query_embedding = model.encode(query)  # ~10-50ms
    results = find_similar(query_embedding, corpus_embeddings)
    return results
```

**Batch pre-computation:**
```python
# Pre-compute corpus embeddings
corpus_embeddings = model.encode(all_documents)  # Once
np.save('embeddings.npy', corpus_embeddings)

# At search time, only embed query
def search(query):
    query_embedding = model.encode(query)  # ~10ms
    corpus_embeddings = np.load('embeddings.npy')
    results = find_similar(query_embedding, corpus_embeddings)
    return results
```

**When to use which:**

| Scenario | Strategy |
|----------|----------|
| Static corpus | Pre-compute all |
| Dynamic corpus | Pre-compute + update incrementally |
| Query-only | Real-time query embedding |
| All dynamic | Real-time everything (slow) |

**Hybrid:**
```python
# Pre-compute corpus, real-time queries
# Update corpus embeddings in background when new docs added
```
