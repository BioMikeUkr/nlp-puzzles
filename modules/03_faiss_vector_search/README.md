# Module 3: FAISS for Vector Search

> Production-ready vector search with Facebook AI Similarity Search

## Why This Matters

FAISS is the industry standard for efficient similarity search at scale. While NumPy works for small datasets (<10K vectors), production systems need FAISS to handle millions of vectors with millisecond latency.

## Key Concepts

### Why FAISS?

**NumPy limitations:**
```python
# NumPy approach - O(n) for every search
similarities = np.dot(corpus_embeddings, query_embedding)
top_k = np.argsort(similarities)[-10:][::-1]
# 1M vectors × 384 dims = 3.8 billion operations per search
```

**FAISS advantages:**
- 10-100x faster search
- Efficient memory usage
- GPU acceleration support
- Multiple index types for different trade-offs

### Index Types

| Index | Search Time | Accuracy | Memory | Use Case |
|-------|-------------|----------|--------|----------|
| `IndexFlatIP` | Slow (exact) | 100% | High | <100K vectors, highest quality |
| `IndexIVFFlat` | Fast | 95-99% | Medium | 100K-10M vectors |
| `IndexHNSWFlat` | Very Fast | 95-99% | High | <10M vectors, low latency |

### Similarity Metrics

```python
# Inner Product (dot product) - for normalized vectors
index = faiss.IndexFlatIP(dimension)

# L2 Distance (Euclidean)
index = faiss.IndexFlatL2(dimension)
```

## Documentation & Resources

- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [Index Selection Guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)

## Self-Assessment Checklist

- [ ] I understand when to use FAISS vs NumPy
- [ ] I can create and populate a FAISS index
- [ ] I know the difference between IndexFlatIP and IndexFlatL2
- [ ] I can save and load FAISS indexes
- [ ] I understand IVF index parameters (nlist, nprobe)
- [ ] I can measure search performance (latency, recall)

---

## Deep Dive Q&A (30 Questions)

### Basics & Setup (1-6)

#### Q1: What is FAISS and when should you use it vs NumPy?

**Answer:**

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.

**Use NumPy when:**
- < 10K vectors
- Exact search required
- Simple prototype

**Use FAISS when:**
- > 100K vectors
- Need sub-second search on millions of vectors
- Production system with latency requirements
- GPU acceleration needed

**Example:**
```python
import numpy as np
import faiss
import time

# NumPy approach
embeddings = np.random.randn(1_000_000, 384).astype('float32')
query = np.random.randn(384).astype('float32')

start = time.time()
similarities = np.dot(embeddings, query)
top_10 = np.argsort(similarities)[-10:][::-1]
print(f"NumPy: {time.time() - start:.3f}s")  # ~0.5s

# FAISS approach
index = faiss.IndexFlatIP(384)
index.add(embeddings)

start = time.time()
D, I = index.search(query.reshape(1, -1), 10)
print(f"FAISS: {time.time() - start:.3f}s")  # ~0.05s (10x faster)
```

---

#### Q2: What's the difference between IndexFlatIP and IndexFlatL2?

**Answer:**

**IndexFlatIP** - Inner Product (dot product):
```python
index = faiss.IndexFlatIP(dimension)
# similarity = np.dot(query, vector)
# Higher score = more similar
# Range: [-∞, +∞], or [-1, 1] for normalized vectors
```

**IndexFlatL2** - L2 Distance (Euclidean):
```python
index = faiss.IndexFlatL2(dimension)
# distance = ||query - vector||²
# Lower score = more similar
# Range: [0, +∞]
```

**When to use which:**
- **IndexFlatIP**: Normalized embeddings, cosine similarity
- **IndexFlatL2**: Euclidean distance, magnitude matters

**Practical example:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["cat", "dog", "car"]

# Get normalized embeddings
embeddings = model.encode(texts, normalize_embeddings=True)

# Use IndexFlatIP for normalized vectors (= cosine similarity)
index = faiss.IndexFlatIP(384)
index.add(embeddings)

query = model.encode(["kitten"], normalize_embeddings=True)
D, I = index.search(query, k=2)
# Returns: [0, 1] - "cat", "dog" (most similar)
```

---

#### Q3: How do you create and populate a FAISS index?

**Answer:**

```python
import faiss
import numpy as np

# 1. Create index
dimension = 384
index = faiss.IndexFlatIP(dimension)

# 2. Check if empty
print(f"Total vectors: {index.ntotal}")  # 0

# 3. Add vectors (must be float32, shape (n, dimension))
embeddings = np.random.randn(1000, 384).astype('float32')

# Normalize for IndexFlatIP
faiss.normalize_L2(embeddings)  # In-place normalization

# Add to index
index.add(embeddings)

print(f"Total vectors: {index.ntotal}")  # 1000

# 4. Search
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)

k = 5  # Top-5 results
distances, indices = index.search(query, k)

print(f"Top-5 indices: {indices[0]}")
print(f"Top-5 scores: {distances[0]}")
```

**Important:**
- Vectors must be `float32` (not float64)
- Shape must be `(n_vectors, dimension)`
- For IndexFlatIP, normalize vectors first

---

#### Q4: How do you save and load a FAISS index?

**Answer:**

```python
import faiss
import numpy as np

# Create and populate index
dimension = 384
index = faiss.IndexFlatIP(dimension)
embeddings = np.random.randn(10000, 384).astype('float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Save to disk
faiss.write_index(index, "my_index.faiss")

# Load from disk
loaded_index = faiss.read_index("my_index.faiss")

print(f"Original: {index.ntotal} vectors")
print(f"Loaded: {loaded_index.ntotal} vectors")

# Verify it works
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)
D1, I1 = index.search(query, 5)
D2, I2 = loaded_index.search(query, 5)

assert np.array_equal(I1, I2), "Indices should match"
```

**File size estimation:**
```python
# IndexFlatIP/L2: dimension × ntotal × 4 bytes
# Example: 384 dims × 1M vectors × 4 bytes = ~1.5 GB
```

---

#### Q5: What data types and shapes does FAISS expect?

**Answer:**

**Critical requirements:**
```python
import numpy as np
import faiss

# ❌ WRONG
embeddings = np.random.randn(1000, 384)  # float64 - won't work!
embeddings_1d = np.random.randn(384)  # 1D - won't work!

# ✅ CORRECT
embeddings = np.random.randn(1000, 384).astype('float32')  # float32, 2D

index = faiss.IndexFlatIP(384)
index.add(embeddings)  # Works!

# Query must also be 2D
query = np.random.randn(384).astype('float32')  # 1D
query = query.reshape(1, -1)  # Convert to (1, 384)
D, I = index.search(query, k=5)  # Works!
```

**Common error handling:**
```python
def prepare_for_faiss(embeddings):
    """Ensure embeddings are FAISS-compatible."""
    # Convert to numpy if needed
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    # Convert to float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype('float32')

    # Ensure 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    return embeddings
```

---

#### Q6: How do you search with FAISS?

**Answer:**

```python
import faiss
import numpy as np

# Setup
index = faiss.IndexFlatIP(384)
corpus_embeddings = np.random.randn(10000, 384).astype('float32')
faiss.normalize_L2(corpus_embeddings)
index.add(corpus_embeddings)

# Single query
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)

k = 10  # Return top-10
distances, indices = index.search(query, k)

print(f"Shape of distances: {distances.shape}")  # (1, 10)
print(f"Shape of indices: {indices.shape}")  # (1, 10)
print(f"Top-10 indices: {indices[0]}")
print(f"Top-10 scores: {distances[0]}")

# Batch queries (multiple queries at once)
queries = np.random.randn(100, 384).astype('float32')
faiss.normalize_L2(queries)

D, I = index.search(queries, k=5)
print(f"Batch shape: {D.shape}")  # (100, 5)

# For each query
for i in range(len(queries)):
    print(f"Query {i}: top-5 = {I[i]}, scores = {D[i]}")
```

**Understanding output:**
- `distances`: similarity scores (higher = more similar for IP)
- `indices`: positions in the corpus (0 to ntotal-1)
- Both have shape `(n_queries, k)`

---

### IVF Indexes (7-12)

#### Q7: What is an IVF index and when to use it?

**Answer:**

**IVF (Inverted File Index)** - Approximate search using clustering.

**How it works:**
1. Cluster corpus into `nlist` clusters (using k-means)
2. At search time, only search `nprobe` closest clusters
3. Much faster than exhaustive search

**Trade-off:**
- Speed: 10-100x faster than Flat
- Accuracy: 95-99% recall (may miss some results)

```python
import faiss

# Create IVF index
dimension = 384
nlist = 100  # Number of clusters

# Step 1: Create quantizer (for clustering)
quantizer = faiss.IndexFlatIP(dimension)

# Step 2: Create IVF index
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Step 3: Train on data (learn clusters)
embeddings = np.random.randn(100000, 384).astype('float32')
faiss.normalize_L2(embeddings)

index.train(embeddings)  # Must train before adding!
print(f"Is trained: {index.is_trained}")

# Step 4: Add data
index.add(embeddings)

# Step 5: Set nprobe (how many clusters to search)
index.nprobe = 10  # Search 10 clusters (out of 100)

# Search
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)
D, I = index.search(query, k=10)
```

**When to use:**
- 100K to 10M vectors
- Can tolerate 1-5% recall loss
- Need faster search than Flat

---

#### Q8: How do you choose nlist and nprobe for IVF?

**Answer:**

**nlist** (number of clusters):
```python
# Rule of thumb: nlist = sqrt(N) to 4*sqrt(N)
N = 1_000_000  # 1M vectors
nlist = int(4 * np.sqrt(N))  # ~4000

# Practical ranges:
# 100K vectors → nlist = 100-400
# 1M vectors → nlist = 1000-4000
# 10M vectors → nlist = 4000-10000
```

**nprobe** (clusters to search):
```python
# Trade-off: speed vs accuracy
index.nprobe = 1    # Fastest, ~80% recall
index.nprobe = 10   # Balanced, ~95% recall
index.nprobe = 100  # Slower, ~99% recall

# Rule: nprobe = nlist/10 for 95%+ recall
```

**Benchmarking:**
```python
import time

# Create index
index = faiss.IndexIVFFlat(quantizer, 384, nlist=1000)
index.train(embeddings)
index.add(embeddings)

# Test different nprobe values
for nprobe in [1, 5, 10, 20, 50]:
    index.nprobe = nprobe

    start = time.time()
    D, I = index.search(queries, k=10)
    elapsed = time.time() - start

    print(f"nprobe={nprobe}: {elapsed:.3f}s for {len(queries)} queries")
```

---

#### Q9: How do you measure recall for approximate indexes?

**Answer:**

Recall = % of true nearest neighbors found by approximate search.

```python
import faiss
import numpy as np

# Create ground truth (exact search)
index_flat = faiss.IndexFlatIP(384)
index_flat.add(embeddings)

D_true, I_true = index_flat.search(queries, k=10)

# Create approximate index
index_ivf = faiss.IndexIVFFlat(quantizer, 384, nlist=1000)
index_ivf.train(embeddings)
index_ivf.add(embeddings)
index_ivf.nprobe = 10

D_approx, I_approx = index_ivf.search(queries, k=10)

# Calculate recall@k
def calculate_recall(true_indices, approx_indices):
    """Calculate recall@k."""
    recalls = []
    for true, approx in zip(true_indices, approx_indices):
        true_set = set(true)
        approx_set = set(approx)
        recall = len(true_set & approx_set) / len(true_set)
        recalls.append(recall)
    return np.mean(recalls)

recall = calculate_recall(I_true, I_approx)
print(f"Recall@10: {recall:.3f}")  # e.g., 0.952 (95.2%)
```

**Typical recall values:**
- Recall@10 > 0.95 → Good for production
- Recall@10 > 0.99 → Excellent
- Recall@10 < 0.90 → Increase nprobe

---

#### Q10: How do you train an IVF index properly?

**Answer:**

Training learns the cluster centroids (k-means).

```python
import faiss
import numpy as np

dimension = 384
nlist = 1000

# Create IVF index
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Prepare training data
# Use representative sample if full dataset is too large
train_size = 100000  # At least 30 * nlist recommended
train_embeddings = embeddings[:train_size].astype('float32')
faiss.normalize_L2(train_embeddings)

# Train
print(f"Is trained: {index.is_trained}")  # False
index.train(train_embeddings)
print(f"Is trained: {index.is_trained}")  # True

# Now can add data
index.add(embeddings)

# Save trained index (preserves clustering)
faiss.write_index(index, "trained_ivf.faiss")

# Load later (no need to retrain)
loaded = faiss.read_index("trained_ivf.faiss")
print(f"Loaded is trained: {loaded.is_trained}")  # True
```

**Important:**
- Train on representative sample (at least 30 × nlist vectors)
- Only need to train once
- Adding more data doesn't require retraining
- Save trained index to avoid retraining

---

#### Q11: Can you update an IVF index incrementally?

**Answer:**

Yes, you can add vectors without retraining.

```python
import faiss

# Initial setup
index = faiss.IndexIVFFlat(quantizer, 384, nlist=1000)
index.train(initial_embeddings)
index.add(initial_embeddings)
print(f"Initial size: {index.ntotal}")

# Add new vectors incrementally
new_embeddings = np.random.randn(5000, 384).astype('float32')
faiss.normalize_L2(new_embeddings)

index.add(new_embeddings)  # No need to retrain!
print(f"After update: {index.ntotal}")

# Search works immediately
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)
D, I = index.search(query, k=10)
```

**When to retrain:**
- Data distribution changes significantly
- Initial training set was too small
- Adding 10x+ more data than original

**Practical pattern:**
```python
# Monthly: full reindex with retraining
if is_monthly_rebuild:
    index = faiss.IndexIVFFlat(quantizer, 384, nlist)
    index.train(all_embeddings)
    index.add(all_embeddings)
    faiss.write_index(index, "index.faiss")

# Daily: incremental updates
else:
    index = faiss.read_index("index.faiss")
    index.add(new_embeddings)  # Just add
    faiss.write_index(index, "index.faiss")
```

---

#### Q12: How do you handle IVF index that's too slow?

**Answer:**

**Optimization strategies:**

1. **Increase nlist, decrease nprobe:**
```python
# More clusters = fewer vectors per cluster = faster
index = faiss.IndexIVFFlat(quantizer, 384, nlist=4000)  # was 1000
index.nprobe = 5  # was 10
```

2. **Use IndexIVFPQ (Product Quantization):**
```python
# Compresses vectors, uses less memory, faster search
m = 64  # number of subquantizers
bits = 8  # bits per subquantizer
index = faiss.IndexIVFPQ(quantizer, 384, nlist=1000, m, bits)
index.train(embeddings)
index.add(embeddings)
# 10x less memory, 2-3x faster, small quality loss
```

3. **Use GPU:**
```python
# 10-100x faster on GPU
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexIVFFlat(quantizer, 384, nlist=1000)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
```

4. **Batch queries:**
```python
# Process multiple queries at once
queries = np.vstack([q1, q2, q3, ...])  # (N, 384)
D, I = index.search(queries, k=10)  # Faster than N individual searches
```

---

### HNSW Index (13-18)

#### Q13: What is HNSW and when to use it?

**Answer:**

**HNSW (Hierarchical Navigable Small World)** - Graph-based approximate search.

**How it works:**
- Builds a multi-layer graph where nodes are vectors
- Searches by navigating through layers (coarse to fine)
- Very fast, high recall, but uses more memory

```python
import faiss

# Create HNSW index
dimension = 384
M = 32  # Number of connections per layer (higher = better quality, more memory)

index = faiss.IndexHNSWFlat(dimension, M)

# Add data (no training needed!)
embeddings = np.random.randn(100000, 384).astype('float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Search parameter: efSearch (higher = better quality, slower)
index.hnsw.efSearch = 64  # Default is 16
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)
D, I = index.search(query, k=10)
```

**Comparison:**

| Index | Speed | Memory | Recall | Training |
|-------|-------|--------|--------|----------|
| Flat | Slow | Low | 100% | No |
| IVF | Fast | Low | 95-99% | Yes |
| HNSW | Very Fast | High | 95-99% | No |

**When to use HNSW:**
- Need lowest latency (< 1ms)
- Can afford 2-3x memory usage
- Don't want to manage training
- < 10M vectors

---

#### Q14: How do you tune HNSW parameters?

**Answer:**

**M** (connections per layer):
```python
# Construction parameter (set once)
M = 16  # Fast, lower quality
M = 32  # Balanced (recommended)
M = 64  # High quality, more memory

# Memory usage: ~M × dimension × 4 bytes per vector
# M=32, dim=384 → ~48 KB per vector
```

**efConstruction** (build quality):
```python
# Higher = better graph, slower construction
index = faiss.IndexHNSWFlat(384, M=32)
index.hnsw.efConstruction = 200  # Default is 40
# Set before adding vectors!
index.add(embeddings)
```

**efSearch** (search quality):
```python
# Higher = better recall, slower search
index.hnsw.efSearch = 16   # Fast, ~90% recall
index.hnsw.efSearch = 64   # Balanced, ~95% recall
index.hnsw.efSearch = 128  # High quality, ~99% recall
# Can change anytime
```

**Tuning workflow:**
```python
# 1. Build with high efConstruction
index = faiss.IndexHNSWFlat(384, M=32)
index.hnsw.efConstruction = 200
index.add(embeddings)

# 2. Test different efSearch values
for ef in [16, 32, 64, 128]:
    index.hnsw.efSearch = ef

    # Measure latency
    start = time.time()
    D, I = index.search(queries, k=10)
    latency = (time.time() - start) / len(queries) * 1000

    # Measure recall
    recall = calculate_recall(I_true, I)

    print(f"efSearch={ef}: {latency:.2f}ms, recall={recall:.3f}")

# Choose efSearch based on latency/quality trade-off
```

---

#### Q15: HNSW vs IVF - which to choose?

**Answer:**

**Use HNSW when:**
- Need lowest latency (< 1ms per query)
- Dataset fits in memory (< 10M vectors)
- Don't want to manage training
- Query latency more important than memory

**Use IVF when:**
- Large dataset (> 10M vectors)
- Memory constrained
- Slightly slower OK (1-10ms)
- Can use GPU for acceleration

**Practical comparison:**
```python
# Dataset: 1M vectors, 384 dims
corpus = np.random.randn(1_000_000, 384).astype('float32')
faiss.normalize_L2(corpus)
queries = np.random.randn(1000, 384).astype('float32')
faiss.normalize_L2(queries)

# HNSW
index_hnsw = faiss.IndexHNSWFlat(384, 32)
index_hnsw.hnsw.efConstruction = 200
index_hnsw.add(corpus)
index_hnsw.hnsw.efSearch = 64

# IVF
quantizer = faiss.IndexFlatIP(384)
index_ivf = faiss.IndexIVFFlat(quantizer, 384, nlist=1000)
index_ivf.train(corpus[:100000])
index_ivf.add(corpus)
index_ivf.nprobe = 10

# Benchmark
import time

start = time.time()
D_hnsw, I_hnsw = index_hnsw.search(queries, 10)
time_hnsw = (time.time() - start) / len(queries) * 1000

start = time.time()
D_ivf, I_ivf = index_ivf.search(queries, 10)
time_ivf = (time.time() - start) / len(queries) * 1000

print(f"HNSW: {time_hnsw:.2f}ms per query")  # ~0.5ms
print(f"IVF:  {time_ivf:.2f}ms per query")   # ~2ms

# Memory
import sys
print(f"HNSW memory: {sys.getsizeof(index_hnsw) / 1e9:.2f} GB")  # ~3 GB
print(f"IVF memory: {sys.getsizeof(index_ivf) / 1e9:.2f} GB")    # ~1.5 GB
```

**Decision tree:**
```
Dataset size?
├─ < 1M vectors → HNSW (if memory OK) or IVF
├─ 1-10M vectors → HNSW (if memory OK) or IVF
└─ > 10M vectors → IVF (or IVF+PQ for compression)

Latency requirement?
├─ < 1ms → HNSW
├─ 1-10ms → IVF
└─ > 10ms → Either

Memory budget?
├─ Limited → IVF
└─ Generous → HNSW
```

---

#### Q16: Can you combine HNSW with other indexes?

**Answer:**

Yes, you can use **HNSWPQ** for memory efficiency:

```python
# HNSW + Product Quantization
# Compresses vectors while keeping HNSW graph structure
m = 64  # subquantizers
bits = 8

# Create hybrid index
index = faiss.IndexHNSWPQ(384, m, M=32)
index.hnsw.efConstruction = 200

# Train PQ component
index.train(embeddings)

# Add data
index.add(embeddings)

# Search
index.hnsw.efSearch = 64
D, I = index.search(queries, k=10)

# Benefits:
# - 4-8x less memory than HNSW
# - Similar speed
# - Small recall loss (~2-5%)
```

**Or use HNSW as refiner:**
```python
# Two-stage: IVF (coarse) → HNSW (refine)
# Retrieve 100 candidates with IVF
index_ivf.nprobe = 5
D_coarse, I_coarse = index_ivf.search(queries, k=100)

# Refine top-100 with exact HNSW search
refined_results = []
for i, candidates in enumerate(I_coarse):
    candidate_vectors = corpus[candidates]
    # Compute exact similarities
    scores = np.dot(candidate_vectors, queries[i])
    top_10 = np.argsort(scores)[-10:][::-1]
    refined_results.append(candidates[top_10])
```

---

#### Q17: How do you handle HNSW index updates?

**Answer:**

HNSW doesn't support deletion, but you can add new vectors:

```python
# Create index
index = faiss.IndexHNSWFlat(384, 32)
index.add(initial_embeddings)
print(f"Initial: {index.ntotal}")

# Add more vectors
new_embeddings = np.random.randn(5000, 384).astype('float32')
faiss.normalize_L2(new_embeddings)
index.add(new_embeddings)
print(f"After update: {index.ntotal}")

# Search immediately available
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)
D, I = index.search(query, k=10)
```

**For deletions (workaround):**
```python
class UpdateableHNSW:
    def __init__(self, dimension, M=32):
        self.dimension = dimension
        self.M = M
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.id_to_offset = {}  # Map external ID → FAISS offset
        self.deleted = set()  # Track deleted IDs

    def add(self, vectors, ids):
        """Add vectors with IDs."""
        start_offset = self.index.ntotal
        self.index.add(vectors)

        for i, id in enumerate(ids):
            self.id_to_offset[id] = start_offset + i

    def delete(self, id):
        """Mark ID as deleted."""
        self.deleted.add(id)

    def search(self, query, k):
        """Search with deleted filtering."""
        # Search for k + len(deleted) to account for filtered results
        D, I = self.index.search(query, k + len(self.deleted))

        # Filter out deleted IDs
        valid_results = []
        for i in range(len(I[0])):
            offset = I[0][i]
            # Find ID for this offset
            id = next((id for id, off in self.id_to_offset.items()
                      if off == offset), None)
            if id not in self.deleted:
                valid_results.append((D[0][i], offset))
                if len(valid_results) == k:
                    break

        return valid_results

    def rebuild(self):
        """Rebuild index without deleted vectors."""
        # Get all non-deleted vectors
        valid_offsets = [off for id, off in self.id_to_offset.items()
                        if id not in self.deleted]

        # Rebuild
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        valid_vectors = vectors[valid_offsets]

        self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
        self.index.add(valid_vectors)
        self.deleted.clear()
```

**Best practice:**
- For frequent updates: use IVF instead
- For HNSW: rebuild index periodically (e.g., nightly)

---

#### Q18: How do you benchmark HNSW performance?

**Answer:**

Complete benchmark measuring latency and recall:

```python
import faiss
import numpy as np
import time

def benchmark_hnsw(corpus, queries, ground_truth_indices,
                   M_values=[16, 32, 64],
                   efSearch_values=[16, 32, 64, 128]):
    """
    Benchmark HNSW with different parameters.

    Args:
        corpus: (N, D) embeddings
        queries: (Q, D) query vectors
        ground_truth_indices: (Q, K) true nearest neighbors
        M_values: List of M values to test
        efSearch_values: List of efSearch values to test
    """
    dimension = corpus.shape[1]
    results = []

    for M in M_values:
        print(f"\nTesting M={M}")

        # Build index
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = 200

        build_start = time.time()
        index.add(corpus)
        build_time = time.time() - build_start

        for efSearch in efSearch_values:
            index.hnsw.efSearch = efSearch

            # Measure search time
            search_start = time.time()
            D, I = index.search(queries, k=ground_truth_indices.shape[1])
            search_time = (time.time() - search_start) / len(queries) * 1000

            # Calculate recall
            recalls = []
            for i in range(len(queries)):
                true_set = set(ground_truth_indices[i])
                pred_set = set(I[i])
                recall = len(true_set & pred_set) / len(true_set)
                recalls.append(recall)
            avg_recall = np.mean(recalls)

            results.append({
                'M': M,
                'efSearch': efSearch,
                'build_time': build_time,
                'latency_ms': search_time,
                'recall': avg_recall
            })

            print(f"  efSearch={efSearch}: "
                  f"latency={search_time:.2f}ms, "
                  f"recall={avg_recall:.4f}")

    return results

# Usage
corpus = np.random.randn(100000, 384).astype('float32')
faiss.normalize_L2(corpus)
queries = np.random.randn(100, 384).astype('float32')
faiss.normalize_L2(queries)

# Ground truth
index_flat = faiss.IndexFlatIP(384)
index_flat.add(corpus)
_, I_true = index_flat.search(queries, k=10)

# Benchmark
results = benchmark_hnsw(corpus, queries, I_true)

# Plot results
import matplotlib.pyplot as plt
for M in [16, 32, 64]:
    data = [r for r in results if r['M'] == M]
    latencies = [r['latency_ms'] for r in data]
    recalls = [r['recall'] for r in data]
    plt.plot(latencies, recalls, marker='o', label=f'M={M}')

plt.xlabel('Latency (ms)')
plt.ylabel('Recall@10')
plt.title('HNSW: Latency vs Recall')
plt.legend()
plt.grid(True)
plt.show()
```

---

### Production & Optimization (19-24)

#### Q19: How do you handle metadata filtering in FAISS?

**Answer:**

FAISS doesn't support native filtering. Use these patterns:

**Pattern 1: ID mapping (post-filtering)**
```python
import faiss
import numpy as np

class FAISSWithMetadata:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []  # List of dicts, one per vector

    def add(self, vectors, metadata_list):
        """Add vectors with metadata."""
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search(self, query, k, filter_fn=None):
        """
        Search with optional metadata filter.

        Args:
            query: (1, D) vector
            k: Number of results
            filter_fn: Function that takes metadata dict and returns bool
        """
        # Retrieve more results to account for filtering
        search_k = k * 10 if filter_fn else k
        D, I = self.index.search(query, search_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            meta = self.metadata[idx]

            # Apply filter
            if filter_fn and not filter_fn(meta):
                continue

            results.append({
                'index': int(idx),
                'score': float(score),
                'metadata': meta
            })

            if len(results) == k:
                break

        return results

# Usage
faiss_meta = FAISSWithMetadata(384)

# Add data
embeddings = np.random.randn(1000, 384).astype('float32')
faiss.normalize_L2(embeddings)
metadata = [
    {'category': 'billing', 'status': 'open'},
    {'category': 'technical', 'status': 'closed'},
    # ...
]
faiss_meta.add(embeddings, metadata)

# Search with filter
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)

results = faiss_meta.search(
    query,
    k=10,
    filter_fn=lambda m: m['category'] == 'billing' and m['status'] == 'open'
)

for r in results:
    print(f"Score: {r['score']:.3f}, Meta: {r['metadata']}")
```

**Pattern 2: Separate indexes per category**
```python
class MultiIndexFAISS:
    def __init__(self, dimension):
        self.dimension = dimension
        self.indexes = {}  # category → index

    def add(self, vectors, categories):
        """Add vectors to category-specific indexes."""
        for vector, category in zip(vectors, categories):
            if category not in self.indexes:
                self.indexes[category] = faiss.IndexFlatIP(self.dimension)

            self.indexes[category].add(vector.reshape(1, -1))

    def search(self, query, k, category=None):
        """Search in specific category or all."""
        if category:
            # Search single index
            return self.indexes[category].search(query, k)
        else:
            # Search all indexes, merge results
            all_results = []
            for cat, index in self.indexes.items():
                D, I = index.search(query, k)
                for score, idx in zip(D[0], I[0]):
                    all_results.append((float(score), cat, int(idx)))

            # Sort by score, return top-k
            all_results.sort(reverse=True)
            return all_results[:k]
```

---

#### Q20: How do you handle very large datasets that don't fit in memory?

**Answer:**

**Strategy 1: On-disk indexes (IndexIVF with invlists on disk)**
```python
# Not directly supported in faiss-cpu
# Use IndexIVFPQ for compression first
m = 64
bits = 8
index = faiss.IndexIVFPQ(quantizer, 384, nlist=10000, m, bits)
index.train(sample)
index.add(vectors)
# Much smaller memory footprint
```

**Strategy 2: Sharding across multiple indexes**
```python
class ShardedFAISS:
    def __init__(self, dimension, n_shards=10):
        self.dimension = dimension
        self.n_shards = n_shards
        self.shards = [
            faiss.IndexFlatIP(dimension) for _ in range(n_shards)
        ]
        self.shard_offsets = [0]

    def add(self, vectors):
        """Distribute vectors across shards."""
        shard_size = len(vectors) // self.n_shards
        for i, shard in enumerate(self.shards):
            start = i * shard_size
            end = start + shard_size if i < self.n_shards - 1 else len(vectors)
            shard.add(vectors[start:end])
            self.shard_offsets.append(self.shard_offsets[-1] + end - start)

    def search(self, query, k):
        """Search all shards, merge results."""
        all_scores = []
        all_indices = []

        for shard_idx, shard in enumerate(self.shards):
            D, I = shard.search(query, k)
            # Adjust indices to global offset
            global_indices = I[0] + self.shard_offsets[shard_idx]
            all_scores.extend(D[0])
            all_indices.extend(global_indices)

        # Sort and take top-k
        combined = list(zip(all_scores, all_indices))
        combined.sort(reverse=True)
        top_k = combined[:k]

        scores = np.array([s for s, _ in top_k])
        indices = np.array([i for _, i in top_k])

        return scores.reshape(1, -1), indices.reshape(1, -1)

# Usage
sharded = ShardedFAISS(384, n_shards=10)
# Each shard handles 1/10 of data
```

**Strategy 3: Use external vector database**
```python
# For truly massive datasets (100M+ vectors):
# - ChromaDB (good for <10M)
# - Pinecone (managed, scalable)
# - Weaviate (self-hosted, scalable)
# - Milvus (enterprise-grade)
```

---

#### Q21: How do you optimize FAISS for low latency?

**Answer:**

**Optimization checklist:**

1. **Choose right index type:**
```python
# < 1M vectors, need <1ms
index = faiss.IndexHNSWFlat(384, 32)
index.hnsw.efSearch = 32  # Lower for speed

# 1-10M vectors, need <5ms
index = faiss.IndexIVFFlat(quantizer, 384, nlist=4000)
index.nprobe = 5  # Lower for speed
```

2. **Use GPU:**
```python
# 10-100x faster for batch queries
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
```

3. **Batch queries:**
```python
# Batch processing is more efficient
# Single query: 1ms each → 100 queries = 100ms
# Batch: 20ms for 100 queries
queries_batch = np.vstack(query_list)  # (100, 384)
D, I = index.search(queries_batch, k=10)
```

4. **Pre-normalize embeddings:**
```python
# Normalize once during indexing, not at search time
faiss.normalize_L2(corpus)
index.add(corpus)

# At search time, just normalize query
faiss.normalize_L2(query)
D, I = index.search(query, k=10)
```

5. **Use float32, not float64:**
```python
# Always use float32
embeddings = embeddings.astype('float32')
```

6. **Consider compression:**
```python
# IndexIVFPQ: 4-8x less memory, 2-3x faster
index = faiss.IndexIVFPQ(quantizer, 384, nlist=1000, m=64, nbits=8)
```

**Latency benchmark:**
```python
import time

def benchmark_latency(index, queries, k=10, warmup=10):
    """Measure search latency."""
    # Warmup
    for _ in range(warmup):
        index.search(queries[:1], k)

    # Measure
    latencies = []
    for query in queries:
        start = time.perf_counter()
        index.search(query.reshape(1, -1), k)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

    return {
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'mean': np.mean(latencies)
    }

results = benchmark_latency(index, queries)
print(f"P50: {results['p50']:.2f}ms")
print(f"P95: {results['p95']:.2f}ms")
print(f"P99: {results['p99']:.2f}ms")
```

---

#### Q22: How do you monitor FAISS index health in production?

**Answer:**

**Key metrics to track:**

```python
import time
import numpy as np
from collections import deque

class FAISSMonitor:
    def __init__(self, index, window_size=1000):
        self.index = index
        self.window_size = window_size

        # Metrics
        self.query_latencies = deque(maxlen=window_size)
        self.query_count = 0
        self.error_count = 0

    def search(self, query, k):
        """Monitored search."""
        try:
            start = time.perf_counter()
            D, I = self.index.search(query, k)
            latency = (time.perf_counter() - start) * 1000

            self.query_latencies.append(latency)
            self.query_count += 1

            return D, I
        except Exception as e:
            self.error_count += 1
            raise

    def get_metrics(self):
        """Get current metrics."""
        if not self.query_latencies:
            return {}

        latencies = list(self.query_latencies)
        return {
            'total_queries': self.query_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / self.query_count if self.query_count > 0 else 0,
            'index_size': self.index.ntotal,
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'latency_mean': np.mean(latencies),
        }

    def print_metrics(self):
        """Print metrics to console."""
        metrics = self.get_metrics()
        print("=== FAISS Index Metrics ===")
        print(f"Index size: {metrics.get('index_size', 0):,} vectors")
        print(f"Total queries: {metrics.get('total_queries', 0):,}")
        print(f"Error rate: {metrics.get('error_rate', 0):.4f}")
        print(f"Latency P50: {metrics.get('latency_p50', 0):.2f}ms")
        print(f"Latency P95: {metrics.get('latency_p95', 0):.2f}ms")
        print(f"Latency P99: {metrics.get('latency_p99', 0):.2f}ms")

# Usage
monitor = FAISSMonitor(index)

for query in queries:
    D, I = monitor.search(query, k=10)

monitor.print_metrics()
```

**Alerts to set up:**
1. Latency P99 > threshold (e.g., 100ms)
2. Error rate > 0.01
3. Index size changes unexpectedly
4. Query rate drops significantly

---

#### Q23: How do you test FAISS indexes before production deployment?

**Answer:**

**Testing checklist:**

```python
import faiss
import numpy as np

def test_faiss_index(index, test_vectors, test_queries, k=10):
    """
    Comprehensive FAISS index testing.

    Returns dict with test results.
    """
    results = {}

    # Test 1: Add vectors
    try:
        index.add(test_vectors)
        results['add_test'] = 'PASS'
        results['index_size'] = index.ntotal
    except Exception as e:
        results['add_test'] = f'FAIL: {e}'
        return results

    # Test 2: Basic search
    try:
        D, I = index.search(test_queries, k)
        results['search_test'] = 'PASS'
        results['search_shape'] = I.shape
    except Exception as e:
        results['search_test'] = f'FAIL: {e}'
        return results

    # Test 3: Verify results
    try:
        assert I.shape == (len(test_queries), k), "Wrong output shape"
        assert D.shape == (len(test_queries), k), "Wrong distance shape"
        assert not np.any(I < 0), "Negative indices"
        assert not np.any(I >= index.ntotal), "Out of bounds indices"
        results['validation_test'] = 'PASS'
    except AssertionError as e:
        results['validation_test'] = f'FAIL: {e}'

    # Test 4: Save/load
    try:
        faiss.write_index(index, '/tmp/test_index.faiss')
        loaded = faiss.read_index('/tmp/test_index.faiss')
        D2, I2 = loaded.search(test_queries, k)
        assert np.array_equal(I, I2), "Results differ after load"
        results['persistence_test'] = 'PASS'
    except Exception as e:
        results['persistence_test'] = f'FAIL: {e}'

    # Test 5: Latency
    try:
        import time
        latencies = []
        for q in test_queries:
            start = time.perf_counter()
            index.search(q.reshape(1, -1), k)
            latencies.append((time.perf_counter() - start) * 1000)

        results['latency_mean'] = np.mean(latencies)
        results['latency_p95'] = np.percentile(latencies, 95)
        results['latency_test'] = 'PASS'
    except Exception as e:
        results['latency_test'] = f'FAIL: {e}'

    # Test 6: Recall (if ground truth available)
    try:
        # Create ground truth with flat index
        ground_truth_index = faiss.IndexFlatIP(test_vectors.shape[1])
        ground_truth_index.add(test_vectors)
        D_true, I_true = ground_truth_index.search(test_queries, k)

        # Calculate recall
        recalls = []
        for i in range(len(test_queries)):
            true_set = set(I_true[i])
            pred_set = set(I[i])
            recall = len(true_set & pred_set) / k
            recalls.append(recall)

        results['recall_mean'] = np.mean(recalls)
        results['recall_test'] = 'PASS' if results['recall_mean'] > 0.9 else 'WARN'
    except Exception as e:
        results['recall_test'] = f'FAIL: {e}'

    return results

# Run tests
test_vectors = np.random.randn(10000, 384).astype('float32')
test_queries = np.random.randn(100, 384).astype('float32')
faiss.normalize_L2(test_vectors)
faiss.normalize_L2(test_queries)

index = faiss.IndexFlatIP(384)
results = test_faiss_index(index, test_vectors, test_queries)

# Print results
print("=== FAISS Index Test Results ===")
for key, value in results.items():
    print(f"{key}: {value}")
```

---

#### Q24: How do you migrate from NumPy to FAISS in existing code?

**Answer:**

**Step-by-step migration:**

**Before (NumPy):**
```python
import numpy as np

class NumPySearch:
    def __init__(self):
        self.corpus_embeddings = None

    def index(self, embeddings):
        """Index embeddings."""
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = embeddings / norms

    def search(self, query, k=10):
        """Search for top-k similar vectors."""
        # Normalize query
        query_norm = query / np.linalg.norm(query)

        # Compute similarities
        similarities = np.dot(self.corpus_embeddings, query_norm)

        # Get top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_scores = similarities[top_indices]

        return top_scores, top_indices
```

**After (FAISS) - Drop-in replacement:**
```python
import faiss
import numpy as np

class FAISSSearch:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def index(self, embeddings):
        """Index embeddings."""
        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32')

        # Normalize
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

    def search(self, query, k=10):
        """Search for top-k similar vectors."""
        # Ensure 2D and float32
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.dtype != np.float32:
            query = query.astype('float32')

        # Normalize
        faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, k)

        # Return in same format as NumPy version
        return distances[0], indices[0]

# Usage - same API!
searcher = FAISSSearch(dimension=384)
searcher.index(corpus_embeddings)
scores, indices = searcher.search(query_embedding, k=10)
```

**Gradual migration with compatibility layer:**
```python
class UnifiedSearch:
    """Supports both NumPy and FAISS backends."""

    def __init__(self, dimension, backend='faiss', threshold=10000):
        self.dimension = dimension
        self.backend = backend
        self.threshold = threshold
        self.numpy_search = None
        self.faiss_search = None

    def index(self, embeddings):
        # Auto-select backend based on size
        if self.backend == 'auto':
            if len(embeddings) < self.threshold:
                self.backend = 'numpy'
            else:
                self.backend = 'faiss'

        if self.backend == 'numpy':
            self.numpy_search = NumPySearch()
            self.numpy_search.index(embeddings)
        else:
            self.faiss_search = FAISSSearch(self.dimension)
            self.faiss_search.index(embeddings)

    def search(self, query, k=10):
        if self.backend == 'numpy':
            return self.numpy_search.search(query, k)
        else:
            return self.faiss_search.search(query, k)
```

---

### Advanced Topics (25-30)

#### Q25: How do you use FAISS with GPU?

**Answer:**

**GPU acceleration provides 10-100x speedup for batch queries.**

```python
import faiss
import numpy as np

# Create CPU index
dimension = 384
index_cpu = faiss.IndexFlatIP(dimension)

# Add data on CPU
embeddings = np.random.randn(100000, 384).astype('float32')
faiss.normalize_L2(embeddings)
index_cpu.add(embeddings)

# Move to GPU
res = faiss.StandardGpuResources()  # GPU resource manager
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0

# Search on GPU (same API)
query = np.random.randn(1, 384).astype('float32')
faiss.normalize_L2(query)
D, I = index_gpu.search(query, k=10)

# Move back to CPU if needed
index_cpu_2 = faiss.index_gpu_to_cpu(index_gpu)
```

**GPU works with IVF too:**
```python
# Train on CPU
quantizer = faiss.IndexFlatIP(384)
index_cpu = faiss.IndexIVFFlat(quantizer, 384, nlist=1000)
index_cpu.train(embeddings)
index_cpu.add(embeddings)

# Move to GPU
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index_gpu.nprobe = 10

# Search
D, I = index_gpu.search(queries, k=10)  # Much faster on GPU
```

**Multi-GPU:**
```python
# Use multiple GPUs
ngpus = faiss.get_num_gpus()
print(f"Available GPUs: {ngpus}")

# Replicate index across GPUs
cpu_index = faiss.IndexFlatIP(384)
cpu_index.add(embeddings)

gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

# Searches automatically distributed across GPUs
D, I = gpu_index.search(large_query_batch, k=10)
```

---

#### Q26: What is Product Quantization (PQ) and when to use it?

**Answer:**

**Product Quantization** compresses vectors for memory efficiency.

**How it works:**
1. Split vector into `m` subvectors
2. Quantize each subvector to `2^bits` centroids
3. Store centroid IDs instead of full vectors

```python
# Original: 384 floats × 4 bytes = 1536 bytes/vector
# PQ: 64 bytes (m=64, bits=8) → 24x compression!

import faiss

dimension = 384
m = 64  # Number of subquantizers
bits = 8  # Bits per subquantizer

# Create PQ index
index = faiss.IndexPQ(dimension, m, bits)

# Train (learn centroids)
index.train(embeddings)

# Add compressed vectors
index.add(embeddings)

# Search (automatic decompression)
D, I = index.search(query, k=10)
```

**IndexIVFPQ combines IVF + PQ:**
```python
# Best of both worlds: fast search + compression
quantizer = faiss.IndexFlatIP(384)
index = faiss.IndexIVFPQ(quantizer, 384, nlist=1000, m=64, nbits=8)

index.train(embeddings)
index.add(embeddings)
index.nprobe = 10

D, I = index.search(query, k=10)

# Memory usage: ~64 bytes per vector (vs 1536 bytes)
# Search speed: Similar to IVF
# Quality: Small loss (~2-5% recall drop)
```

**When to use PQ:**
- Dataset too large for memory
- Can tolerate small quality loss
- Need to scale to 100M+ vectors

---

#### Q27: How do you handle vector updates (add/delete)?

**Answer:**

**Adding is easy, deleting is hard in FAISS.**

**Adding new vectors:**
```python
# All indexes support adding
index.add(new_embeddings)  # Appends to index
```

**Deleting vectors - workarounds:**

**Method 1: IDMap wrapper (supports removal)**
```python
# Wrap index with IDMap for ID management
index_base = faiss.IndexFlatIP(384)
index = faiss.IndexIDMap(index_base)

# Add with IDs
ids = np.array([1, 2, 3, 4, 5], dtype='int64')
embeddings = np.random.randn(5, 384).astype('float32')
faiss.normalize_L2(embeddings)
index.add_with_ids(embeddings, ids)

# Remove by ID
index.remove_ids(np.array([2, 4], dtype='int64'))

print(f"Remaining: {index.ntotal} vectors")
```

**Method 2: Rebuild periodically**
```python
class RebuildableIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.embeddings = []
        self.ids = []
        self.deleted_ids = set()
        self.index = None
        self._rebuild()

    def add(self, embedding, id):
        """Add single vector."""
        self.embeddings.append(embedding)
        self.ids.append(id)
        self.index.add(embedding.reshape(1, -1))

    def delete(self, id):
        """Mark for deletion."""
        self.deleted_ids.add(id)

        # Rebuild if too many deleted
        if len(self.deleted_ids) > len(self.ids) * 0.1:
            self._rebuild()

    def _rebuild(self):
        """Rebuild index without deleted vectors."""
        # Filter out deleted
        valid_embeddings = []
        valid_ids = []
        for emb, id in zip(self.embeddings, self.ids):
            if id not in self.deleted_ids:
                valid_embeddings.append(emb)
                valid_ids.append(id)

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dimension)
        if valid_embeddings:
            vectors = np.vstack(valid_embeddings)
            faiss.normalize_L2(vectors)
            self.index.add(vectors)

        self.embeddings = valid_embeddings
        self.ids = valid_ids
        self.deleted_ids.clear()

    def search(self, query, k):
        return self.index.search(query, k)
```

**Method 3: Use external database**
```python
# For frequent updates, use vector DB instead:
# - ChromaDB: supports updates/deletes
# - Pinecone: managed, supports updates
# - Weaviate: supports CRUD operations
```

---

#### Q28: How do you combine FAISS with sentence-transformers?

**Answer:**

**End-to-end semantic search pipeline:**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SemanticSearchEngine:
    """
    Complete semantic search with embeddings + FAISS.
    """

    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        index_type='flat'
    ):
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Create FAISS index
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist=100
            )
        elif index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dimension, M=32)

        self.documents = []
        self.is_trained = False

    def index_documents(self, documents):
        """Index a list of text documents."""
        self.documents = documents

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )

        # Ensure float32
        embeddings = embeddings.astype('float32')

        # Train if needed (IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Training index...")
            self.index.train(embeddings)
            self.is_trained = True

        # Add to index
        print("Adding to index...")
        self.index.add(embeddings)
        print(f"Indexed {self.index.ntotal} documents")

    def search(self, query, k=5, return_scores=True):
        """
        Search for similar documents.

        Args:
            query: Text query
            k: Number of results
            return_scores: Include similarity scores

        Returns:
            List of (document, score) tuples or just documents
        """
        # Generate query embedding
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        # Ensure 2D and float32
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Format results
        results = []
        for score, idx in zip(distances[0], indices[0]):
            doc = self.documents[idx]
            if return_scores:
                results.append((doc, float(score)))
            else:
                results.append(doc)

        return results

    def save(self, path):
        """Save index and documents."""
        import json

        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")

        # Save documents and metadata
        with open(f"{path}/documents.json", 'w') as f:
            json.dump({
                'documents': self.documents,
                'model_name': self.model.name_or_path,
                'dimension': self.dimension
            }, f)

    @classmethod
    def load(cls, path):
        """Load saved index."""
        import json

        # Load metadata
        with open(f"{path}/documents.json", 'r') as f:
            data = json.load(f)

        # Create instance
        instance = cls(model_name=data['model_name'])

        # Load FAISS index
        instance.index = faiss.read_index(f"{path}/index.faiss")
        instance.documents = data['documents']

        return instance

# Usage
engine = SemanticSearchEngine(index_type='ivf')

documents = [
    "How to reset password",
    "Cannot login after password change",
    "Invoice payment failed",
    "Unable to download invoice",
    "Account locked after failed attempts",
]

engine.index_documents(documents)

# Search
results = engine.search("password reset issues", k=3)
for doc, score in results:
    print(f"{score:.3f}: {doc}")

# Save
engine.save("./search_engine")

# Load later
loaded_engine = SemanticSearchEngine.load("./search_engine")
```

---

#### Q29: How do you debug slow FAISS searches?

**Answer:**

**Debugging workflow:**

```python
import faiss
import numpy as np
import time

def debug_faiss_performance(index, queries, k=10):
    """
    Comprehensive FAISS performance debugging.
    """
    print("=== FAISS Performance Debug ===\n")

    # 1. Index info
    print(f"Index type: {type(index).__name__}")
    print(f"Index size: {index.ntotal:,} vectors")
    print(f"Dimension: {index.d}")

    # 2. Check index parameters
    if isinstance(index, faiss.IndexIVFFlat):
        print(f"nlist: {index.nlist}")
        print(f"nprobe: {index.nprobe}")
        quantizer_size = index.quantizer.ntotal
        print(f"Quantizer size: {quantizer_size}")
        print(f"Is trained: {index.is_trained}")

    if isinstance(index, faiss.IndexHNSWFlat):
        print(f"M: {index.hnsw.M}")
        print(f"efSearch: {index.hnsw.efSearch}")

    # 3. Measure single query latency
    print("\n--- Single Query Latency ---")
    single_latencies = []
    for i in range(min(100, len(queries))):
        start = time.perf_counter()
        index.search(queries[i:i+1], k)
        latency = (time.perf_counter() - start) * 1000
        single_latencies.append(latency)

    print(f"Mean: {np.mean(single_latencies):.2f}ms")
    print(f"P50: {np.percentile(single_latencies, 50):.2f}ms")
    print(f"P95: {np.percentile(single_latencies, 95):.2f}ms")
    print(f"P99: {np.percentile(single_latencies, 99):.2f}ms")

    # 4. Measure batch latency
    print("\n--- Batch Query Latency ---")
    batch_sizes = [1, 10, 100, min(1000, len(queries))]
    for batch_size in batch_sizes:
        batch = queries[:batch_size]
        start = time.perf_counter()
        index.search(batch, k)
        total_time = (time.perf_counter() - start) * 1000
        per_query = total_time / batch_size
        print(f"Batch={batch_size}: {per_query:.2f}ms per query")

    # 5. Memory usage
    print("\n--- Memory Usage ---")
    import sys
    memory_mb = sys.getsizeof(index) / (1024 * 1024)
    print(f"Index memory: {memory_mb:.2f} MB")

    # 6. Suggestions
    print("\n--- Optimization Suggestions ---")

    if isinstance(index, faiss.IndexFlatIP) or isinstance(index, faiss.IndexFlatL2):
        if index.ntotal > 100000:
            print("⚠️  Using Flat index with >100K vectors")
            print("   → Consider IndexIVFFlat or IndexHNSWFlat")

    if isinstance(index, faiss.IndexIVFFlat):
        if index.nprobe > index.nlist / 10:
            print("⚠️  nprobe is high relative to nlist")
            print(f"   → Current: nprobe={index.nprobe}, nlist={index.nlist}")
            print(f"   → Try: nprobe={index.nlist // 10}")

        avg_cluster_size = index.ntotal / index.nlist
        if avg_cluster_size > 1000:
            print("⚠️  Large cluster size")
            print(f"   → Current: {avg_cluster_size:.0f} vectors per cluster")
            print(f"   → Try: nlist={int(np.sqrt(index.ntotal) * 4)}")

    if np.mean(single_latencies) > 10:
        print("⚠️  High latency detected")
        print("   → Consider GPU acceleration")
        print("   → Batch queries when possible")
        print("   → Use IndexIVFPQ for compression")

# Usage
debug_faiss_performance(index, test_queries)
```

**Common issues and fixes:**

```python
# Issue 1: Flat index too slow
# Fix: Switch to IVF or HNSW
index = faiss.IndexIVFFlat(quantizer, 384, nlist=1000)

# Issue 2: IVF too slow
# Fix: Reduce nprobe
index.nprobe = 5  # was 20

# Issue 3: Too many vectors per cluster
# Fix: Increase nlist
nlist = int(4 * np.sqrt(index.ntotal))

# Issue 4: High memory usage
# Fix: Use Product Quantization
index = faiss.IndexIVFPQ(quantizer, 384, nlist=1000, m=64, nbits=8)
```

---

#### Q30: How do you validate FAISS results are correct?

**Answer:**

**Validation strategies:**

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def validate_faiss_results(
    model,
    documents,
    index,
    test_queries,
    k=10
):
    """
    Validate FAISS results against ground truth.
    """
    print("=== FAISS Results Validation ===\n")

    # 1. Generate ground truth with exact search
    print("Generating ground truth...")
    embeddings = model.encode(documents, normalize_embeddings=True)
    embeddings = embeddings.astype('float32')

    ground_truth_index = faiss.IndexFlatIP(embeddings.shape[1])
    ground_truth_index.add(embeddings)

    # 2. Compare results
    validation_results = []

    for query_text in test_queries:
        query_emb = model.encode(query_text, normalize_embeddings=True)
        query_emb = query_emb.reshape(1, -1).astype('float32')

        # Ground truth
        D_true, I_true = ground_truth_index.search(query_emb, k)

        # FAISS result
        D_faiss, I_faiss = index.search(query_emb, k)

        # Calculate recall
        true_set = set(I_true[0])
        faiss_set = set(I_faiss[0])
        recall = len(true_set & faiss_set) / k

        # Check score ordering
        is_sorted = np.all(D_faiss[0][:-1] >= D_faiss[0][1:])

        # Compare top result
        top_match = I_true[0][0] == I_faiss[0][0]

        validation_results.append({
            'query': query_text,
            'recall': recall,
            'top_match': top_match,
            'sorted': is_sorted,
            'true_top': documents[I_true[0][0]],
            'faiss_top': documents[I_faiss[0][0]],
            'true_score': float(D_true[0][0]),
            'faiss_score': float(D_faiss[0][0])
        })

    # 3. Print results
    avg_recall = np.mean([r['recall'] for r in validation_results])
    top_match_rate = np.mean([r['top_match'] for r in validation_results])

    print(f"Average Recall@{k}: {avg_recall:.4f}")
    print(f"Top Result Match Rate: {top_match_rate:.4f}")

    print("\n--- Sample Results ---")
    for i, result in enumerate(validation_results[:3]):
        print(f"\nQuery {i+1}: {result['query']}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  Top Match: {'✓' if result['top_match'] else '✗'}")
        print(f"  Ground Truth: {result['true_top'][:50]}...")
        print(f"  FAISS Result: {result['faiss_top'][:50]}...")

    # 4. Validation checks
    print("\n--- Validation Checks ---")

    if avg_recall < 0.9:
        print(f"❌ Low recall: {avg_recall:.4f} < 0.9")
        print("   → Increase nprobe (IVF) or efSearch (HNSW)")
    else:
        print(f"✅ Good recall: {avg_recall:.4f}")

    if top_match_rate < 0.8:
        print(f"⚠️  Low top match rate: {top_match_rate:.4f}")
    else:
        print(f"✅ Good top match rate: {top_match_rate:.4f}")

    all_sorted = all(r['sorted'] for r in validation_results)
    if not all_sorted:
        print("❌ Results not sorted by score")
    else:
        print("✅ Results properly sorted")

    return validation_results

# Usage
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = ["doc1", "doc2", "doc3", ...]
test_queries = ["query1", "query2", "query3"]

# Build index
embeddings = model.encode(documents, normalize_embeddings=True).astype('float32')
index = faiss.IndexIVFFlat(faiss.IndexFlatIP(384), 384, nlist=100)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10

# Validate
results = validate_faiss_results(model, documents, index, test_queries)
```

---

## Summary

FAISS is essential for production vector search. Key takeaways:

1. **Start simple**: IndexFlatIP for < 100K vectors
2. **Scale up**: IndexIVFFlat for 100K-10M vectors
3. **Optimize**: IndexHNSWFlat for lowest latency
4. **Compress**: IndexIVFPQ for memory efficiency
5. **Always measure**: Benchmark latency and recall
6. **Test thoroughly**: Validate before production

Next: Combine FAISS with RAG in Module 6!
