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

## Practice Questions

See [QUESTIONS.md](./QUESTIONS.md) for 30 deep-dive questions with detailed answers covering:
- Basics & Setup (Q1-Q6)
- IVF Indexes (Q7-Q12)
- HNSW Index (Q13-Q18)
- Production & Optimization (Q19-Q24)
- Advanced Topics (Q25-Q30)

---

## Summary

FAISS is essential for production vector search. Key takeaways:

1. **Start simple**: IndexFlatIP for < 100K vectors
2. **Scale up**: IndexIVFFlat for 100K-10M vectors
3. **Optimize**: IndexHNSWFlat for lowest latency
4. **Compress**: IndexIVFPQ for memory efficiency
5. **Always measure**: Benchmark latency and recall
6. **Test thoroughly**: Validate before production

Next: Combine FAISS with cross-encoders in Module 4!
