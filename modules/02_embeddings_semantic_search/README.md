# Module 2: Text Embeddings & Semantic Search

> Foundation for semantic search, RAG, and similarity-based applications

## Why This Matters

Embeddings are the foundation of modern NLP. They convert text into numerical vectors that capture semantic meaning, enabling:
- Semantic search (find similar documents by meaning, not keywords)
- Clustering and classification
- RAG systems (retrieval step)
- Duplicate detection

## Key Concepts

### What are Embeddings?
- Dense vector representations of text (e.g., 384 or 768 dimensions)
- Similar texts have similar vectors (close in vector space)
- Capture semantic meaning, not just keywords

### Sentence Transformers
- Library for state-of-the-art sentence embeddings
- Pre-trained models: `all-MiniLM-L6-v2`, `gte-base`, `e5-base`
- Simple API: `model.encode(sentences)`

### Similarity Metrics
| Metric | Formula | Range | Use Case |
|--------|---------|-------|----------|
| Cosine Similarity | cos(A,B) = A·B / (‖A‖‖B‖) | [-1, 1] | Most common, direction-based |
| Dot Product | A·B | (-∞, +∞) | When vectors are normalized |
| Euclidean Distance | ‖A-B‖ | [0, +∞) | When magnitude matters |

### Normalization
- L2 normalization: vector / ‖vector‖
- After normalization: cosine similarity = dot product
- Most models output normalized embeddings

## Documentation & Resources

### Official Docs:
- [Sentence Transformers](https://www.sbert.net/)
- [SBERT Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

### Leaderboards:
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb)

### Papers:
- [Sentence-BERT (2019)](https://arxiv.org/abs/1908.10084)
- [E5 Embeddings](https://arxiv.org/abs/2212.03533)
- [GTE Embeddings](https://arxiv.org/abs/2308.03281)

### Tutorials:
- [SBERT Training Overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html)
- [Pinecone - Embeddings Guide](https://www.pinecone.io/learn/sentence-embeddings/)

## Self-Assessment Checklist

- [ ] I can explain what embeddings are and why they're useful
- [ ] I can use sentence-transformers to encode text
- [ ] I can calculate cosine similarity between vectors
- [ ] I understand when to use cosine vs dot product vs euclidean
- [ ] I can find similar documents using embeddings
- [ ] I understand normalization and its effects

---

## Practice Questions

See [QUESTIONS.md](./QUESTIONS.md) for 30 deep-dive questions with detailed answers covering:
- Fundamentals (Q1-Q10): What are embeddings, similarity metrics, model selection
- Implementation (Q11-Q20): Semantic search, clustering, caching, benchmarking
- Debugging & Troubleshooting (Q21-Q25): Performance issues, memory optimization
- Trade-offs & Decisions (Q26-Q30): OpenAI vs open-source, fine-tuning, chunking strategies
