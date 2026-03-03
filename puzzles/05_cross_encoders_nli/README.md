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

## Practice Questions

See [QUESTIONS.md](./QUESTIONS.md) for 30 deep-dive questions with detailed answers covering:
- Architecture & Design (Q1-Q10)
- Implementation & Coding (Q11-Q20)
- Debugging & Troubleshooting (Q21-Q25)
- Trade-offs & Decisions (Q26-Q30)

---

## Additional Resources

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [Cross-Encoder Documentation](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
