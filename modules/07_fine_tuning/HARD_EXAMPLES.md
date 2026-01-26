# Hard Training Examples Documentation

## Overview

Generated 500 hard triplet and pair examples for challenging fine-tuning.

## What Makes Them "Hard"?

**Hard negatives** are semantically similar to the query but NOT relevant:

```python
# EASY negative (obvious difference)
Query: "How to reset password?"
Negative: "Professional plan costs $99/month"  # Completely unrelated

# HARD negative (subtle difference)
Query: "How to reset password?"
Negative: "Password recovery from account settings"  # Related but not the same
```

Hard negatives force the model to learn fine-grained distinctions.

## Data Structure

### Triplets: `training_triplets_hard.json` (500 examples)
```json
{
  "anchor": "What is transfer learning?",
  "positive": "Reuse pretrained model weights on new domain-specific task",
  "negative": "Domain adaptation adjusts model from source to target distribution"
}
```

### Pairs: `training_pairs_hard.json` (500 examples)
```json
{
  "query": "How to authenticate API?",
  "positive": "Add Bearer token to Authorization header for each request",
  "negative": "Include credentials in Basic Auth header with base64 encoding"
}
```

## Domains Covered

1. **ML/NLP** (40 queries)
   - Transfer learning, BERT, transformers, attention mechanisms
   - Hard negatives: Similar ML concepts (GPT vs BERT, CNN vs RNN)

2. **API/Auth** (30 queries)
   - OAuth, JWT, API keys, rate limiting
   - Hard negatives: Related auth methods (Basic Auth vs Bearer token)

3. **Database** (30 queries)
   - SQL optimization, indexes, ACID, sharding
   - Hard negatives: Related DB concepts (clustered vs non-clustered index)

4. **DevOps** (30 queries)
   - Docker, Kubernetes, CI/CD, monitoring
   - Hard negatives: Related tools (Docker vs VM, Swarm vs K8s)

5. **Python** (30 queries)
   - List comprehensions, decorators, async/await
   - Hard negatives: Similar Python features (generator vs list)

## Generation Strategy

### Same-Domain Hard Negatives
Most hard negatives come from the SAME domain:
- Query about BERT → Negative about Transformers
- Query about OAuth → Negative about JWT
- Semantic similarity: 0.6-0.8 (challenging but learnable)

### Cross-Domain Hard Negatives
Some negatives from different domains for variety:
- ML query → DevOps negative
- Forces model to learn domain boundaries

## Training Benefits

1. **Better generalization**: Model learns subtle differences
2. **Stronger separation**: Larger margins between relevant and irrelevant
3. **Realistic**: Production queries often have similar distractors
4. **Faster convergence**: Informative gradients from hard examples

## Usage

### 02_triplet_loss.ipynb
```python
with open('../fixtures/input/training_triplets_hard.json', 'r') as f:
    triplets = json.load(f)
```

### 03_contrastive_loss.ipynb
```python
with open('../fixtures/input/training_pairs_hard.json', 'r') as f:
    pairs = json.load(f)
```

### 04_transformers_trainer.ipynb
```python
with open('../fixtures/input/training_triplets_hard.json', 'r') as f:
    triplets = json.load(f)
```

## Statistics

- Total triplets: 500
- Total pairs: 500 (creates 1000 examples: 500 positive + 500 negative)
- Domains: 5
- Avg query length: 6 words
- Avg positive length: 12 words
- Avg negative length: 11 words
- Hard negative similarity: 0.6-0.8 (cosine similarity with baseline model)

## Expected Results

With hard examples, expect:
- **Initial accuracy**: 60-70% (harder than easy negatives)
- **Final accuracy**: 85-95% (after training)
- **Margin improvement**: +0.2 to +0.4
- **Training time**: 3-4 epochs (vs 10+ for easy examples)

## Hard Negative Mining

For production, implement automatic hard negative mining:

```python
# Find hard negatives (similar but not relevant)
def mine_hard_negatives(query, corpus, model, k=10):
    query_emb = model.encode(query)
    corpus_embs = model.encode(corpus)

    # Get top-k most similar
    similarities = cosine_similarity([query_emb], corpus_embs)[0]
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # Filter out actual positives
    hard_negatives = [corpus[i] for i in top_k_indices
                      if i not in positive_indices]

    return hard_negatives[:k]
```

## Comparison: Easy vs Hard Negatives

| Metric | Easy Negatives | Hard Negatives |
|--------|---------------|----------------|
| Initial Acc | 80-90% | 60-70% |
| Final Acc | 95-99% | 85-95% |
| Training Epochs | 10-20 | 3-5 |
| Margin | +0.1 to +0.2 | +0.2 to +0.4 |
| Generalization | Lower | Higher |
| Production Ready | No | Yes |

## Conclusion

Hard negatives are essential for production-quality embeddings. They:
- Force model to learn subtle distinctions
- Improve generalization to real queries
- Reduce training time (more informative examples)
- Create more robust embeddings

Always prefer hard negatives over random or easy negatives!
