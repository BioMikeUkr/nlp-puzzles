# Module 6: RAG (Retrieval-Augmented Generation)

> Production RAG systems: chunking, retrieval, reranking, and generation

## Why This Matters

LLMs have knowledge cutoff dates and can't access your private data. RAG solves this by retrieving relevant context from your documents and injecting it into the LLM prompt. This enables LLMs to answer questions about your data accurately without fine-tuning.

## Key Concepts

### What is RAG?

**Traditional LLM:** Limited to training data
```python
response = llm("What's our Q4 revenue?")
# "I don't have access to that information"
```

**RAG Pipeline:**
```
1. User asks: "What's our Q4 revenue?"
2. Retrieve relevant docs from your database
3. Inject docs as context: "Given: Q4 revenue was $2.5M... Question: What's our Q4 revenue?"
4. LLM answers: "Q4 revenue was $2.5M"
```

### RAG Pipeline Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       v
┌─────────────────────────┐
│ 1. Query Enhancement    │  (optional: expand, rephrase)
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│ 2. Retrieval            │
│  - Embed query          │
│  - Search FAISS (top 100)│
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│ 3. Reranking            │  (optional but recommended)
│  - Cross-encoder        │
│  - Rerank to top 5      │
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│ 4. Context Construction │
│  - Format chunks        │
│  - Check token limit    │
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│ 5. Generation           │
│  - Build prompt         │
│  - Call LLM             │
│  - Return answer + sources│
└──────┬──────────────────┘
       │
       v
┌─────────────┐
│   Answer    │
└─────────────┘
```

## Document Chunking

**Problem:** Documents are too long for LLM context

**Solution:** Split into chunks

### Chunking Strategies

**1. Fixed-size chunking:**
```python
# Simple but can split mid-sentence
chunks = [text[i:i+500] for i in range(0, len(text), 500)]
```

**2. Sentence-based chunking:**
```python
# Split on sentence boundaries
sentences = text.split('. ')
chunks = []
current_chunk = ""
for sent in sentences:
    if len(current_chunk) + len(sent) < 500:
        current_chunk += sent + ". "
    else:
        chunks.append(current_chunk)
        current_chunk = sent + ". "
```

**3. Recursive character splitting (recommended):**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,  # Overlap prevents context loss
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)
```

### Chunking Parameters

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| **chunk_size** | 500-1000 chars | Larger = more context, fewer chunks |
| **chunk_overlap** | 50-200 chars | Prevents splitting related content |
| **separators** | `["\n\n", "\n", ". "]` | Prioritize semantic boundaries |

## Indexing Pipeline

```python
# 1. Load documents
documents = load_docs("data/")

# 2. Chunk documents
chunks = []
for doc in documents:
    doc_chunks = splitter.split_text(doc.content)
    for i, chunk in enumerate(doc_chunks):
        chunks.append({
            "text": chunk,
            "source": doc.filename,
            "chunk_id": i,
            "metadata": doc.metadata
        })

# 3. Embed chunks
embeddings = embed_model.encode([c["text"] for c in chunks])

# 4. Build FAISS index
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

# 5. Save index + metadata
faiss.write_index(index, "index.faiss")
save_json(chunks, "chunks.json")
```

## Retrieval Pipeline

```python
def retrieve(query: str, k: int = 5):
    # 1. Embed query
    query_emb = embed_model.encode(query)

    # 2. Search FAISS (retrieve more for reranking)
    scores, indices = index.search(query_emb, k=100)
    candidates = [chunks[i] for i in indices[0]]

    # 3. Rerank with cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = cross_encoder.predict(pairs)

    # 4. Sort by rerank scores
    ranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    return [chunk for chunk, score in ranked]
```

## Prompt Engineering for RAG

### Basic Template

```python
template = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
```

### Advanced Template

```python
template = """You are a helpful assistant answering questions based on provided context.

Instructions:
- Answer ONLY based on the context below
- If the answer is not in the context, say "I don't have enough information"
- Cite sources using [Source: filename]
- Be concise but complete

Context:
{context}

Question: {question}

Answer:"""
```

### Context Construction

```python
def build_context(chunks: list, max_tokens: int = 3000):
    """Build context from chunks, respecting token limit"""
    context_parts = []
    total_tokens = 0

    for chunk in chunks:
        # Estimate tokens
        chunk_tokens = len(chunk["text"]) // 4  # ~4 chars per token

        if total_tokens + chunk_tokens > max_tokens:
            break

        # Format chunk
        context_parts.append(
            f"[Source: {chunk['source']}]\n{chunk['text']}\n"
        )
        total_tokens += chunk_tokens

    return "\n---\n".join(context_parts)
```

## Evaluation Metrics

### Retrieval Metrics

**Recall@k:** What % of relevant docs are in top-k?
```python
def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)
```

**MRR (Mean Reciprocal Rank):**
```python
def mrr(retrieved_ids, relevant_ids):
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0
```

### Generation Metrics

**Answer Relevance:** Does answer address question?
**Faithfulness:** Is answer supported by retrieved context?
**Context Precision:** Are retrieved chunks relevant?

## Common Patterns

### Pattern 1: Simple RAG

```python
def simple_rag(question: str):
    # Retrieve
    chunks = retrieve(question, k=5)
    context = build_context(chunks)

    # Generate
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = llm(prompt)

    return response
```

### Pattern 2: RAG with Citations

```python
def rag_with_citations(question: str):
    chunks = retrieve(question, k=5)

    # Build numbered context
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[{i}] {chunk['text']}\n\n"

    prompt = f"""Context:
{context}

Question: {question}

Answer (cite sources using [1], [2], etc.):"""

    response = llm(prompt)

    return {
        "answer": response,
        "sources": [{"id": i+1, "source": c["source"]}
                   for i, c in enumerate(chunks)]
    }
```

### Pattern 3: Conversational RAG

```python
class ConversationalRAG:
    def __init__(self):
        self.history = []

    def ask(self, question: str):
        # Retrieve using question + history context
        full_context = self._build_history_context()
        search_query = full_context + "\n" + question

        chunks = retrieve(search_query, k=5)
        context = build_context(chunks)

        # Generate with conversation history
        messages = [
            {"role": "system", "content": "Answer based on context"},
            *self.history,
            {"role": "user", "content": f"Context: {context}\n\n{question}"}
        ]

        response = llm.chat(messages)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response
```

## Documentation & Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [RAG Evaluation](https://docs.ragas.io/)

## Self-Assessment Checklist

- [ ] I understand how RAG works end-to-end
- [ ] I can chunk documents effectively
- [ ] I know when to use overlap in chunking
- [ ] I can build and query FAISS indexes
- [ ] I understand the retrieve-rerank pattern
- [ ] I can construct prompts with context
- [ ] I know how to handle token limits
- [ ] I can implement citation tracking

---


## Practice Questions

See [QUESTIONS.md](./QUESTIONS.md) for 30 deep-dive questions with detailed answers covering:
- Architecture & Design (Q1-Q10)
- Implementation & Coding (Q11-Q20)
- Debugging & Troubleshooting (Q21-Q25)
- Trade-offs & Decisions (Q26-Q30)

---

## Additional Resources

- [RAG Paper (original)](https://arxiv.org/abs/2005.11401)
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag/)
- [Evaluating RAG Systems](https://docs.ragas.io/)
- [Production RAG](https://github.com/run-llama/llama_index)
