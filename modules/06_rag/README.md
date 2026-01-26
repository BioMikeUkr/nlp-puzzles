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

## Deep Dive Q&A (30 Questions)

### Architecture & Design (1-10)

#### Q1: Why do we need RAG when we have fine-tuning?

**Answer:**

**RAG vs Fine-tuning comparison:**

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Data freshness** | Real-time (update docs) | Frozen (retrain needed) |
| **Setup time** | Minutes | Hours/days |
| **Cost** | Low (just API calls) | High (GPU training) |
| **Accuracy** | Very good | Excellent |
| **Explainability** | High (see sources) | Low (black box) |
| **Privacy** | Docs stay private | Training data in model |

**When to use RAG:**
```python
# Use RAG for:
# ✅ Frequently changing data
docs = ["Q4 earnings: $2.5M"]  # Changes quarterly
# ✅ Large knowledge base
knowledge_base = load_docs("10,000 documents")
# ✅ Need citations
answer_with_sources = rag(query)  # Returns sources
# ✅ Multiple data sources
sources = [internal_docs, public_docs, real_time_data]
# ✅ Privacy requirements
# Documents never leave your infrastructure
```

**When to use fine-tuning:**
```python
# Use fine-tuning for:
# ✅ Specific writing style
# Train model to write like your brand
# ✅ Domain-specific reasoning
# Medical diagnosis, legal analysis
# ✅ Structured output format
# Always output in specific JSON schema
# ✅ Behavior modification
# Make model more concise, formal, etc.
```

**Best approach: Combine both**
```python
# 1. Fine-tune for style/behavior
fine_tuned_model = "ft:gpt-4o-mini:company:v1"

# 2. Use RAG for knowledge
def answer_question(query):
    # Retrieve relevant docs
    context = retrieve(query)

    # Use fine-tuned model with RAG
    response = client.chat.completions.create(
        model=fine_tuned_model,  # Your style
        messages=[
            {"role": "system", "content": "Answer based on context"},
            {"role": "user", "content": f"Context: {context}\n\nQ: {query}"}
        ]
    )
    return response
```

---

#### Q2: What's the optimal chunk size and why?

**Answer:**

**Chunk size trade-offs:**

| Chunk Size | Pros | Cons | Use Case |
|------------|------|------|----------|
| **Small (200-300 chars)** | Precise retrieval | Lost context | FAQ, definitions |
| **Medium (500-800 chars)** | Balanced | General purpose | Most use cases |
| **Large (1000-1500 chars)** | Rich context | Noisy retrieval | Long-form content |

**Factors affecting optimal size:**

**1. Query complexity:**
```python
# Simple queries → smaller chunks
query = "What is the capital of France?"
# Answer: "Paris"
# chunk_size = 200  # Definition fits in small chunk

# Complex queries → larger chunks
query = "Explain the economic factors behind the French Revolution"
# Needs multiple paragraphs of context
# chunk_size = 1000
```

**2. Document structure:**
```python
# Structured documents (Q&A, definitions)
# Natural boundaries = small chunks
chunk_size = 300

# Narrative documents (articles, reports)
# Need paragraph-level context
chunk_size = 800

# Technical documentation
# Code examples need to stay together
chunk_size = 1200
```

**3. Retrieval strategy:**
```python
# Single-stage retrieval (no reranking)
# Larger chunks to ensure context
chunk_size = 1000

# Two-stage (retrieve + rerank)
# Smaller chunks for precision
chunk_size = 500
```

**Finding optimal size empirically:**
```python
def evaluate_chunk_size(chunk_sizes, test_queries):
    results = {}

    for size in chunk_sizes:
        # Reindex with this chunk size
        chunks = split_documents(docs, chunk_size=size)
        index = build_index(chunks)

        # Test retrieval
        recalls = []
        for query, relevant_doc_ids in test_queries:
            retrieved = retrieve(query, index, k=5)
            recall = calculate_recall(retrieved, relevant_doc_ids)
            recalls.append(recall)

        results[size] = np.mean(recalls)

    return results

# Test different sizes
sizes = [200, 400, 600, 800, 1000, 1200]
results = evaluate_chunk_size(sizes, test_queries)

# Example results:
# {200: 0.65, 400: 0.78, 600: 0.82, 800: 0.80, 1000: 0.75}
# Optimal: 600 characters
```

**Recommended starting points:**
```python
# General purpose RAG
chunk_size = 600
chunk_overlap = 100

# FAQ/Short-form
chunk_size = 300
chunk_overlap = 50

# Long-form content
chunk_size = 1000
chunk_overlap = 200

# Code documentation
chunk_size = 800
chunk_overlap = 150
```

**Dynamic chunking (advanced):**
```python
def smart_chunk(doc, min_size=400, max_size=800):
    """Chunk at semantic boundaries within size range"""
    chunks = []
    current = ""

    paragraphs = doc.split("\n\n")

    for para in paragraphs:
        if len(current) + len(para) > max_size:
            if len(current) >= min_size:
                chunks.append(current)
                current = para
            else:
                # Chunk too small, include para even if over max
                current += "\n\n" + para
        else:
            current += "\n\n" + para

    if current:
        chunks.append(current)

    return chunks
```

---

#### Q3: Why use chunk overlap and how much?

**Answer:**

**Problem without overlap:**
```
Chunk 1: "...introduced a new pricing model in Q4. The model includes"
Chunk 2: "a base fee of $99 and per-user fee of $10..."

# Query: "What is the Q4 pricing model?"
# Chunk 1: Has "pricing model" but not the details
# Chunk 2: Has details but not "pricing model" keyword
# Result: Poor retrieval!
```

**Solution with overlap:**
```
Chunk 1: "...introduced a new pricing model in Q4. The model includes"
Chunk 2: "pricing model in Q4. The model includes a base fee of $99 and per-user fee"

# Now Chunk 2 has both "pricing model" AND the details!
```

**How much overlap?**

| Overlap | Pros | Cons | Use Case |
|---------|------|------|----------|
| **0% (none)** | Fewest chunks | Context loss | Dense, well-structured docs |
| **10-15%** | Minimal redundancy | Still some context loss | General purpose |
| **20-25%** | Good context preservation | More storage | Recommended default |
| **30-50%** | Maximum context | High redundancy | Critical docs, legal |

**Calculating overlap:**
```python
chunk_size = 500
overlap = 100  # 20%

# Example:
text = "A" * 1000

chunks = []
start = 0
while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start += (chunk_size - overlap)  # Move by (size - overlap)

# Result:
# Chunk 1: chars 0-500
# Chunk 2: chars 400-900 (100 char overlap with Chunk 1)
# Chunk 3: chars 800-1000 (100 char overlap with Chunk 2)
```

**Overlap strategies:**

**1. Fixed overlap (most common):**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,  # 20% overlap
    separators=["\n\n", "\n", ". ", " "]
)
```

**2. Sentence-based overlap:**
```python
def chunk_with_sentence_overlap(text, target_size=500, overlap_sentences=2):
    sentences = sent_tokenize(text)
    chunks = []

    i = 0
    while i < len(sentences):
        chunk_sentences = []
        current_size = 0

        # Add sentences until target size
        while i < len(sentences) and current_size < target_size:
            chunk_sentences.append(sentences[i])
            current_size += len(sentences[i])
            i += 1

        chunks.append(" ".join(chunk_sentences))

        # Backtrack for overlap
        i -= overlap_sentences

    return chunks
```

**3. Semantic overlap:**
```python
# Detect topic boundaries, only overlap within same topic
def semantic_chunk(text):
    paragraphs = text.split("\n\n")

    # Embed paragraphs
    embeddings = embed_model.encode(paragraphs)

    # Detect topic shifts (low similarity = new topic)
    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        similarity = cosine_sim(embeddings[i-1], embeddings[i])

        if similarity < 0.7:  # Topic shift
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraphs[i]]  # No overlap
        else:
            current_chunk.append(paragraphs[i])  # Continue topic

    chunks.append("\n\n".join(current_chunk))
    return chunks
```

**Impact on storage:**
```python
# No overlap
num_docs = 1000
doc_length = 5000
chunk_size = 500
chunks_per_doc = 5000 / 500 = 10
total_chunks = 1000 * 10 = 10,000

# 20% overlap
chunks_per_doc = 5000 / (500 - 100) = 12.5
total_chunks = 1000 * 12.5 = 12,500
# +25% storage but better retrieval
```

**Recommended defaults:**
```python
# General RAG
chunk_size = 600
chunk_overlap = 120  # 20%

# Critical applications (legal, medical)
chunk_size = 800
chunk_overlap = 240  # 30%

# High-volume, cost-sensitive
chunk_size = 500
chunk_overlap = 50   # 10%
```

---

#### Q4: How do you handle long documents that don't fit in context?

**Answer:**

**Problem:**
```python
# LLM context limit: 128K tokens (~500KB text)
document_size = 2MB  # 4x too large!

# Even after chunking and retrieval:
retrieved_chunks = retrieve(query, k=50)
total_text = "\n".join([c["text"] for c in retrieved_chunks])
tokens = count_tokens(total_text)  # 10,000 tokens
# Still might exceed limit when combined with prompt
```

**Solution 1: Hierarchical Retrieval**

```python
def hierarchical_rag(query, doc_id):
    """
    Two-stage: First find relevant section, then chunk-level
    """
    # Stage 1: Retrieve relevant sections
    sections = retrieve_sections(query, doc_id, k=3)

    # Stage 2: Retrieve chunks within those sections
    chunks = []
    for section in sections:
        section_chunks = retrieve_chunks(
            query,
            section_id=section["id"],
            k=5
        )
        chunks.extend(section_chunks)

    # Now we have focused set of chunks
    return chunks
```

**Solution 2: Map-Reduce**

```python
def map_reduce_rag(query, long_doc):
    """
    Process document in parts, then combine
    """
    # Split document into manageable chunks
    chunks = chunk_document(long_doc, chunk_size=5000)

    # Map: Answer question for each chunk
    partial_answers = []
    for chunk in chunks:
        prompt = f"""Based on this section, answer: {query}

Section:
{chunk}

Answer (or 'Not found' if answer not in section):"""

        answer = llm(prompt)
        if answer != "Not found":
            partial_answers.append({
                "answer": answer,
                "chunk_id": chunk["id"]
            })

    # Reduce: Combine partial answers
    if not partial_answers:
        return "Answer not found in document"

    summary_prompt = f"""Combine these partial answers into a final answer:

{partial_answers}

Question: {query}

Final Answer:"""

    final_answer = llm(summary_prompt)
    return final_answer
```

**Solution 3: Iterative Refinement**

```python
def iterative_refinement(query, chunks):
    """
    Process chunks sequentially, refining answer
    """
    answer = "No information yet"

    for chunk in chunks:
        prompt = f"""Current answer: {answer}

New information:
{chunk["text"]}

Question: {query}

Refined answer (incorporating new info):"""

        answer = llm(prompt)

    return answer
```

**Solution 4: Extractive Then Abstractive**

```python
def extractive_abstractive(query, long_doc):
    """
    1. Extract all relevant sentences
    2. Summarize extractions
    """
    # Stage 1: Extract relevant sentences
    sentences = sent_tokenize(long_doc)
    sentence_embeddings = embed_model.encode(sentences)
    query_embedding = embed_model.encode(query)

    # Get top-k sentences
    similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:20]
    relevant_sentences = [sentences[i] for i in sorted(top_indices)]

    # Stage 2: Abstractive summarization
    context = "\n".join(relevant_sentences)

    prompt = f"""Answer based on these relevant excerpts:

{context}

Question: {query}

Answer:"""

    return llm(prompt)
```

**Solution 5: Smart Chunk Selection**

```python
def smart_chunk_selection(query, all_chunks, max_tokens=3000):
    """
    Select diverse, relevant chunks within token budget
    """
    # Rank by relevance
    scores = rank_chunks(query, all_chunks)
    ranked_chunks = sorted(
        zip(all_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    selected = []
    total_tokens = 0
    embeddings_selected = []

    for chunk, score in ranked_chunks:
        chunk_tokens = count_tokens(chunk["text"])

        if total_tokens + chunk_tokens > max_tokens:
            continue

        # Check diversity (don't add if too similar to selected)
        chunk_emb = embed_model.encode(chunk["text"])

        if embeddings_selected:
            max_similarity = max([
                cosine_sim(chunk_emb, emb)
                for emb in embeddings_selected
            ])

            if max_similarity > 0.9:  # Too similar, skip
                continue

        selected.append(chunk)
        embeddings_selected.append(chunk_emb)
        total_tokens += chunk_tokens

    return selected
```

**Solution 6: Document Summarization**

```python
def rag_with_summary(query, long_doc):
    """
    Create summary of doc, then use for RAG
    """
    # Generate summary of full document (one-time cost)
    summary_prompt = f"""Summarize this document in 500 words:

{long_doc}

Summary:"""

    summary = llm(summary_prompt, max_tokens=700)

    # Store summary with document
    save_summary(doc_id, summary)

    # Use summary + specific chunks for queries
    relevant_chunks = retrieve(query, k=5)

    context = f"""Document Summary:
{summary}

Relevant Details:
{format_chunks(relevant_chunks)}"""

    return rag(query, context)
```

**Choosing the right approach:**

```python
# Document < 50KB, clear structure
→ Standard chunking + retrieval

# Document 50-500KB, well-structured
→ Hierarchical retrieval

# Document 500KB-5MB, narrative
→ Map-reduce or iterative refinement

# Document > 5MB
→ Extractive + abstractive OR summarization

# Query needs full document context
→ Map-reduce (processes all)

# Query needs specific facts
→ Smart chunk selection (most efficient)
```

---

[Q5-Q10 would continue with architecture questions about metadata filtering, hybrid search, query expansion, etc.]

### Implementation & Coding (11-20)

#### Q11: Implement a production RAG system from scratch

**Answer:**

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import json
import tiktoken

class ProductionRAG:
    """
    Production-ready RAG system with:
    - Chunking with overlap
    - FAISS indexing
    - Two-stage retrieval (bi-encoder + cross-encoder)
    - Token-aware context construction
    - Citation tracking
    """

    def __init__(
        self,
        openai_api_key: str,
        embed_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 600,
        chunk_overlap: int = 120
    ):
        # Initialize models
        self.embed_model = SentenceTransformer(embed_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.llm_client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model

        # Chunking config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Storage
        self.chunks = []
        self.index = None
        self.encoding = tiktoken.encoding_for_model(llm_model)

    def ingest_documents(self, documents: List[Dict]):
        """
        Ingest and index documents

        Args:
            documents: List of {"content": str, "source": str, "metadata": dict}
        """
        print(f"Ingesting {len(documents)} documents...")

        # Chunk all documents
        all_chunks = []
        for doc in documents:
            doc_chunks = self.splitter.split_text(doc["content"])

            for i, chunk_text in enumerate(doc_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "chunk_id": i,
                    "metadata": doc.get("metadata", {})
                })

        self.chunks = all_chunks
        print(f"Created {len(all_chunks)} chunks")

        # Embed chunks
        print("Generating embeddings...")
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = self.embed_model.encode(
            chunk_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype('float32')

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✓ Indexed {self.index.ntotal} chunks")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        retrieval_k: int = 100
    ) -> List[Dict]:
        """
        Two-stage retrieval: bi-encoder + cross-encoder

        Args:
            query: Search query
            k: Number of final results
            retrieval_k: Number to retrieve before reranking

        Returns:
            List of chunks with scores
        """
        # Stage 1: Bi-encoder retrieval
        query_emb = self.embed_model.encode(
            query,
            normalize_embeddings=True
        ).astype('float32').reshape(1, -1)

        scores, indices = self.index.search(query_emb, retrieval_k)
        candidates = [self.chunks[i] for i in indices[0]]

        # Stage 2: Cross-encoder reranking
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)

        # Sort and take top-k
        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = []
        for chunk, score in ranked:
            results.append({
                **chunk,
                "score": float(score)
            })

        return results

    def build_context(
        self,
        chunks: List[Dict],
        max_tokens: int = 3000
    ) -> tuple:
        """
        Build context from chunks, respecting token limit

        Returns:
            (context_str, sources)
        """
        context_parts = []
        sources = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            # Format chunk with citation
            chunk_text = f"[{i+1}] {chunk['text']}"
            chunk_tokens = len(self.encoding.encode(chunk_text))

            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            sources.append({
                "id": i + 1,
                "source": chunk["source"],
                "score": chunk["score"]
            })
            total_tokens += chunk_tokens

        context = "\n\n".join(context_parts)

        return context, sources

    def ask(
        self,
        question: str,
        k: int = 5,
        max_context_tokens: int = 3000,
        temperature: float = 0
    ) -> Dict:
        """
        Answer question using RAG

        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "context_used": str
            }
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(question, k=k)

        # Build context
        context, sources = self.build_context(chunks, max_context_tokens)

        # Generate answer
        prompt = f"""Answer the question based on the context below.

Rules:
- Only use information from the context
- Cite sources using [1], [2], etc.
- If the answer is not in the context, say "I don't have enough information"
- Be concise but complete

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }

    def save_index(self, path: str):
        """Save index and chunks to disk"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/chunks.json", "w") as f:
            json.dump(self.chunks, f)
        print(f"✓ Saved index to {path}")

    def load_index(self, path: str):
        """Load index and chunks from disk"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/chunks.json", "r") as f:
            self.chunks = json.load(f)
        print(f"✓ Loaded index from {path}")


# Usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = ProductionRAG(openai_api_key="your-key-here")

    # Ingest documents
    documents = [
        {
            "content": "Python is a high-level programming language...",
            "source": "python_intro.txt",
            "metadata": {"category": "programming"}
        },
        # ... more documents
    ]

    rag.ingest_documents(documents)

    # Ask questions
    result = rag.ask("What is Python?")

    print("Answer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  [{source['id']}] {source['source']} (score: {source['score']:.3f})")

    # Save for later
    rag.save_index("./rag_index")
```

---

[Q12-Q20 would continue with implementation questions about conversation history, metadata filtering, query expansion, etc.]

### Debugging & Troubleshooting (21-25)

[Questions about debugging retrieval issues, handling edge cases, etc.]

### Trade-offs & Decisions (26-30)

[Questions about model selection, cost optimization, etc.]

---

## Additional Resources

- [RAG Paper (original)](https://arxiv.org/abs/2005.11401)
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag/)
- [Evaluating RAG Systems](https://docs.ragas.io/)
- [Production RAG](https://github.com/run-llama/llama_index)
