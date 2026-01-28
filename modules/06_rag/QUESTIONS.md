# Deep Dive Q&A - RAG (Retrieval-Augmented Generation)

> 30 questions covering production RAG systems, chunking, retrieval, and generation

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

#### Q5: How do you implement metadata filtering in retrieval?

**Answer:**

**Basic metadata filtering:**
```python
from typing import Dict, List

class DocumentWithMetadata:
    def __init__(self, text: str, metadata: dict, embedding: list):
        self.text = text
        self.metadata = metadata
        self.embedding = embedding

# Documents with rich metadata
docs = [
    DocumentWithMetadata(
        text="Python is a programming language",
        metadata={"type": "tutorial", "language": "en", "year": 2024},
        embedding=[...]
    ),
    DocumentWithMetadata(
        text="Java is statically typed",
        metadata={"type": "article", "language": "en", "year": 2023},
        embedding=[...]
    )
]

def retrieve_with_filters(query: str, metadata_filter: dict) -> List:
    """Retrieve only documents matching filters"""
    results = []

    for doc in docs:
        # Check filters
        match = True
        for key, value in metadata_filter.items():
            if doc.metadata.get(key) != value:
                match = False
                break

        if match:
            # Calculate relevance
            similarity = cosine_similarity(query_embedding, doc.embedding)
            results.append((doc, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)

# Usage
results = retrieve_with_filters(
    query="Python basics",
    metadata_filter={"type": "tutorial", "year": 2024}
)
```

**Advanced filtering:**
```python
class MetadataFilter:
    def __init__(self):
        self.conditions = []

    def add_condition(self, field: str, operator: str, value):
        """Add filter condition"""
        self.conditions.append((field, operator, value))

    def matches(self, metadata: dict) -> bool:
        """Check if metadata matches all conditions"""
        for field, operator, value in self.conditions:
            doc_value = metadata.get(field)

            if operator == "==":
                if doc_value != value:
                    return False
            elif operator == ">":
                if not (doc_value > value):
                    return False
            elif operator == "<":
                if not (doc_value < value):
                    return False
            elif operator == "in":
                if doc_value not in value:
                    return False

        return True

# Usage
filter = MetadataFilter()
filter.add_condition("year", ">", 2023)
filter.add_condition("type", "in", ["tutorial", "guide"])
filter.add_condition("language", "==", "en")

matching_docs = [d for d in docs if filter.matches(d.metadata)]
```

---

#### Q6: How do you implement hybrid search (semantic + keyword)?

**Answer:**

**Combining semantic and keyword search:**
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridSearch:
    def __init__(self, embed_model, bm25_model):
        self.embed_model = embed_model
        self.bm25 = bm25_model
        self.documents = []

    def index(self, docs: list):
        """Index documents for both search types"""
        self.documents = docs

        # Index for semantic search
        texts = [d["text"] for d in docs]
        self.embeddings = self.embed_model.encode(texts)

        # Index for keyword search
        self.bm25.index(texts)

    def search(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        Hybrid search with weighted combination

        Args:
            alpha: Weight for semantic (1-alpha for keyword)
        """
        # Semantic search
        query_emb = self.embed_model.encode(query)
        semantic_scores = cosine_similarity([query_emb], self.embeddings)[0]

        # Normalize to [0, 1]
        semantic_scores = (semantic_scores - semantic_scores.min()) / (
            semantic_scores.max() - semantic_scores.min() + 1e-10
        )

        # Keyword search
        keyword_scores = self.bm25.get_scores(query)

        # Normalize to [0, 1]
        keyword_scores = (keyword_scores - keyword_scores.min()) / (
            keyword_scores.max() - keyword_scores.min() + 1e-10
        )

        # Combine scores
        combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

        # Get top-k
        top_indices = np.argsort(combined_scores)[-k:][::-1]

        return [
            {"doc": self.documents[i], "score": combined_scores[i]}
            for i in top_indices
        ]

# Usage
hybrid = HybridSearch(embed_model, bm25)
hybrid.index(documents)

# 50% semantic, 50% keyword
results = hybrid.search("Python tutorial", alpha=0.5)

# More semantic-focused
results = hybrid.search("Python tutorial", alpha=0.8)
```

**When to use each:**
```python
# Semantic search excels at:
# - "What does AI mean?" (not exact match needed)
# - Paraphrased queries
# - Conceptual search

# Keyword search excels at:
# - "Find ticket #12345" (exact match)
# - Technical terms
# - Named entities

# Hybrid is best for general-purpose RAG
```

---

#### Q7: How do you build conversational RAG (multi-turn)?

**Answer:**

**Multi-turn context management:**
```python
from datetime import datetime

class ConversationalRAG:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self.conversation = []
        self.context_window = 5  # Keep last 5 turns

    def add_turn(self, role: str, content: str):
        """Add to conversation history"""
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

    def get_relevant_history(self, query: str) -> str:
        """Get relevant previous context"""
        # Keep recent turns
        recent = self.conversation[-self.context_window*2:]

        history_str = ""
        for turn in recent:
            if turn["role"] == "user":
                history_str += f"User: {turn['content']}\n"
            else:
                history_str += f"Assistant: {turn['content']}\n"

        return history_str

    def answer(self, query: str) -> str:
        """Answer with conversation context"""
        self.add_turn("user", query)

        # Retrieve relevant docs
        docs = self.retriever.retrieve(query)

        # Build prompt with history
        history = self.get_relevant_history(query)

        prompt = f"""Answer based on context and conversation history.

Conversation history:
{history}

Relevant documents:
{format_docs(docs)}

Current question: {query}

Answer:"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        self.add_turn("assistant", answer)

        return answer

# Usage
rag = ConversationalRAG(retriever, llm_client)

rag.answer("Tell me about Python")
rag.answer("What about async?")  # Remembers Python context
rag.answer("How does it relate to my earlier question?")  # Multi-turn
```

---

#### Q8: How do you implement query expansion for better retrieval?

**Answer:**

**Expanding queries to improve recall:**
```python
class QueryExpander:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def expand_query(self, query: str, num_expansions: int = 3) -> list:
        """Generate alternative phrasings of query"""
        prompt = f"""Generate {num_expansions} alternative ways to ask this question:

Question: {query}

Alternative questions (one per line):"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        alternatives = response.choices[0].message.content.split('\n')
        return [query] + [alt.strip() for alt in alternatives if alt.strip()]

    def retrieve_with_expansion(self, query: str, retriever, k: int = 5):
        """Retrieve using expanded queries"""
        # Generate alternatives
        expanded_queries = self.expand_query(query)

        # Retrieve for each
        all_results = {}
        for q in expanded_queries:
            results = retriever.retrieve(q, k=k)
            for result in results:
                doc_id = result["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = {"doc": result, "score": 0}

                all_results[doc_id]["score"] += result["score"]

        # Return merged results sorted by accumulated score
        merged = sorted(
            all_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:k]

        return merged

# Usage
expander = QueryExpander(llm_client)

# Original query
query = "How to train models?"

# Retrieved docs improve with expansion
results = expander.retrieve_with_expansion(query, retriever)
```

---

#### Q9: How do you implement hierarchical chunking for structured documents?

**Answer:**

**Hierarchical structure:**
```python
class HierarchicalChunk:
    def __init__(self, level: int, title: str, content: str, children: list = None):
        self.level = level  # 0=heading1, 1=heading2, etc
        self.title = title
        self.content = content
        self.children = children or []
        self.full_text = self._get_full_text()

    def _get_full_text(self) -> str:
        """Full text including hierarchy"""
        text = f"# {self.title}\n{self.content}"

        for child in self.children:
            text += "\n" + child._get_full_text()

        return text

def build_hierarchy(markdown_text: str) -> List[HierarchicalChunk]:
    """Parse markdown into hierarchy"""
    lines = markdown_text.split('\n')
    stack = []  # Stack of open levels
    root_chunks = []

    for line in lines:
        if line.startswith('#'):
            # Count # for level
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('# ').strip()

            # Pop stack to current level
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            chunk = HierarchicalChunk(
                level=level,
                title=title,
                content=""
            )

            if stack:
                # Add as child to current parent
                stack[-1]["chunk"].children.append(chunk)
            else:
                # Top-level chunk
                root_chunks.append(chunk)

            stack.append({"level": level, "chunk": chunk})
        else:
            # Add content to current chunk
            if stack:
                stack[-1]["chunk"].content += line + "\n"

    return root_chunks

# Usage
doc = """# Python
Python is a language.

## Syntax
Variables are defined with =.

## Functions
def my_func(): pass

# JavaScript
JavaScript runs in browsers.

## Async
Promises and async/await
"""

hierarchy = build_hierarchy(doc)

# Now can retrieve at different levels
# - Section level (larger chunks)
# - Subsection level (medium chunks)
# - Paragraph level (small chunks)
```

---

#### Q10: How do you evaluate RAG systems?

**Answer:**

**Evaluation metrics:**
```python
from typing import List, Dict

class RAGEvaluator:
    @staticmethod
    def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Fraction of relevant docs that were retrieved"""
        if not relevant_ids:
            return 1.0

        matching = len(set(retrieved_ids) & set(relevant_ids))
        return matching / len(relevant_ids)

    @staticmethod
    def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Fraction of retrieved docs that are relevant"""
        if not retrieved_ids:
            return 0.0

        matching = len(set(retrieved_ids) & set(relevant_ids))
        return matching / len(retrieved_ids)

    @staticmethod
    def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Mean Reciprocal Rank (how high is first relevant?)"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1 / (i + 1)

        return 0.0

    @staticmethod
    def calculate_ndcg(scores: List[float], relevant_indices: List[int]) -> float:
        """Normalized Discounted Cumulative Gain"""
        # DCG
        dcg = sum([
            1 / (i + 2) if i in relevant_indices else 0
            for i in range(len(scores))
        ])

        # IDCG (ideal ranking)
        idcg = sum([1 / (i + 2) for i in range(min(len(relevant_indices), len(scores)))])

        return dcg / idcg if idcg > 0 else 0

# Usage
evaluator = RAGEvaluator()

# For a query
retrieved = ["doc1", "doc2", "doc3"]
relevant = ["doc1", "doc4"]

recall = evaluator.calculate_recall(retrieved, relevant)  # 0.5
precision = evaluator.calculate_precision(retrieved, relevant)  # 0.33
mrr = evaluator.calculate_mrr(retrieved, relevant)  # 1.0 (first match)

print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"MRR: {mrr:.2f}")
```

**Batch evaluation:**
```python
def evaluate_rag(test_queries: List[Dict]) -> Dict:
    """
    Evaluate RAG on test set

    Args:
        test_queries: [
            {
                "query": str,
                "relevant_doc_ids": List[str]
            }
        ]
    """
    recalls = []
    precisions = []
    mrrs = []

    for test in test_queries:
        retrieved = retriever.retrieve(test["query"], k=5)
        retrieved_ids = [r["id"] for r in retrieved]

        recall = evaluator.calculate_recall(
            retrieved_ids,
            test["relevant_doc_ids"]
        )
        recalls.append(recall)

        precision = evaluator.calculate_precision(
            retrieved_ids,
            test["relevant_doc_ids"]
        )
        precisions.append(precision)

        mrr = evaluator.calculate_mrr(
            retrieved_ids,
            test["relevant_doc_ids"]
        )
        mrrs.append(mrr)

    return {
        "avg_recall": np.mean(recalls),
        "avg_precision": np.mean(precisions),
        "avg_mrr": np.mean(mrrs)
    }

# Run evaluation
metrics = evaluate_rag(test_queries)
print(f"Avg Recall: {metrics['avg_recall']:.3f}")
```

---

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

#### Q12: How do you maintain RAG with conversation memory?

**Answer:**

**Memory-augmented RAG:**
```python
class MemoryAugmentedRAG:
    def __init__(self, retriever, llm_client, memory_size=10):
        self.retriever = retriever
        self.llm_client = llm_client
        self.memory = []  # Store (query, retrieved_docs) pairs
        self.memory_size = memory_size

    def extract_entities(self, text: str) -> List[str]:
        """Extract key entities for memory"""
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Extract key entities: {text}"
            }]
        )

        return response.choices[0].message.content.split(',')

    def augment_query(self, query: str) -> str:
        """Augment query with memory context"""
        if not self.memory:
            return query

        # Get recent relevant queries from memory
        memory_context = "\n".join([
            f"Previous: {m['query']}"
            for m in self.memory[-3:]
        ])

        augmented = f"""{memory_context}

Current: {query}"""

        return augmented

    def answer(self, query: str) -> Dict:
        """Answer with memory augmentation"""
        # Augment query with history
        augmented_query = self.augment_query(query)

        # Retrieve documents
        docs = self.retriever.retrieve(augmented_query)

        # Generate answer
        prompt = f"""Based on these documents and prior context:

{format_docs(docs)}

Question: {query}

Answer:"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        # Store in memory
        self.memory.append({
            "query": query,
            "docs": docs,
            "answer": answer
        })

        # Keep memory bounded
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

        return {
            "answer": answer,
            "docs": docs,
            "memory_size": len(self.memory)
        }
```

---

#### Q13: How do you stream RAG responses?

**Answer:**

**Streaming with retrieval:**
```python
def stream_rag_response(query: str, retriever, llm_client):
    """Stream RAG response token-by-token"""

    # Retrieve documents (sync, not streamed)
    docs = retriever.retrieve(query, k=5)

    # Build context
    context = format_docs(docs)

    # Stream generation
    prompt = f"""Answer based on context:

{context}

Question: {query}

Answer:"""

    with llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ) as stream:
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {
                    "token": chunk.choices[0].delta.content,
                    "sources": docs  # Static sources (retrieved once)
                }

# Usage in web app
@app.post("/rag/stream")
async def rag_stream(query: str):
    async def generate():
        for item in stream_rag_response(query, retriever, llm_client):
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

#### Q14: How do you handle multi-document retrieval and synthesis?

**Answer:**

**Multi-document synthesis:**
```python
class MultiDocumentSynthesizer:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def synthesize(self, query: str, num_sources: int = 3) -> Dict:
        """
        Retrieve from multiple sources and synthesize
        """
        # Retrieve from different sources
        results = {}
        for source in ["internal_docs", "external_docs", "faq"]:
            docs = self.retriever.retrieve(
                query,
                source_filter=source,
                k=num_sources
            )
            results[source] = docs

        # Synthesize across sources
        synthesis_prompt = f"""Compare and synthesize information from multiple sources:

Internal docs:
{format_docs(results['internal_docs'])}

External docs:
{format_docs(results['external_docs'])}

FAQ:
{format_docs(results['faq'])}

Question: {query}

Synthesized answer (note: cite source for each claim):"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": synthesis_prompt}]
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": results
        }

# Usage
synthesizer = MultiDocumentSynthesizer(retriever, llm_client)

result = synthesizer.synthesize("How to deploy ML models?")
print(result["answer"])
```

---

#### Q15: How do you implement response caching in RAG?

**Answer:**

**Intelligent caching:**
```python
import hashlib
import redis

class RAGCache:
    def __init__(self, redis_host="localhost"):
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
        self.ttl = 3600  # 1 hour

    def _cache_key(self, query: str) -> str:
        """Generate cache key"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"rag:{query_hash}"

    def get_cached(self, query: str):
        """Try to get from cache"""
        key = self._cache_key(query)
        cached = self.redis.get(key)

        if cached:
            return json.loads(cached)

        return None

    def cache_result(self, query: str, result: Dict):
        """Cache RAG result"""
        key = self._cache_key(query)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(result)
        )

    def should_cache(self, query: str) -> bool:
        """Decide if query result is cache-worthy"""
        # Don't cache: current, personalized, short
        non_cacheable = ["today", "now", "my ", "I "]
        return not any(word in query.lower() for word in non_cacheable)

def rag_with_cache(query: str, retriever, llm_client, cache):
    """RAG with intelligent caching"""

    # Try cache first
    if cache.should_cache(query):
        cached = cache.get_cached(query)
        if cached:
            return {**cached, "from_cache": True}

    # Retrieve and generate
    docs = retriever.retrieve(query)
    answer = generate_answer(docs, query, llm_client)

    result = {"answer": answer, "docs": docs}

    # Cache if appropriate
    if cache.should_cache(query):
        cache.cache_result(query, result)

    return {**result, "from_cache": False}
```

---

#### Q16: How do you implement async retrieval and generation?

**Answer:**

**Parallel async operations:**
```python
import asyncio
from openai import AsyncOpenAI

class AsyncRAG:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    async def retrieve_async(self, query: str, k: int = 5):
        """Async retrieval (wrap sync retriever)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retriever.retrieve(query, k=k)
        )

    async def generate_async(self, context: str, query: str):
        """Async generation"""
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Context: {context}\n\nQ: {query}"
            }]
        )

        return response.choices[0].message.content

    async def answer_async(self, query: str):
        """Combined async RAG"""
        # Retrieve and generate in parallel
        docs = await self.retrieve_async(query)
        context = format_docs(docs)

        answer = await self.generate_async(context, query)

        return {"answer": answer, "docs": docs}

# Usage
async_rag = AsyncRAG(retriever, llm_client)

result = asyncio.run(async_rag.answer_async("What is RAG?"))
```

---

#### Q17: How do you implement feedback loops for RAG improvement?

**Answer:**

**Learning from user feedback:**
```python
class RAGWithFeedback:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self.feedback_log = []

    def answer(self, query: str) -> Dict:
        """Answer with feedback tracking"""
        docs = self.retriever.retrieve(query, k=5)
        answer = generate_answer(docs, query, self.llm_client)

        # Create response with feedback mechanism
        response_id = str(uuid.uuid4())

        return {
            "id": response_id,
            "answer": answer,
            "docs": docs,
            "query": query
        }

    def log_feedback(self, response_id: str, feedback: Dict):
        """Log user feedback"""
        self.feedback_log.append({
            "response_id": response_id,
            "feedback": feedback,  # {"helpful": True, "docs_relevant": False}
            "timestamp": datetime.now()
        })

    def analyze_feedback(self) -> Dict:
        """Analyze feedback to improve"""
        # Calculate metrics
        total = len(self.feedback_log)
        helpful = sum(1 for f in self.feedback_log if f["feedback"].get("helpful"))
        relevant_docs = sum(1 for f in self.feedback_log if f["feedback"].get("docs_relevant"))

        return {
            "total_responses": total,
            "helpful_rate": helpful / max(total, 1),
            "doc_relevance": relevant_docs / max(total, 1)
        }

    def get_problematic_queries(self) -> List[str]:
        """Find queries with poor feedback"""
        bad_feedback = [
            f for f in self.feedback_log
            if not f["feedback"].get("helpful", True)
        ]

        return [b["query"] for b in bad_feedback[-10:]]
```

---

#### Q18: How do you add confidence scores to RAG answers?

**Answer:**

**Confidence estimation:**
```python
class ConfidentRAG:
    def __init__(self, retriever, llm_client, reranker):
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = reranker

    def calculate_confidence(self, query: str, docs: List, answer: str) -> float:
        """Estimate answer confidence"""
        # Factor 1: Document relevance
        doc_relevance = np.mean([d["score"] for d in docs])

        # Factor 2: Agreement among top documents
        doc_texts = [d["text"] for d in docs[:3]]
        agreement = self._measure_agreement(doc_texts, answer)

        # Factor 3: Query coverage in documents
        query_coverage = self._measure_coverage(query, doc_texts)

        # Combine factors
        confidence = (
            0.5 * doc_relevance +
            0.3 * agreement +
            0.2 * query_coverage
        )

        return min(confidence, 1.0)  # Clamp to [0, 1]

    def _measure_agreement(self, texts: List[str], answer: str) -> float:
        """Measure how many docs support the answer"""
        support = 0

        for text in texts:
            # Check if document contains key terms from answer
            answer_terms = set(answer.lower().split())
            text_terms = set(text.lower().split())

            overlap = len(answer_terms & text_terms) / len(answer_terms)
            if overlap > 0.5:
                support += 1

        return support / len(texts)

    def _measure_coverage(self, query: str, texts: List[str]) -> float:
        """Measure if documents cover query"""
        query_terms = set(query.lower().split())

        all_terms = set()
        for text in texts:
            all_terms.update(text.lower().split())

        return len(query_terms & all_terms) / len(query_terms)

    def answer_with_confidence(self, query: str) -> Dict:
        """Generate answer with confidence"""
        docs = self.retriever.retrieve(query, k=5)
        answer = generate_answer(docs, query, self.llm_client)

        confidence = self.calculate_confidence(query, docs, answer)

        return {
            "answer": answer,
            "confidence": confidence,
            "docs": docs,
            "label": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        }
```

---

#### Q19: How do you monitor RAG system performance?

**Answer:**

**Monitoring and observability:**
```python
from datetime import datetime
import logging

class RAGMonitor:
    def __init__(self):
        self.metrics = []
        self.logger = logging.getLogger("RAG")

    def track_query(self, query: str, response_time: float, num_docs: int, confidence: float):
        """Track RAG query metrics"""
        self.metrics.append({
            "timestamp": datetime.now(),
            "query_length": len(query),
            "response_time": response_time,
            "num_docs": num_docs,
            "confidence": confidence
        })

    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        if not self.metrics:
            return {}

        response_times = [m["response_time"] for m in self.metrics]
        confidences = [m["confidence"] for m in self.metrics]

        return {
            "total_queries": len(self.metrics),
            "avg_response_time": np.mean(response_times),
            "p99_response_time": np.percentile(response_times, 99),
            "avg_confidence": np.mean(confidences),
            "slow_queries": sum(1 for rt in response_times if rt > 2)
        }

    def get_alerts(self) -> List[str]:
        """Generate alerts for issues"""
        alerts = []

        report = self.get_performance_report()

        if report.get("avg_response_time", 0) > 2:
            alerts.append(f"Slow queries: {report['avg_response_time']:.1f}s avg")

        if report.get("avg_confidence", 1) < 0.5:
            alerts.append(f"Low confidence: {report['avg_confidence']:.2f} avg")

        return alerts
```

---

#### Q20: How do you version control and iterate on RAG components?

**Answer:**

**Component versioning:**
```python
from enum import Enum
import json

class RAGComponentVersion:
    def __init__(self, name: str, version: str, config: dict):
        self.name = name
        self.version = version
        self.config = config
        self.created_at = datetime.now()

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "config": self.config,
            "created_at": self.created_at.isoformat()
        }

class VersionedRAG:
    def __init__(self):
        self.embedder_version = "all-mpnet-base-v2"
        self.reranker_version = "ms-marco-MiniLM-L-6-v2"
        self.llm_version = "gpt-4o-mini"
        self.chunk_config = {"size": 600, "overlap": 120}

    def get_config(self) -> dict:
        """Get full RAG config"""
        return {
            "embedder": self.embedder_version,
            "reranker": self.reranker_version,
            "llm": self.llm_version,
            "chunking": self.chunk_config,
            "timestamp": datetime.now().isoformat()
        }

    def save_checkpoint(self, path: str):
        """Save RAG configuration"""
        with open(f"{path}/rag_config.json", "w") as f:
            json.dump(self.get_config(), f, indent=2)

    def load_checkpoint(self, path: str):
        """Load RAG configuration"""
        with open(f"{path}/rag_config.json") as f:
            config = json.load(f)

        self.embedder_version = config["embedder"]
        self.reranker_version = config["reranker"]
        self.llm_version = config["llm"]
        self.chunk_config = config["chunking"]

# Experimentation
def experiment_embedder_version(test_queries: list):
    """A/B test different embedders"""
    embedders = ["all-mpnet-base-v2", "all-t5-3b"]
    results = {}

    for embedder in embedders:
        rag = VersionedRAG()
        rag.embedder_version = embedder

        metrics = evaluate_rag(rag, test_queries)
        results[embedder] = metrics

    return results

# Compare: which embedder performs better?
```

---

### Debugging & Troubleshooting (21-25)

#### Q21: How do you debug poor retrieval quality?

**Answer:**

**Diagnostic workflow:**
```python
class RetrievalDebugger:
    def __init__(self, retriever, test_queries):
        self.retriever = retriever
        self.test_queries = test_queries

    def analyze_retrieval(self, query: str, relevant_docs: List[str]):
        """Analyze why retrieval failed"""
        print(f"Query: {query}")
        print(f"Expected relevant docs: {relevant_docs}")

        # Retrieve
        results = self.retriever.retrieve(query, k=10)
        retrieved_ids = [r["id"] for r in results]

        print(f"Retrieved docs: {retrieved_ids}")

        # Diagnose
        missed = set(relevant_docs) - set(retrieved_ids)

        if missed:
            print(f"\n✗ Missed {len(missed)} relevant docs:")
            for doc_id in missed:
                self._why_missed(query, doc_id)

    def _why_missed(self, query: str, doc_id: str):
        """Why was this doc not retrieved?"""
        doc = get_doc_by_id(doc_id)

        # Measure similarity
        similarity = calculate_similarity(query, doc["text"])

        print(f"\n  Doc: {doc_id}")
        print(f"  Similarity: {similarity:.3f}")

        # Check overlapping terms
        query_terms = set(query.lower().split())
        doc_terms = set(doc["text"].lower().split())
        overlap = query_terms & doc_terms

        print(f"  Term overlap: {len(overlap)}/{len(query_terms)}")
        if not overlap:
            print(f"  → NO OVERLAPPING TERMS")
            print(f"  → Try: expand query, improve chunking, or retrain embedder")

        # Check embedding quality
        query_emb = embed_model.encode(query)
        doc_emb = embed_model.encode(doc["text"])
        emb_sim = cosine_similarity([query_emb], [doc_emb])[0][0]

        print(f"  Embedding similarity: {emb_sim:.3f}")

# Usage
debugger = RetrievalDebugger(retriever, test_queries)

debugger.analyze_retrieval(
    query="What is machine learning?",
    relevant_docs=["doc1", "doc2", "doc3"]
)
```

**Common issues and fixes:**
```python
# Issue 1: Low term overlap
# Cause: Query phrasing doesn't match document
# Fix: Query expansion, or improve chunking

# Issue 2: Good embedding similarity but not retrieved
# Cause: Other docs have higher similarity
# Fix: Adjust chunk size (smaller = more precise)

# Issue 3: Consistent low recall
# Cause: Embedder not suited for domain
# Fix: Fine-tune embedder or use domain-specific model
```

---

#### Q22: How do you handle hallucinations in RAG answers?

**Answer:**

**Preventing hallucinations:**
```python
class HallucinationDetector:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def detect_hallucination(self, context: str, answer: str) -> bool:
        """Check if answer is grounded in context"""
        prompt = f"""Given this context, is the answer fully supported by the context?

Context:
{context}

Answer:
{answer}

Are all claims in the answer supported by the context? Answer: yes/no"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content.lower()
        return "no" in result  # True if hallucination detected

    def constrain_generation(self, context: str, query: str):
        """Generate answer constrained to context"""
        prompt = f"""Answer ONLY using information from the context.
If information is not in the context, say "Not found in context."

Context:
{context}

Question: {query}

Answer (constrained to context):"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Reduce randomness
        )

        return response.choices[0].message.content

# Usage
detector = HallucinationDetector(llm_client)

answer = generate_rag_answer(context, query)

if detector.detect_hallucination(context, answer):
    print("Hallucination detected!")
    answer = detector.constrain_generation(context, query)
```

---

#### Q23: How do you optimize for slow RAG performance?

**Answer:**

**Performance profiling:**
```python
import time
from functools import wraps

def time_operation(func):
    """Measure operation time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        print(f"{func.__name__}: {elapsed:.3f}s")
        return result

    return wrapper

class PerformanceOptimizedRAG:
    @time_operation
    def retrieve(self, query: str) -> List:
        # Measure retrieval time
        return self.retriever.retrieve(query)

    @time_operation
    def generate(self, context: str, query: str) -> str:
        # Measure generation time
        return call_llm(context, query)

    def full_answer(self, query: str):
        docs = self.retrieve(query)
        answer = self.generate(format_docs(docs), query)

        return answer

# Profile RAG
rag = PerformanceOptimizedRAG()
result = rag.full_answer("What is RAG?")

# Output:
# retrieve: 0.234s ← Slow! (should be <0.1s)
# generate: 1.823s ← Expected (1-3s is normal)
```

**Optimization strategies:**
```python
# Slow retrieval (<0.1s expected)
# → Reduce k (fewer docs to retrieve)
# → Use smaller embedding model
# → Add indexing (FAISS faster than exhaustive search)

# Slow generation (1-3s expected)
# → Use smaller LLM (gpt-4o-mini vs gpt-4o)
# → Reduce context size
# → Enable streaming (feels faster)

# Example optimizations
def optimized_rag(query: str):
    # Fast retrieval
    docs = retriever.retrieve(query, k=3)  # Reduce from 5

    # Rerank top-1 for quality
    top_doc = rerank([docs[0]], query)[0]

    # Minimal context
    context = top_doc["text"][:500]  # Truncate

    # Stream generation
    return stream_generate(context, query)
```

---

#### Q24: How do you handle missing context in RAG?

**Answer:**

**Detecting and recovering from missing context:**
```python
class ContextAwarenessRAG:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def check_context_coverage(self, query: str, docs: List) -> float:
        """Estimate if docs fully cover query"""
        context_text = " ".join([d["text"] for d in docs])

        prompt = f"""Rate how well these documents answer this query (0-100):

Query: {query}

Documents:
{context_text}

Coverage score (0-100):"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            score = int(response.choices[0].message.content.strip())
            return score / 100
        except:
            return 0.5

    def answer_with_fallback(self, query: str):
        """Answer with fallback for missing context"""
        # Retrieve documents
        docs = self.retriever.retrieve(query, k=5)

        # Check coverage
        coverage = self.check_context_coverage(query, docs)

        if coverage < 0.5:
            print(f"Low coverage ({coverage:.1%}), expanding search...")

            # Expand search
            expanded_query = self._expand_query(query)
            more_docs = self.retriever.retrieve(expanded_query, k=10)
            docs.extend(more_docs)

        # Generate answer
        answer = generate_answer(docs, query, self.llm_client)

        return {
            "answer": answer,
            "coverage": coverage,
            "num_docs": len(docs)
        }

    def _expand_query(self, query: str) -> str:
        """Expand query to find more related docs"""
        prompt = f"""Expand this query to find more relevant documents:

Query: {query}

Expanded query (broader, includes synonyms):"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
```

---

#### Q25: How do you handle inconsistent RAG outputs?

**Answer:**

**Consistency strategies:**
```python
class ConsistentRAG:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def answer_consistently(self, query: str, runs: int = 3) -> Dict:
        """Generate multiple answers and pick consensus"""
        answers = []

        for run in range(runs):
            # Same retrieval
            docs = self.retriever.retrieve(query, k=5)

            # Multiple generations (with temperature=0 should be similar)
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": format_docs(docs) + f"\nQ: {query}"
                }],
                temperature=0  # Deterministic
            )

            answers.append(response.choices[0].message.content)

        # Check consistency
        if len(set(answers)) == 1:
            return {"answer": answers[0], "consistency": "high"}

        # Inconsistent: pick longest (usually most complete)
        best = max(answers, key=len)

        return {
            "answer": best,
            "consistency": "low",
            "variants": len(set(answers))
        }
```

---

### Trade-offs & Decisions (26-30)

#### Q26: Embedding model selection - which one to use?

**Answer:**

**Model comparison:**
```python
# Small models (fast, low memory)
# - all-MiniLM-L6-v2 (22M params)
# - Best for: production, mobile, cost-sensitive
# - Speed: 100+ docs/sec
# - Accuracy: Good for general tasks

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Medium models (balanced)
# - all-mpnet-base-v2 (109M params)
# - Best for: most RAG systems
# - Speed: 50+ docs/sec
# - Accuracy: Very good

embedder = SentenceTransformer("all-mpnet-base-v2")

# Large models (highest accuracy)
# - all-t5-3b (3B params)
# - Best for: research, complex queries
# - Speed: 10+ docs/sec
# - Accuracy: Excellent

embedder = SentenceTransformer("all-t5-3b")

# Domain-specific
# - ance (optimized for retrieval)
# - e5-base (multilingual)
# - Best for: specialized domains

embedder = SentenceTransformer("ance")

# Decision tree:
# Production RAG → all-MiniLM-L6-v2 (default)
# High accuracy needed → all-mpnet-base-v2
# Research quality → all-t5-3b
# Domain-specific → fine-tune or domain model
```

---

#### Q27: Latency vs accuracy trade-off

**Answer:**

**Tuning for different requirements:**
```python
# High latency tolerance (24h batch)
def rag_high_accuracy():
    # Use largest embedder
    embedder = SentenceTransformer("all-t5-3b")

    # Retrieve more documents
    docs = retriever.retrieve(query, k=20)

    # Use best LLM
    llm = "gpt-4o"  # Highest quality

    return generate_answer(docs, query, llm)

# Low latency, high throughput (user-facing)
def rag_low_latency():
    # Use small embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Fewer documents
    docs = retriever.retrieve(query, k=3)

    # Faster LLM
    llm = "gpt-4o-mini"  # 4x faster

    # Or stream for perceived speed
    return stream_generate(docs, query)

# Typical latencies:
# Retrieval: 50-500ms (depends on index size)
# Generation: 1-5s (depends on response length)
# Total: 1.5-5.5s
```

---

#### Q28: RAG vs fine-tuning for knowledge

**Answer:**

**When to use each:**
```python
# RAG (Retrieval-Augmented Generation)
# When:
# ✓ Knowledge changes frequently (quarterly earnings)
# ✓ Need to cite sources
# ✓ Large knowledge base (1M+ documents)
# ✓ Quick setup needed
# Cost: 0.15-10 USD per 1M tokens
# Accuracy: 7-9/10 (depends on retrieval quality)

def use_rag():
    docs = retrieve_from_vector_db(query)
    return generate_answer(docs, query)

# Fine-tuning (Knowledge in model weights)
# When:
# ✓ Knowledge is stable (not changing)
# ✓ Specific writing style needed
# ✓ Privacy critical (no data sent to API)
# ✓ Can afford training time (hours)
# Cost: 5-50 USD training + inference cost
# Accuracy: 8-10/10 (model knows deeply)

def use_finetuning():
    fine_tuned_model = "ft:gpt-4o-mini:mycompany:v1"
    return generate_answer(query, model=fine_tuned_model)

# Hybrid (best of both)
# Fine-tune for style, RAG for knowledge
def hybrid():
    docs = retrieve_from_vector_db(query)
    return generate_with_finetuned_model(docs, query)

# Decision:
# Static knowledge → Fine-tune
# Dynamic knowledge → RAG
# Both? → Hybrid
```

---

#### Q29: Cost optimization in RAG

**Answer:**

**Reducing RAG costs:**
```python
# 1. Reduce embeddings calls
@lru_cache(maxsize=10000)
def embed_cached(text: str):
    return embed_model.encode(text)

# 2. Use smaller LLM
# gpt-4o-mini vs gpt-4o: 16x cheaper
response = client.chat.completions.create(
    model="gpt-4o-mini"  # Not gpt-4o
)

# 3. Reduce context size
docs = retriever.retrieve(query, k=3)  # Not k=10
context = format_docs(docs[:100])  # Truncate

# 4. Batch operations
# Embed 1000 docs in 1 call vs 1000 calls

# 5. Use local embedder (not API)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Free
# vs
embeddings = client.embeddings.create(model="text-embedding-3-large")  # Paid

# Cost breakdown for 100K queries/month:
embeddings = 0  # Free (local)
retrieval = 50  # FAISS index
generation = 500  # LLM calls at 0.15/1M input + 0.60/1M output
total = 550  # ~$550/month

# Optimization example:
# Before: 1000/month
# After: 550/month (45% savings)
```

---

#### Q30: Scaling RAG for production

**Answer:**

**Production deployment:**
```python
# 1. Vector index scaling
# Option A: FAISS (local, millions of vectors)
import faiss
index = faiss.IndexFlatIP(embedding_dim)

# Option B: Pinecone (hosted, 100M+ vectors)
import pinecone
index = pinecone.Index("rag-index")

# Option C: Weaviate (cloud-native, distributed)

# 2. Document storage
# SQLite: ≤100K docs
# PostgreSQL: ≤10M docs with vector extension
# Specialized: MongoDB, Elasticsearch

# 3. Cache layer
# Redis for query caching
# Reduces API calls by 40-60%

# 4. Async processing
# Queue-based architecture
# Process 100s of concurrent requests

class ProductionRAG:
    def __init__(self):
        # Distributed setup
        self.vector_index = pinecone.Index("production")
        self.document_db = PostgreSQL()
        self.cache = redis.Redis()
        self.queue = RabbitMQ()

    async def answer(self, query: str):
        # Check cache
        cached = self.cache.get(query)
        if cached:
            return cached

        # Queue for async processing if urgent
        self.queue.enqueue(process_query, query)

        # Return streaming result
        docs = self.vector_index.query(query, top_k=5)
        answer = await generate_async(docs, query)

        # Cache result
        self.cache.setex(query, 3600, answer)

        return answer
```

**Scaling checklist:**
```
✓ Vector index: Pinecone or distributed FAISS
✓ Document store: PostgreSQL + vector extension
✓ Cache: Redis (query cache + embedding cache)
✓ Async: Queue-based (RabbitMQ, Celery)
✓ Monitoring: Prometheus + alerting
✓ Logging: ELK stack for debugging
✓ Load testing: Simulate production traffic
✓ Auto-scaling: Container orchestration (K8s)
```

---

