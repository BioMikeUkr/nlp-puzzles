# Module 14: LangChain & Orchestration — Interview Questions

## Architecture & Design (10 questions)

**Q1. What is LCEL and what problem does it solve?**

LCEL (LangChain Expression Language) is a declarative way to compose LangChain components using the pipe operator `|`. It solves the problem of boilerplate wiring between prompts, LLMs, and parsers. Every LCEL component is a `Runnable` with a uniform interface (`.invoke()`, `.stream()`, `.batch()`), so you can compose arbitrarily complex pipelines without writing glue code. The key benefit is that streaming works transparently — if any component supports streaming, the whole chain streams.

---

**Q2. What is a Runnable and which interface does it expose?**

A `Runnable` is the base abstraction in LCEL. Every LangChain component (prompt templates, LLMs, output parsers, retrievers, custom functions) implements it. The interface:
- `.invoke(input)` — single synchronous call
- `.stream(input)` — generator yielding chunks
- `.batch(inputs)` — parallel calls over a list
- `.ainvoke()`, `.astream()`, `.abatch()` — async versions

This uniformity means you can swap any component in a chain without changing the calling code.

---

**Q3. How does a LangChain agent differ from a simple LCEL chain?**

A chain is a fixed, linear sequence of steps (prompt → LLM → parser). An agent is a loop: the LLM decides which tool to call, the tool is executed, the result is fed back, and the LLM decides the next step — until it produces a final answer. The key difference is **dynamic branching**: a chain's steps are predetermined; an agent's steps depend on LLM reasoning at runtime.

---

**Q4. Explain the role of `MessagesPlaceholder` in an agent prompt.**

`MessagesPlaceholder` reserves a slot in the prompt template for a list of messages at a variable position. For agents, it holds `agent_scratchpad` — the sequence of tool calls and their results accumulated during the agent's reasoning loop. Without it, the agent couldn't "remember" what tools it already called. It's inserted after the human message so the LLM sees: system instruction → user question → [tool calls so far] → decide next action.

---

**Q5. What are the trade-offs between RAG-as-a-chain vs RAG-as-a-tool inside an agent?**

| | RAG chain | RAG as agent tool |
|--|-----------|-------------------|
| **When to use** | Query always needs retrieval | Agent decides if retrieval is needed |
| **Control** | Always retrieves | May skip if answer already known |
| **Latency** | Lower (no agent loop) | Higher (extra LLM decision step) |
| **Complexity** | Low | High |

RAG-as-a-tool is better when the agent handles multiple query types and retrieval is only sometimes needed.

---

**Q6. Why is the docstring of a `@tool` function critical?**

The docstring becomes the tool's description in the LLM's function schema. The agent reads this description to decide when and whether to call the tool. A vague docstring ("processes data") leads to wrong tool selection. A precise docstring ("searches the company knowledge base for policy information; use when the customer asks about procedures, policies, or troubleshooting steps") guides the agent to call it in the right situations.

---

**Q7. How does `RunnablePassthrough` work and when is it needed?**

`RunnablePassthrough` passes its input unchanged. It's needed in RAG chains where you need to supply both retrieved context and the original question to the prompt:
```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
```
When you pass `"How long does a refund take?"`, `RunnablePassthrough` forwards it as `question` while the retriever processes it for `context`. Without it, the question string would need to be duplicated or restructured.

---

**Q8. What does `max_iterations` in `AgentExecutor` protect against?**

It caps the number of tool-call cycles the agent is allowed before the executor forces a stop. Without it, a confused LLM can loop indefinitely (calling the same tool repeatedly, getting stuck). Setting `max_iterations=5` means: after 5 tool calls, the executor returns whatever the agent has produced so far. It's a safety valve, not a design target — a well-designed agent should converge in 2-3 iterations.

---

**Q9. When should you use LangChain vs custom code?**

Use LangChain for:
- Agents that choose between multiple tools dynamically
- Rapid prototyping of complex multi-step pipelines
- Teams already familiar with the framework

Use custom code for:
- Simple single-call classification or extraction
- Performance-critical paths (every abstraction adds overhead)
- Long-term maintenance where framework API stability matters
- Full control over prompts, retries, and error handling

The guiding principle: if your pipeline is a fixed sequence of API calls, custom code is simpler. If the LLM needs to decide which path to take, an agent framework earns its complexity.

---

**Q10. How does LangChain's FAISS integration work for RAG?**

`FAISS.from_texts(texts, embeddings, metadatas)` embeds all texts using the provided embeddings model (e.g., `OpenAIEmbeddings`) and builds an in-memory FAISS index. `vectorstore.as_retriever(search_kwargs={"k": 3})` creates a `VectorStoreRetriever` that runs semantic search on `.invoke(query)`, returning the top-k `Document` objects. In a RAG chain, the retriever is piped into a formatter function that converts documents to a context string.

---

## Implementation & Coding (10 questions)

**Q11. Write a minimal LCEL chain that classifies a support ticket into a category and returns JSON.**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", 'Classify the support ticket. Return JSON: {"category": "billing|technical|account|shipping", "priority": "high|medium|low"}'),
    ("human", "{ticket}"),
])

chain = prompt | llm | JsonOutputParser()
result = chain.invoke({"ticket": "I was charged twice"})
# → {"category": "billing", "priority": "high"}
```

---

**Q12. Implement a `@tool` that wraps a FAISS retriever for use inside an agent.**

```python
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for policy and troubleshooting information.
    Use this when the customer asks about procedures, policies, refunds, shipping,
    account settings, or how to resolve a specific issue.

    Args:
        query: The customer's question or search terms.

    Returns:
        Relevant knowledge base articles as a formatted string.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )
```

---

**Q13. Build a full LangChain agent with two tools using `create_tool_calling_agent`.**

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [classify_ticket_tool, search_knowledge_base]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent. Use the available tools to help customers."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

result = executor.invoke({"input": "Why am I getting 429 errors from your API?"})
print(result["output"])
```

---

**Q14. How do you stream tokens from an LCEL chain?**

```python
for chunk in chain.stream({"ticket": "API is down"}):
    print(chunk, end="", flush=True)
```

For `JsonOutputParser`, streaming yields partial dicts as they arrive. For `StrOutputParser`, it yields string fragments. If the LLM doesn't support streaming, chunks will be yielded all at once at the end.

---

**Q15. What is `RunnableLambda` and when would you use it instead of a plain function?**

`RunnableLambda` wraps any Python function as a `Runnable`, giving it `.invoke()`, `.stream()`, `.batch()`, and async variants. Use it when you need a plain function inside an LCEL pipe:
```python
from langchain_core.runnables import RunnableLambda

preprocess = RunnableLambda(lambda x: x.strip().lower())
chain = preprocess | prompt | llm | StrOutputParser()
```

Without `RunnableLambda`, you can't put a plain `lambda` directly in a pipe.

---

**Q16. How would you batch-process 20 tickets through a LangChain classification chain?**

```python
tickets = [{"ticket": t["text"]} for t in tickets_json]
results = chain.batch(tickets)  # runs in parallel by default
```

`.batch()` accepts a list of inputs and runs them concurrently using `asyncio`. You can control concurrency with `config={"max_concurrency": 5}`. This is equivalent to `asyncio.gather()` over `.ainvoke()` calls.

---

**Q17. How do you make an agent tool return structured output instead of a raw string?**

The `@tool` function must return a string (tools always return strings to the agent), but you can serialize structured data:
```python
@tool
def classify_ticket(ticket_text: str) -> str:
    """..."""
    result = classification_chain.invoke({"ticket": ticket_text})
    return json.dumps(result)  # {"category": "billing", "priority": "high"}
```

The agent then reads the JSON string as its observation. If the final answer needs structured output, parse at the `AgentExecutor` output level.

---

**Q18. How do you prevent an agent from calling tools in an infinite loop?**

Three mechanisms:
1. `max_iterations=5` in `AgentExecutor` — hard cap on tool calls
2. Clear, non-overlapping tool descriptions — prevents the agent from trying the same tool repeatedly
3. `handle_parsing_errors=True` — gracefully handles malformed tool calls instead of crashing, which can trigger retries
```python
executor = AgentExecutor(
    agent=agent, tools=tools,
    max_iterations=5,
    handle_parsing_errors=True,
)
```

---

**Q19. How would you evaluate RAG quality programmatically?**

```python
def evaluate_rag(rag_chain, test_questions):
    scores = []
    for q in test_questions:
        answer = rag_chain.invoke(q["question"]).lower()
        hits = sum(1 for kw in q["expected_keywords"] if kw.lower() in answer)
        scores.append(hits / len(q["expected_keywords"]))
    return sum(scores) / len(scores)  # avg keyword coverage

coverage = evaluate_rag(rag_chain, test_questions)
assert coverage >= 0.7, f"RAG quality too low: {coverage:.2%}"
```

This checks that answers contain expected keywords from the relevant articles.

---

**Q20. How do you use `ChatPromptTemplate.from_messages` with a system prompt and history?**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="history"),  # conversation history
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # tool calls
])
```

`MessagesPlaceholder` inserts a list of `BaseMessage` objects at that position. Each placeholder has a `variable_name` that must be provided when invoking the chain.

---

## Debugging & Troubleshooting (5 questions)

**Q21. The agent keeps calling the same tool repeatedly without making progress. How do you debug this?**

1. Set `verbose=True` in `AgentExecutor` to see each tool call and observation
2. Check if the tool is returning useful output — an empty or error response causes the agent to retry
3. Check the tool docstring — if it's too vague, the agent may not understand when it has enough information
4. Add `return_intermediate_steps=True` to inspect the full reasoning trace:
```python
result = executor.invoke({"input": "..."}, config={"callbacks": []})
for step in result["intermediate_steps"]:
    print(step[0].tool, "→", step[1])
```
5. Lower `max_iterations` to force early termination while debugging

---

**Q22. The RAG chain returns "I don't have that information" even when the answer exists. What's wrong?**

Common causes:
1. **Wrong embedding model** — the query and documents were embedded with different models
2. **k too small** — `search_kwargs={"k": 1}` might miss the right document; increase to 3-5
3. **Text chunking issues** — the relevant text is split across chunks that don't individually answer the question
4. **Prompt too strict** — if the prompt says "use ONLY the context", a slightly imprecise retrieval leads to "don't know"

Debug by checking what the retriever actually returns:
```python
docs = retriever.invoke("How long does a refund take?")
for doc in docs:
    print(doc.page_content[:100])
```

---

**Q23. `JsonOutputParser` raises a parse error. How do you handle this gracefully?**

The LLM may not always return valid JSON. Options:
1. Add `"Return ONLY valid JSON, no markdown code blocks"` to the system prompt
2. Wrap with a fallback parser:
```python
from langchain_core.runnables import RunnableWithFallbacks

safe_chain = chain.with_fallbacks([backup_chain])
```
3. Use a custom output function that catches parse errors:
```python
def safe_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to extract JSON from markdown fences
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group()) if match else {}
```

---

**Q24. The agent tool raises an exception. What happens and how should you handle it?**

By default, an unhandled exception in a tool propagates and crashes the `AgentExecutor`. To handle gracefully:
1. Return error strings instead of raising exceptions inside tools:
```python
@tool
def classify_ticket(ticket_text: str) -> str:
    """..."""
    try:
        result = chain.invoke({"ticket": ticket_text})
        return json.dumps(result)
    except Exception as e:
        return f"Error: {str(e)}"
```
2. Set `handle_parsing_errors=True` in `AgentExecutor` for malformed tool call recovery

The agent then receives the error string as its observation and can decide to try a different approach.

---

**Q25. FAISS index is rebuilt on every run, making startup slow. How would you persist it?**

```python
# Save
vectorstore.save_local("faiss_index")

# Load (fast, no re-embedding)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

This serializes the FAISS index and document store to disk. On subsequent runs, loading takes milliseconds instead of re-embedding all documents. Rebuild only when the knowledge base changes.

---

## Trade-offs & Decisions (5 questions)

**Q26. When would you choose `JsonOutputParser` over `PydanticOutputParser`?**

`JsonOutputParser` returns a plain dict — fast, no validation, flexible structure. `PydanticOutputParser` validates the JSON against a Pydantic model, raises errors on invalid output, and gives you typed access. Use `PydanticOutputParser` when you need guaranteed field presence and type safety (e.g., downstream code depends on specific fields). Use `JsonOutputParser` for prototyping or when the LLM output structure is variable.

---

**Q27. The agent sometimes answers directly without calling any tools. Is this a problem?**

Not necessarily. Tool-calling agents are designed to use tools only when needed. If the LLM can answer from its training data (e.g., "What is Python?"), it's correct to skip tools. It becomes a problem if the agent skips tools when it should use them (e.g., answering a company-specific policy question from training data instead of searching the knowledge base). Fix this by strengthening the system prompt: "Always use the search_knowledge_base tool for any questions about our company's policies, pricing, or procedures."

---

**Q28. How does `.batch()` differ from calling `.invoke()` in a loop?**

`.batch()` runs calls concurrently (using `asyncio.gather` internally), while a loop is sequential. For 20 tickets with 1-second LLM latency each, a loop takes ~20 seconds; `.batch()` takes ~1-2 seconds (limited by API rate limits and concurrency). However, `.batch()` can overwhelm rate limits — use `config={"max_concurrency": 5}` to cap parallel calls. For sequential processing with dependencies between calls, stick to a loop.

---

**Q29. Why might you wrap a RAG chain as an agent tool rather than always running RAG?**

When the agent handles multiple query types, many questions don't require retrieval:
- "What is your refund policy?" → needs KB search
- "Can you summarize what I just told you?" → no search needed
- "Classify this ticket: [text]" → classification, not retrieval

Making RAG a tool lets the LLM skip the embedding + retrieval latency when unnecessary. The trade-off: an extra LLM decision step adds ~0.5-1 second. Worth it when >30% of queries don't need retrieval.

---

**Q30. How do you choose the number of retrieved documents (k) for RAG?**

Start with `k=3-5`. Factors that push k higher:
- Long answers that require synthesizing multiple sources
- Short documents (single paragraphs) where any one might be partially relevant
- Diverse question types requiring different articles

Factors that push k lower:
- Long documents (k=3 already fills the context window)
- Token budget constraints (each document adds to prompt cost)
- Precision matters more than recall (irrelevant context hurts LLM focus)

Measure experimentally: run your test questions with k=1, 3, 5, 7, and plot keyword coverage vs. cost.
