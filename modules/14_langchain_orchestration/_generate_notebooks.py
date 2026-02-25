"""Generate all notebooks for Module 14: LangChain & Orchestration."""
import nbformat
import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nb():
    n = nbformat.v4.new_notebook()
    n.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return n


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(src):
    return nbformat.v4.new_code_cell(src)


def save(notebook, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        nbformat.write(notebook, f)
    print(f"  wrote {path}")


BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared setup blocks
# ---------------------------------------------------------------------------

LANGCHAIN_SETUP = """\
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

OPENAI_API_KEY = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
print("✓ LLM initialized")
"""

FIXTURES_SETUP = """\
import json, os

# In Jupyter, os.getcwd() returns the notebook's directory (learning/, tasks/, solutions/)
# Fixtures are one level up at ../fixtures/input/
fixtures = os.path.normpath(os.path.join(os.getcwd(), "..", "fixtures", "input"))

with open(os.path.join(fixtures, "tickets.json")) as f:
    tickets = json.load(f)

with open(os.path.join(fixtures, "knowledge_base.json")) as f:
    knowledge_base = json.load(f)

with open(os.path.join(fixtures, "test_questions.json")) as f:
    test_questions = json.load(f)

print(f"✓ Loaded {len(tickets)} tickets, {len(knowledge_base)} KB articles, {len(test_questions)} test questions")
"""

# ---------------------------------------------------------------------------
# Learning 01 — LCEL Basics
# ---------------------------------------------------------------------------

def make_learning_01():
    n = nb()
    n.cells = [
        md("# Learning 01: LCEL Basics\n\nLangChain Expression Language (LCEL) lets you compose chains with the `|` pipe operator. Every component is a `Runnable` with `.invoke()`, `.stream()`, and `.batch()`."),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        md("## 1. Your First Chain: Ticket Classifier\n\nA chain is: **prompt → LLM → parser**"),
        code("""\
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", 'Classify the support ticket. Return ONLY valid JSON with keys "category" (billing/technical/account/shipping) and "priority" (high/medium/low). No markdown, no explanation.'),
    ("human", "{ticket}"),
])

chain = prompt | llm | JsonOutputParser()

result = chain.invoke({"ticket": "I was charged $49.99 twice this billing cycle"})
print(result)
# → {'category': 'billing', 'priority': 'high'}
"""),
        md("## 2. Single Call, Batch, and Streaming\n\nAll three modes use the same chain — only the calling method changes."),
        code("""\
# Single call
result = chain.invoke({"ticket": "API returns 500 errors in production"})
print("Single:", result)

# Batch (parallel calls — much faster than a loop)
tickets_batch = [
    {"ticket": "I cannot reset my password"},
    {"ticket": "My package has not arrived after 2 weeks"},
    {"ticket": "Dark mode does not persist between sessions"},
]
results = chain.batch(tickets_batch)
for t, r in zip(tickets_batch, results):
    print(f"  {t['ticket'][:40]!r:45} → {r}")
"""),
        code("""\
# Streaming — tokens arrive as they are generated
from langchain_core.output_parsers import StrOutputParser

str_chain = prompt | llm | StrOutputParser()

print("Streaming: ", end="")
for chunk in str_chain.stream({"ticket": "My account was hacked"}):
    print(chunk, end="", flush=True)
print()
"""),
        md("## 3. `RunnablePassthrough` — keep the original input\n\nWhen you need both the original input AND some derived value in the same prompt:"),
        code("""\
from langchain_core.runnables import RunnablePassthrough

context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the support question using the provided context. Be concise."),
    ("human", "Context: {context}\\n\\nQuestion: {question}"),
])

# We'll build a real RAG chain in the next notebook
# For now just see how RunnablePassthrough threads values through
print("RunnablePassthrough lets you keep the original input alongside transformed values")
print("Usage: {context: retriever | format_docs, question: RunnablePassthrough()}")
"""),
        md("## 4. `RunnableLambda` — wrap any Python function"),
        code("""\
from langchain_core.runnables import RunnableLambda

# Preprocess input before sending to LLM
preprocess = RunnableLambda(lambda x: {"ticket": x.strip().lower()})
preprocess_chain = preprocess | chain

result = preprocess_chain.invoke("  I WAS CHARGED TWICE  ")
print("Preprocessed result:", result)
"""),
        md("## Summary\n\n- LCEL uses `|` to compose `Runnable` components\n- Every chain supports `.invoke()`, `.stream()`, `.batch()`\n- `RunnablePassthrough` threads the input unchanged\n- `RunnableLambda` wraps any Python function\n- The same chain works for single calls, streaming, and batch"),
    ]
    save(n, os.path.join(BASE, "learning", "01_lcel_basics.ipynb"))


# ---------------------------------------------------------------------------
# Learning 02 — RAG Pipeline
# ---------------------------------------------------------------------------

def make_learning_02():
    n = nb()
    n.cells = [
        md("# Learning 02: RAG Pipeline with LangChain\n\nRetrieval-Augmented Generation: embed documents into a vector store, retrieve relevant ones at query time, and generate a grounded answer."),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## 1. Build the Vector Store"),
        code("""\
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Extract texts and metadata from the knowledge base
texts = [article["content"] for article in knowledge_base]
metadatas = [{"id": article["id"], "title": article["title"]} for article in knowledge_base]

# Embed and index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"✓ Indexed {len(texts)} articles")
"""),
        md("## 2. Test the Retriever"),
        code("""\
docs = retriever.invoke("How long does a refund take?")
for i, doc in enumerate(docs):
    print(f"[{i+1}] {doc.metadata['title']}")
    print(f"    {doc.page_content[:120]}...")
    print()
"""),
        md("## 3. Build the RAG Chain"),
        code("""\
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\\n\\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )

rag_prompt = ChatPromptTemplate.from_template(\"\"\"\\
Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:\"\"\")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How long does a refund take?")
print(answer)
"""),
        md("## 4. Evaluate Against Test Questions"),
        code("""\
print("RAG Evaluation\\n" + "="*50)
for q in test_questions:
    answer = rag_chain.invoke(q["question"])
    hits = [kw for kw in q["expected_keywords"] if kw.lower() in answer.lower()]
    score = len(hits) / len(q["expected_keywords"])
    status = "✓" if score >= 0.5 else "✗"
    print(f"{status} Q{q['id']}: {q['question'][:50]!r}")
    print(f"   Score: {score:.0%} | Found: {hits}")
    print()
"""),
        md("## 5. Persist the Index (Optional)"),
        code("""\
# Save to disk — avoids re-embedding on next run
vectorstore.save_local("faiss_index")
print("✓ Saved to faiss_index/")

# Load on next run:
# vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
"""),
        md("## Summary\n\n- `FAISS.from_texts()` embeds and indexes documents\n- `as_retriever()` turns the vectorstore into an LCEL-compatible Runnable\n- RAG chain: `{context: retriever | format_docs, question: RunnablePassthrough()} | prompt | llm | parser`\n- Evaluate with expected keywords per question"),
    ]
    save(n, os.path.join(BASE, "learning", "02_rag_pipeline.ipynb"))


# ---------------------------------------------------------------------------
# Learning 03 — Agents and Tools
# ---------------------------------------------------------------------------

def make_learning_03():
    n = nb()
    n.cells = [
        md("# Learning 03: Agents and Tools\n\nAgents use an LLM to dynamically decide which tools to call. Unlike chains (fixed sequence), agents loop until the task is complete.\n\n**LangChain 1.x API**: Use `create_agent` from `langchain.agents`. The agent accepts messages and returns messages."),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## 1. Define Tools with `@tool`\n\nThe **docstring** is what the agent reads to decide when to use the tool. Make it specific."),
        code("""\
import json
from langchain_core.tools import tool

@tool
def classify_ticket(ticket_text: str) -> str:
    \"\"\"Classify a customer support ticket by category and priority.
    Use this when you need to determine what type of issue a ticket describes
    and how urgent it is.

    Args:
        ticket_text: The full text of the support ticket.

    Returns:
        JSON string with 'category' (billing/technical/account/shipping)
        and 'priority' (high/medium/low) fields.
    \"\"\"
    prompt = ChatPromptTemplate.from_messages([
        ("system", 'Return ONLY valid JSON: {{"category": "billing|technical|account|shipping", "priority": "high|medium|low"}}'),
        ("human", "{ticket}"),
    ])
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"ticket": ticket_text})
    return json.dumps(result)

# Test the tool directly
print(classify_ticket.invoke({"ticket_text": "I was charged twice this month"}))
"""),
        code("""\
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

texts = [article["content"] for article in knowledge_base]
metadatas = [{"id": a["id"], "title": a["title"]} for a in knowledge_base]
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def search_knowledge_base(query: str) -> str:
    \"\"\"Search the company knowledge base for policy and troubleshooting information.
    Use this when the customer asks about refunds, shipping, API rate limits, account
    issues, password reset, cancellation, team management, or any company procedures.

    Args:
        query: The customer's question or search terms.

    Returns:
        Relevant knowledge base articles as formatted text.
    \"\"\"
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant articles found."
    return "\\n\\n".join(
        f"[Article {i+1}: {doc.metadata['title']}]\\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

print(search_knowledge_base.invoke({"query": "API rate limit 429 error"}))
"""),
        code("""\
@tool
def escalate_to_human(reason: str, priority: str) -> str:
    \"\"\"Escalate a support ticket to a human agent.
    Use this when: (1) the issue is urgent (priority=high) and involves account
    compromise, data loss, or financial damage; (2) you cannot find the answer
    in the knowledge base; (3) the customer is frustrated and needs human empathy.

    Args:
        reason: Brief explanation of why escalation is needed.
        priority: The urgency level (high/medium/low).

    Returns:
        Confirmation that the ticket has been escalated.
    \"\"\"
    return f"✓ Ticket escalated to human agent (priority={priority}). Reason: {reason}"

tools = [classify_ticket, search_knowledge_base, escalate_to_human]
print(f"✓ Defined {len(tools)} tools: {[t.name for t in tools]}")
"""),
        md("## 2. Build the Agent\n\nIn **LangChain 1.x**, use `create_agent` from `langchain.agents`.\nIt takes `(model, tools, system_prompt=...)` and returns a LangGraph agent.\n\nThe agent accepts `{\"messages\": [...]}` and returns `{\"messages\": [...]}`."),
        code("""\
from langchain.agents import create_agent

agent = create_agent(
    llm,
    tools,
    system_prompt=\"\"\"\\
You are a customer support agent for a SaaS company.
Your job is to help customers by:
1. Classifying their issue
2. Searching the knowledge base for relevant information
3. Providing a clear, helpful answer
4. Escalating to a human if the issue is urgent or you cannot help

Always classify the ticket first, then search for information.\"\"\",
)
print("✓ Agent ready")
"""),
        md("## 3. Run the Agent\n\nInvoke with a messages list. The final answer is in the last message."),
        code("""\
# Easy question — should search KB and answer directly
result = agent.invoke({
    "messages": [{"role": "user", "content": "Why is my API returning 429 errors?"}]
})

final_answer = result["messages"][-1].content
print("Final answer:")
print(final_answer)
"""),
        code("""\
# Urgent — should classify as high priority and escalate
result = agent.invoke({
    "messages": [{"role": "user", "content": "My account was hacked! I see login attempts from Russia. Lock it now!"}]
})
print("Final answer:", result["messages"][-1].content)
"""),
        code("""\
# Policy question — should search KB
result = agent.invoke({
    "messages": [{"role": "user", "content": "What happens to my data if I cancel my subscription?"}]
})
print("Final answer:", result["messages"][-1].content)
"""),
        md("## 4. Inspect Tool Calls\n\nAll messages (user, tool calls, tool results, assistant) are in `result[\"messages\"]`."),
        code("""\
from langchain_core.messages import ToolMessage, AIMessage

result = agent.invoke({
    "messages": [{"role": "user", "content": "I ordered a laptop stand but received a mouse pad. Order #45123."}]
})

print("All messages:")
for msg in result["messages"]:
    name = type(msg).__name__
    content = str(msg.content)[:80] if msg.content else ""
    # AIMessage may have tool_calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  [{name}] → calls tool: {tc['name']}({tc['args']})")
    elif isinstance(msg, ToolMessage):
        print(f"  [{name}] tool={msg.name}: {content}...")
    else:
        print(f"  [{name}]: {content}")

print("\\nFinal answer:", result["messages"][-1].content)
"""),
        md("## Summary\n\n- `@tool` decorator converts a function into an agent tool; the docstring guides the agent\n- `create_agent(llm, tools, system_prompt=...)` builds the agent (LangChain 1.x / LangGraph)\n- Agent accepts `{\"messages\": [...]}` and returns `{\"messages\": [...]}`\n- Final answer: `result[\"messages\"][-1].content`\n- Inspect tool calls by filtering `ToolMessage` instances from messages"),
    ]
    save(n, os.path.join(BASE, "learning", "03_agents_and_tools.ipynb"))


# ---------------------------------------------------------------------------
# Task 01 — LCEL Classification Chain
# ---------------------------------------------------------------------------

PRIORITY_DEFINITIONS = """\
Priority rules:
- high: financial damage (duplicate charge, unauthorized charge), system down affecting production, security compromise, data loss
- medium: non-critical bugs with workarounds, missing documents (invoices), degraded performance, account access issues
- low: informational/how-to questions, cosmetic bugs, feature preferences, general inquiries
"""

def make_task_01():
    task = nb()
    task.cells = [
        md("# Task 01: LCEL Classification Chain\n\nBuild a ticket classification chain using LCEL. The chain should take a ticket text and return `{\"category\": ..., \"priority\": ...}`."),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## Part 1: Build the Classification Chain\n\nCreate a `chain` using LCEL that:\n- Takes `{\"ticket\": <text>}` as input\n- Returns a dict with `category` (billing/technical/account/shipping) and `priority` (high/medium/low)\n\nHint: add **priority definitions** to your system prompt — the model's default sense of urgency differs from the dataset labels."),
        code("""\
# YOUR CODE HERE
# Define: prompt, chain
# chain should accept {"ticket": str} and return dict

"""),
        code("""\
# TEST — Structural check (no API call)
assert 'chain' in dir() or 'chain' in locals(), "Define a variable named 'chain'"
assert hasattr(chain, 'invoke'), "chain must have .invoke() method"
print("✓ chain has .invoke()")
"""),
        md("## Part 2: Batch Classification\n\nImplement `classify_all(chain, tickets) -> list[dict]` that classifies all 20 tickets using `.batch()`."),
        code("""\
# YOUR CODE HERE
def classify_all(chain, tickets):
    pass

"""),
        code("""\
# TEST — Run classification on all 20 tickets
results = classify_all(chain, tickets)

assert isinstance(results, list), "classify_all must return a list"
assert len(results) == len(tickets), f"Expected {len(tickets)} results, got {len(results)}"

VALID_CATEGORIES = {"billing", "technical", "account", "shipping"}
VALID_PRIORITIES = {"high", "medium", "low"}

for i, r in enumerate(results):
    assert isinstance(r, dict), f"Result {i} must be a dict"
    assert "category" in r, f"Result {i} missing 'category'"
    assert "priority" in r, f"Result {i} missing 'priority'"
    assert r["category"] in VALID_CATEGORIES, f"Result {i}: invalid category {r['category']!r}"
    assert r["priority"] in VALID_PRIORITIES, f"Result {i}: invalid priority {r['priority']!r}"

print(f"✓ All {len(results)} results have valid format")
"""),
        code("""\
# TEST — Accuracy check (real LLM output)
correct_cat = sum(
    1 for r, t in zip(results, tickets)
    if r.get("category") == t["category"]
)
correct_pri = sum(
    1 for r, t in zip(results, tickets)
    if r.get("priority") == t["priority"]
)

cat_acc = correct_cat / len(tickets)
pri_acc = correct_pri / len(tickets)

print(f"Category accuracy: {cat_acc:.0%} ({correct_cat}/{len(tickets)})")
print(f"Priority accuracy:  {pri_acc:.0%} ({correct_pri}/{len(tickets)})")

# Show mistakes
for r, t in zip(results, tickets):
    if r.get("category") != t["category"] or r.get("priority") != t["priority"]:
        cat_mark = "✓" if r.get("category") == t["category"] else "✗"
        pri_mark = "✓" if r.get("priority") == t["priority"] else "✗"
        print(f"  ID {t['id']}: cat={cat_mark}{r.get('category')} pri={pri_mark}{r.get('priority')}  (expected: {t['category']}/{t['priority']})")

assert cat_acc >= 0.80, f"Category accuracy {cat_acc:.0%} < 80% — improve your prompt"
assert pri_acc >= 0.60, f"Priority accuracy {pri_acc:.0%} < 60% — add priority definitions to your prompt"
print("\\n✓ Accuracy targets met!")
"""),
        md("## Part 3: Chain-of-Thought Variant\n\nCreate `cot_chain` that includes a `reasoning` field explaining the classification before giving the final answer. This often improves accuracy."),
        code("""\
# YOUR CODE HERE
# Define cot_chain — same interface as chain but returns dict with "reasoning", "category", "priority"

"""),
        code("""\
# TEST — CoT format check
sample_result = cot_chain.invoke({"ticket": "I was charged twice this billing cycle."})

assert isinstance(sample_result, dict), "cot_chain must return a dict"
assert "category" in sample_result, "Missing 'category'"
assert "priority" in sample_result, "Missing 'priority'"
assert "reasoning" in sample_result, "Missing 'reasoning' field — add chain-of-thought"
assert len(sample_result["reasoning"]) > 20, "'reasoning' should be a real explanation"

print("✓ CoT chain format correct")
print(f"  Reasoning: {sample_result['reasoning'][:80]}...")
print(f"  Category: {sample_result['category']}, Priority: {sample_result['priority']}")
"""),
    ]
    save(task, os.path.join(BASE, "tasks", "task_01_lcel_chain.ipynb"))

    sol = nb()
    sol.cells = [
        md("# Task 01 Solution: LCEL Classification Chain"),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## Part 1: Classification Chain"),
        code("""\
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a support ticket classifier. "
        "Return ONLY valid JSON with exactly two keys:\\n"
        '  "category": one of billing | technical | account | shipping\\n'
        '  "priority": one of high | medium | low\\n\\n'
        "Priority definitions:\\n"
        "  high: financial damage (duplicate/unauthorized charge), production system down, "
        "security compromise (hacked account), data loss\\n"
        "  medium: non-critical bugs with workarounds, missing documents (invoices), "
        "degraded performance, account access issues\\n"
        "  low: informational/how-to questions, cosmetic bugs, feature preferences\\n\\n"
        "No markdown, no code blocks, no explanation."
    )),
    ("human", "{ticket}"),
])

chain = prompt | llm | JsonOutputParser()

test = chain.invoke({"ticket": "I was charged twice"})
print("Test result:", test)
"""),
        code("""\
assert hasattr(chain, 'invoke')
print("✓ chain has .invoke()")
"""),
        md("## Part 2: Batch Classification"),
        code("""\
def classify_all(chain, tickets):
    inputs = [{"ticket": t["text"]} for t in tickets]
    return chain.batch(inputs)

results = classify_all(chain, tickets)
"""),
        code("""\
VALID_CATEGORIES = {"billing", "technical", "account", "shipping"}
VALID_PRIORITIES = {"high", "medium", "low"}

for i, r in enumerate(results):
    assert isinstance(r, dict)
    assert "category" in r
    assert "priority" in r
    assert r["category"] in VALID_CATEGORIES
    assert r["priority"] in VALID_PRIORITIES

print(f"✓ All {len(results)} results valid")
"""),
        code("""\
correct_cat = sum(1 for r, t in zip(results, tickets) if r.get("category") == t["category"])
correct_pri = sum(1 for r, t in zip(results, tickets) if r.get("priority") == t["priority"])

cat_acc = correct_cat / len(tickets)
pri_acc = correct_pri / len(tickets)
print(f"Category: {cat_acc:.0%}, Priority: {pri_acc:.0%}")

for r, t in zip(results, tickets):
    if r.get("category") != t["category"] or r.get("priority") != t["priority"]:
        print(f"  ID {t['id']}: got {r} | expected {t['category']}/{t['priority']}")

assert cat_acc >= 0.80, f"Category {cat_acc:.0%} < 80%"
assert pri_acc >= 0.60, f"Priority {pri_acc:.0%} < 60%"
print("✓ Accuracy targets met!")
"""),
        md("## Part 3: CoT Chain"),
        code("""\
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a support ticket classifier. Think step by step, then classify.\\n"
        'Return ONLY valid JSON: {{"reasoning": "brief explanation", '
        '"category": "billing|technical|account|shipping", "priority": "high|medium|low"}}'
    )),
    ("human", "{ticket}"),
])

cot_chain = cot_prompt | llm | JsonOutputParser()

sample_result = cot_chain.invoke({"ticket": "I was charged twice this billing cycle."})
print(f"Reasoning: {sample_result['reasoning'][:80]}...")
print(f"Category: {sample_result['category']}, Priority: {sample_result['priority']}")
"""),
        code("""\
assert "reasoning" in sample_result
assert len(sample_result["reasoning"]) > 20
print("✓ CoT chain format correct")
"""),
    ]
    save(sol, os.path.join(BASE, "solutions", "task_01_lcel_chain_solution.ipynb"))


# ---------------------------------------------------------------------------
# Task 02 — RAG Pipeline
# ---------------------------------------------------------------------------

def make_task_02():
    task = nb()
    task.cells = [
        md("# Task 02: RAG Pipeline\n\nBuild a Retrieval-Augmented Generation pipeline over the company knowledge base. The pipeline should answer customer questions grounded in the knowledge base articles."),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## Part 1: Build the Vector Store\n\nCreate a FAISS vector store from the knowledge base articles using OpenAI embeddings. Create a retriever that returns the top 3 most relevant documents."),
        code("""\
# YOUR CODE HERE
# Define: embeddings_model, vectorstore, retriever

"""),
        code("""\
# TEST — Structural check (no API call)
assert 'retriever' in dir() or 'retriever' in locals(), "Define a variable named 'retriever'"
assert hasattr(retriever, 'invoke'), "retriever must have .invoke() method"
print("✓ retriever defined and callable")
"""),
        code("""\
# TEST — Retriever returns relevant results
refund_docs = retriever.invoke("How long does a refund take?")

assert len(refund_docs) > 0, "Retriever returned no documents"
assert len(refund_docs) <= 5, "Retrieved too many documents (expected k=3)"

titles = [doc.metadata.get("title", "") for doc in refund_docs]
print(f"Retrieved {len(refund_docs)} docs: {titles}")
assert any("refund" in t.lower() or "policy" in t.lower() for t in titles), \
    "Refund article not retrieved for refund query — check your index"
print("✓ Retriever returns relevant documents")
"""),
        md("## Part 2: Build the RAG Chain\n\nCreate a `rag_chain` using LCEL that:\n- Takes a question string as input\n- Retrieves relevant articles\n- Generates a grounded answer\n\nFormat retrieved documents before inserting them into the prompt."),
        code("""\
# YOUR CODE HERE
# Define: format_docs(docs) -> str, rag_prompt, rag_chain

"""),
        code("""\
# TEST — RAG chain produces answers
answer = rag_chain.invoke("How long does a refund take to process?")

assert isinstance(answer, str), "rag_chain must return a string"
assert len(answer) > 20, "Answer too short — the chain may not be working"

answer_lower = answer.lower()
found = any(kw in answer_lower for kw in ["5-7", "business days", "refund"])
print(f"Answer: {answer[:200]}")
assert found, "Answer doesn't mention refund timeline — check your retriever and prompt"
print("✓ RAG chain produces relevant answers")
"""),
        md("## Part 3: Evaluate Against All Test Questions\n\nRun the RAG chain over all 5 test questions and check keyword coverage."),
        code("""\
# YOUR CODE HERE
# Implement evaluate_rag(rag_chain, test_questions) -> float
# Returns average keyword coverage (0.0 to 1.0)

"""),
        code("""\
# TEST — Evaluation over all test questions
coverage = evaluate_rag(rag_chain, test_questions)

print(f"\\nOverall keyword coverage: {coverage:.0%}")

for q in test_questions:
    answer = rag_chain.invoke(q["question"]).lower()
    hits = [kw for kw in q["expected_keywords"] if kw.lower() in answer]
    score = len(hits) / len(q["expected_keywords"])
    status = "✓" if score >= 0.5 else "✗"
    print(f"  {status} Q{q['id']}: {q['question'][:50]!r} — {score:.0%} ({hits})")

assert coverage >= 0.60, f"RAG quality {coverage:.0%} < 60% — improve retrieval or prompt"
print(f"\\n✓ RAG evaluation passed ({coverage:.0%} coverage)")
"""),
    ]
    save(task, os.path.join(BASE, "tasks", "task_02_rag_pipeline.ipynb"))

    sol = nb()
    sol.cells = [
        md("# Task 02 Solution: RAG Pipeline"),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## Part 1: Vector Store"),
        code("""\
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

texts = [article["content"] for article in knowledge_base]
metadatas = [{"id": a["id"], "title": a["title"]} for a in knowledge_base]

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"✓ Indexed {len(texts)} articles")
"""),
        code("""\
assert hasattr(retriever, 'invoke')
print("✓ retriever defined")

refund_docs = retriever.invoke("How long does a refund take?")
titles = [doc.metadata.get("title", "") for doc in refund_docs]
print(f"Retrieved: {titles}")
assert any("refund" in t.lower() or "policy" in t.lower() for t in titles)
print("✓ Retriever returns relevant documents")
"""),
        md("## Part 2: RAG Chain"),
        code("""\
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\\n\\n".join(
        f"[{i+1}] {doc.metadata['title']}\\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

rag_prompt = ChatPromptTemplate.from_template(\"\"\"\\
Answer the customer's question using ONLY the provided knowledge base articles.
If the answer is not in the articles, say "I don't have that information."
Be specific and cite relevant details.

Knowledge Base Articles:
{context}

Customer Question: {question}

Answer:\"\"\")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How long does a refund take to process?")
print(answer)
"""),
        code("""\
assert isinstance(answer, str) and len(answer) > 20
assert any(kw in answer.lower() for kw in ["5-7", "business days", "refund"])
print("✓ RAG chain test passed")
"""),
        md("## Part 3: Evaluation"),
        code("""\
def evaluate_rag(rag_chain, test_questions):
    scores = []
    for q in test_questions:
        answer = rag_chain.invoke(q["question"]).lower()
        hits = sum(1 for kw in q["expected_keywords"] if kw.lower() in answer)
        scores.append(hits / len(q["expected_keywords"]))
    return sum(scores) / len(scores)

coverage = evaluate_rag(rag_chain, test_questions)
print(f"Coverage: {coverage:.0%}")
"""),
        code("""\
for q in test_questions:
    answer = rag_chain.invoke(q["question"]).lower()
    hits = [kw for kw in q["expected_keywords"] if kw.lower() in answer]
    score = len(hits) / len(q["expected_keywords"])
    status = "✓" if score >= 0.5 else "✗"
    print(f"  {status} Q{q['id']}: {score:.0%} ({hits})")

assert coverage >= 0.60
print(f"\\n✓ RAG evaluation passed ({coverage:.0%})")
"""),
    ]
    save(sol, os.path.join(BASE, "solutions", "task_02_rag_pipeline_solution.ipynb"))


# ---------------------------------------------------------------------------
# Task 03 — LangChain Support Agent (LangChain 1.x / LangGraph API)
# ---------------------------------------------------------------------------

def make_task_03():
    task = nb()
    task.cells = [
        md("# Task 03: LangChain Support Agent\n\nBuild a full customer support agent using `create_agent` from LangChain 1.x. The agent must use multiple tools to autonomously handle support queries.\n\n**API**: `create_agent(llm, tools, system_prompt=...)` → `.invoke({\"messages\": [...]})`"),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## Part 1: Define the Tools\n\nYou need three tools:\n\n1. **`classify_ticket`** — calls an LCEL chain to classify category + priority\n2. **`search_knowledge_base`** — RAG over `knowledge_base.json` using FAISS\n3. **`escalate_to_human`** — logs escalation (no LLM call needed)\n\nRemember: the **docstring is the agent's guide** — write specific, actionable descriptions."),
        code("""\
# YOUR CODE HERE
# Implement all three tools using @tool decorator

"""),
        code("""\
# TEST — Tool structure checks (no API call)
for tool_fn in [classify_ticket, search_knowledge_base, escalate_to_human]:
    assert hasattr(tool_fn, 'name'), f"{tool_fn} must be a LangChain tool"
    assert hasattr(tool_fn, 'description'), f"{tool_fn} must have a description"
    assert len(tool_fn.description) > 50, f"{tool_fn.name} docstring too short — be specific"
    print(f"  ✓ {tool_fn.name!r}: {tool_fn.description[:60]}...")

print("\\n✓ All three tools defined correctly")
"""),
        code("""\
# TEST — Tools callable with invoke
import json

result = classify_ticket.invoke({"ticket_text": "I was charged twice this billing cycle"})
assert isinstance(result, str), "classify_ticket must return a string"
parsed = json.loads(result)
assert "category" in parsed and "priority" in parsed, "classify_ticket must return JSON with category and priority"
print(f"✓ classify_ticket works: {parsed}")

kb_result = search_knowledge_base.invoke({"query": "refund policy"})
assert isinstance(kb_result, str) and len(kb_result) > 50, "search_knowledge_base must return non-empty text"
print(f"✓ search_knowledge_base works: {kb_result[:80]}...")

esc_result = escalate_to_human.invoke({"reason": "account compromise", "priority": "high"})
assert isinstance(esc_result, str)
print(f"✓ escalate_to_human works: {esc_result}")
"""),
        md("## Part 2: Build the Agent\n\nCreate an `agent` using `create_agent` with:\n- A system prompt instructing: classify → search → respond → escalate if urgent\n- All three tools\n\nInvoke with `{\"messages\": [{\"role\": \"user\", \"content\": \"...\"}]}`"),
        code("""\
# YOUR CODE HERE
# from langchain.agents import create_agent
# Define: agent

"""),
        code("""\
# TEST — Agent has .invoke() method
assert hasattr(agent, 'invoke'), "agent must have .invoke() method"

# Quick structural test: invoke must return dict with 'messages'
test_result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}]
})
assert "messages" in test_result, "agent.invoke() must return dict with 'messages' key"
assert len(test_result["messages"]) > 0, "messages list must not be empty"
print("✓ Agent structure correct")
print(f"  Final message type: {type(test_result['messages'][-1]).__name__}")
"""),
        md("## Part 3: Run the Agent on Real Queries"),
        code("""\
def ask_agent(agent, question):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content

# Test 1: Policy question — agent should search KB
print("=" * 60)
print("Test 1: API rate limit question")
print("=" * 60)
answer1 = ask_agent(agent, "Why is my API returning 429 errors? How do I fix this?")
print("Answer:", answer1)
"""),
        code("""\
# TEST — Policy answer quality
assert len(answer1) > 50, "Answer too short"
found_kw = any(kw in answer1.lower() for kw in ["429", "rate limit", "retry", "too many"])
assert found_kw, "Answer should mention rate limiting — did the agent search the KB?"
print("✓ API rate limit answer is informative")
"""),
        code("""\
# Test 2: Urgent security issue — agent should escalate or provide security steps
print("=" * 60)
print("Test 2: Compromised account (high priority)")
print("=" * 60)
answer2 = ask_agent(agent, "My account was hacked! I see login attempts from Russia. I need this fixed IMMEDIATELY.")
print("Answer:", answer2)
"""),
        code("""\
# TEST — Urgent issue handling
assert len(answer2) > 30, "Answer too short for urgent issue"
handled = any(kw in answer2.lower() for kw in ["high", "escalat", "lock", "password", "2fa", "security", "immediately"])
assert handled, "Agent should handle urgent security issue — check tools and system prompt"
print("✓ Urgent issue handled appropriately")
"""),
        code("""\
# Test 3: Cancellation policy
print("=" * 60)
print("Test 3: Cancellation and data retention")
print("=" * 60)
answer3 = ask_agent(agent, "What happens to my data if I cancel my subscription?")
print("Answer:", answer3)
"""),
        code("""\
# TEST — Cancellation answer quality
found_kw = any(kw in answer3.lower() for kw in ["90 days", "90", "export", "delet", "cancel"])
assert found_kw, "Answer should mention data retention period — did the agent find the cancellation article?"
print("✓ Cancellation policy answer is informative")
"""),
        md("## Part 4: Inspect Tool Calls\n\nFilter messages for tool calls to verify the agent actually used tools."),
        code("""\
# YOUR CODE HERE
# Run the agent on a question, then inspect which tools were called
# Hint: filter result["messages"] for ToolMessage instances

"""),
        code("""\
# TEST — Agent uses tools
from langchain_core.messages import ToolMessage

assert 'tool_messages' in dir() or 'tool_messages' in locals(), \
    "Define tool_messages by filtering result['messages'] for ToolMessage instances"

assert len(tool_messages) > 0, \
    "Agent made no tool calls — check your system prompt and tool descriptions"

tool_names_used = [m.name for m in tool_messages]
print(f"✓ Agent made {len(tool_messages)} tool call(s): {tool_names_used}")
assert "search_knowledge_base" in tool_names_used or "classify_ticket" in tool_names_used, \
    "Agent should use at least search_knowledge_base or classify_ticket"
print("✓ Agent used tools correctly")
"""),
    ]
    save(task, os.path.join(BASE, "tasks", "task_03_support_agent.ipynb"))

    sol = nb()
    sol.cells = [
        md("# Task 03 Solution: LangChain Support Agent"),
        md("## Setup"),
        code(LANGCHAIN_SETUP),
        code(FIXTURES_SETUP),
        md("## Part 1: Define Tools"),
        code("""\
import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Build vector store for KB search
texts = [a["content"] for a in knowledge_base]
metadatas = [{"id": a["id"], "title": a["title"]} for a in knowledge_base]
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("✓ Vector store ready")
"""),
        code("""\
@tool
def classify_ticket(ticket_text: str) -> str:
    \"\"\"Classify a customer support ticket by category and priority.
    Use this FIRST for every customer query to determine the type and urgency
    of the issue before deciding how to respond.

    Args:
        ticket_text: The customer's support message.

    Returns:
        JSON string with 'category' (billing/technical/account/shipping)
        and 'priority' (high/medium/low).
    \"\"\"
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", 'Return ONLY valid JSON: {{"category": "billing|technical|account|shipping", "priority": "high|medium|low"}}'),
        ("human", "{ticket}"),
    ])
    chain = classify_prompt | llm | JsonOutputParser()
    try:
        result = chain.invoke({"ticket": ticket_text})
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"category": "technical", "priority": "medium", "error": str(e)})


@tool
def search_knowledge_base(query: str) -> str:
    \"\"\"Search the company knowledge base for policy and troubleshooting information.
    Use this when the customer asks about: refunds, billing, API rate limits, password
    reset, shipping tracking, account security, mobile app issues, subscription
    cancellation, team management, or any company procedures and policies.

    Args:
        query: The customer's question or the topic to search for.

    Returns:
        Relevant knowledge base articles with titles and content.
    \"\"\"
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant articles found in the knowledge base."
    return "\\n\\n".join(
        f"[{doc.metadata['title']}]\\n{doc.page_content}"
        for doc in docs
    )


@tool
def escalate_to_human(reason: str, priority: str) -> str:
    \"\"\"Escalate the support ticket to a human agent.
    Use this when: (1) the issue is high priority involving account compromise,
    data loss, or financial damage; (2) the knowledge base does not contain
    a resolution; (3) the customer explicitly requests to speak with a human.

    Args:
        reason: Brief description of why escalation is needed.
        priority: Urgency level: high, medium, or low.

    Returns:
        Confirmation that a human agent has been assigned.
    \"\"\"
    wait_time = "1 hour" if priority == "high" else "4 hours" if priority == "medium" else "24 hours"
    return (
        f"✓ Ticket escalated to a human support agent (priority={priority}). "
        f"A team member will contact you within {wait_time}. "
        f"Reason: {reason}"
    )

tools = [classify_ticket, search_knowledge_base, escalate_to_human]
print(f"✓ Defined {len(tools)} tools")
"""),
        code("""\
# Structural checks
for t in tools:
    assert hasattr(t, 'name')
    assert len(t.description) > 50
    print(f"  ✓ {t.name}")
print("✓ Tool checks passed")
"""),
        code("""\
# Callable checks
r = classify_ticket.invoke({"ticket_text": "I was charged twice"})
parsed = json.loads(r)
assert "category" in parsed and "priority" in parsed
print(f"✓ classify_ticket: {parsed}")

kb = search_knowledge_base.invoke({"query": "refund policy"})
assert len(kb) > 50
print(f"✓ search_knowledge_base: {kb[:80]}...")

esc = escalate_to_human.invoke({"reason": "account compromise", "priority": "high"})
print(f"✓ escalate_to_human: {esc}")
"""),
        md("## Part 2: Build Agent"),
        code("""\
from langchain.agents import create_agent

agent = create_agent(
    llm,
    tools,
    system_prompt=\"\"\"\\
You are a customer support agent for a SaaS company. Follow this process for every query:

1. CLASSIFY: Use classify_ticket to determine the issue type and urgency
2. SEARCH: Use search_knowledge_base to find relevant policy or troubleshooting info
3. RESPOND: Give a clear, specific answer based on what you found
4. ESCALATE: If the issue is high priority (account compromise, data loss, financial damage)
   OR you cannot find a resolution, use escalate_to_human

Always be helpful, specific, and cite the relevant policy or steps from the knowledge base.\"\"\",
)
print("✓ Agent ready")
"""),
        code("""\
assert hasattr(agent, 'invoke')
test_result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
assert "messages" in test_result
print("✓ Agent structure correct")
"""),
        md("## Part 3: Run Agent"),
        code("""\
def ask_agent(agent, question):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content

answer1 = ask_agent(agent, "Why is my API returning 429 errors? How do I fix this?")
print("Answer:", answer1)
"""),
        code("""\
assert len(answer1) > 50
assert any(kw in answer1.lower() for kw in ["429", "rate limit", "retry", "too many"])
print("✓ API rate limit answer correct")
"""),
        code("""\
answer2 = ask_agent(agent, "My account was hacked! I see login attempts from Russia. URGENT!")
print("Answer:", answer2)
"""),
        code("""\
assert len(answer2) > 30
handled = any(kw in answer2.lower() for kw in ["high", "escalat", "lock", "password", "2fa", "security"])
assert handled
print("✓ Urgent issue handled")
"""),
        code("""\
answer3 = ask_agent(agent, "What happens to my data if I cancel my subscription?")
print("Answer:", answer3)
"""),
        code("""\
assert any(kw in answer3.lower() for kw in ["90", "export", "delet", "cancel"])
print("✓ Cancellation answer correct")
"""),
        md("## Part 4: Inspect Tool Calls"),
        code("""\
from langchain_core.messages import ToolMessage

result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I reset my password if my reset email is not arriving?"}]
})

tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
tool_names_used = [m.name for m in tool_messages]
print(f"Tool calls: {tool_names_used}")
print(f"Answer: {result['messages'][-1].content}")

assert len(tool_messages) > 0
assert "search_knowledge_base" in tool_names_used or "classify_ticket" in tool_names_used
print("✓ Agent used tools correctly")
"""),
    ]
    save(sol, os.path.join(BASE, "solutions", "task_03_support_agent_solution.ipynb"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Module 14 notebooks...\n")
    print("Learning notebooks:")
    make_learning_01()
    make_learning_02()
    make_learning_03()
    print("\nTask + solution notebooks:")
    make_task_01()
    make_task_02()
    make_task_03()
    print("\n✓ All notebooks generated successfully.")
