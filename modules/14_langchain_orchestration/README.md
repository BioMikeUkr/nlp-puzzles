# Module 14: LangChain & Orchestration

## Overview

LangChain is a framework for building LLM-powered applications. This module covers the modern
LangChain stack: LCEL (LangChain Expression Language) for composing chains, building RAG pipelines
as agent tools, and creating agents that use multiple tools to solve complex tasks autonomously.

### Learning Objectives
- Build chains using LCEL pipe operator (`|`)
- Implement RAG pipelines with LangChain components
- Create custom tools using the `@tool` decorator
- Build a full LangChain agent with multiple tools
- Understand when to use LangChain vs custom code

---

## 1. LangChain Expression Language (LCEL)

LCEL is the core abstraction for composing LangChain components. Every component is a `Runnable`
that supports `.invoke()`, `.stream()`, and `.batch()`.

### Basic chain
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the ticket. Return JSON: {{\"category\": \"...\", \"priority\": \"...\"}}"),
    ("human", "{ticket}"),
])

chain = prompt | llm | JsonOutputParser()
result = chain.invoke({"ticket": "I was charged twice"})
# → {"category": "billing", "priority": "high"}
```

### All Runnables support the same interface
```python
# Single call
result = chain.invoke({"ticket": "API is down"})

# Streaming (yields tokens as they arrive)
for chunk in chain.stream({"ticket": "API is down"}):
    print(chunk, end="", flush=True)

# Batch (parallel calls)
results = chain.batch([
    {"ticket": "I was charged twice"},
    {"ticket": "App crashes"},
])
```

### `RunnablePassthrough` — pass inputs through unchanged
```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### `RunnableLambda` — wrap any Python function
```python
from langchain_core.runnables import RunnableLambda

preprocess = RunnableLambda(lambda x: x.strip().lower())
chain = preprocess | prompt | llm | StrOutputParser()
```

---

## 2. Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Basic chat template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

# With conversation history (for agents)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

---

## 3. Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

# String (default)
chain = prompt | llm | StrOutputParser()

# JSON — parses LLM response to dict
chain = prompt | llm | JsonOutputParser()

# Pydantic — validates JSON against a model
from pydantic import BaseModel

class Classification(BaseModel):
    category: str
    priority: str

chain = prompt | llm | PydanticToolsParser(tools=[Classification])
```

---

## 4. RAG Pipeline with LangChain

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Build vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG chain
def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.
If not found in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How long does a refund take?")
```

---

## 5. Tools

Tools are Python functions decorated with `@tool` that agents can choose to call.
The docstring is critical — it tells the agent **when** to use the tool.

```python
from langchain_core.tools import tool

@tool
def classify_ticket(ticket_text: str) -> str:
    """Classify a support ticket by category and priority.
    Use this when you need to determine what type of issue a ticket describes
    and how urgent it is.

    Args:
        ticket_text: The full text of the support ticket.

    Returns:
        JSON string with 'category' and 'priority' fields.
    """
    # ... implementation
    return json.dumps({"category": "billing", "priority": "high"})
```

Key tool design rules:
1. **Docstring is the agent's guide** — be specific about when to use it
2. **Args must be typed** — LangChain uses types to build the tool schema
3. **Return strings** — convert any complex output to string
4. **Handle errors gracefully** — return error strings, don't raise exceptions

---

## 6. LangChain Agents

Agents use tools to autonomously complete tasks. The agent decides which tools to call
and in what order based on the task.

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# System prompt for the agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent. Use the tools available to help customers."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Build agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# Agent executor runs the agent loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,       # prints tool calls and observations
    max_iterations=5,   # prevent infinite loops
)

# Invoke
result = agent_executor.invoke({"input": "My API is returning 429 errors"})
print(result["output"])
```

### Agent execution loop
1. Agent receives the input
2. Agent decides which tool to call (or to answer directly)
3. Tool is called, result returned to agent as "observation"
4. Agent decides next action based on observation
5. Repeat until agent produces final answer

---

## 7. When to Use LangChain vs Custom Code

| | LangChain | Custom Code |
|--|-----------|-------------|
| **Best for** | Complex pipelines, agents, rapid prototyping | Simple chains, full control, performance |
| **Learning curve** | High (framework abstractions) | Low (plain Python) |
| **Flexibility** | Medium (framework constraints) | Full |
| **Debugging** | Harder (many abstractions) | Easier (direct API calls) |
| **Maintenance** | Depends on framework versions | Stable |
| **Community tools** | Rich ecosystem | Build yourself |

**Use LangChain when**:
- Building agents that need to choose between multiple tools
- Rapid prototyping of complex pipelines
- Team is already familiar with the framework

**Use custom code when**:
- Simple classification or extraction (single API call)
- Performance is critical (every abstraction has overhead)
- You need full control over prompts and parsing
- Long-term maintenance stability is important

---

## Module Structure

```
14_langchain_orchestration/
├── README.md              # This file
├── QUESTIONS.md           # 30 interview questions
├── requirements.txt
├── fixtures/input/        # tickets.json, knowledge_base.json, test_questions.json
├── learning/
│   ├── 01_lcel_basics.ipynb
│   ├── 02_rag_pipeline.ipynb
│   └── 03_agents_and_tools.ipynb
├── tasks/
│   ├── task_01_lcel_chain.ipynb
│   ├── task_02_rag_pipeline.ipynb
│   └── task_03_support_agent.ipynb
└── solutions/
    ├── task_01_lcel_chain_solution.ipynb
    ├── task_02_rag_pipeline_solution.ipynb
    └── task_03_support_agent_solution.ipynb
```
