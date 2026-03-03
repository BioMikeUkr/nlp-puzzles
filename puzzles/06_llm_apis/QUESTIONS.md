# Deep Dive Q&A - LLM APIs

> 30 questions covering LLM API usage, Pydantic validation, and production patterns

### Architecture & Design (1-10)

#### Q1: What are structured outputs and why are they critical for production?

**Answer:**

**Structured outputs** force the LLM to return data in a specific JSON schema validated by Pydantic models.

**Problem with raw text:**
```python
# Ask LLM to classify sentiment
response = "The sentiment appears to be positive, confidence around 85%"

# Parsing nightmare:
# - "positive" vs "Positive" vs "pos"
# - "85%" vs "0.85" vs "85"
# - Extra words like "appears to be", "around"
# - What if it says "very positive"?
```

**Solution with structured outputs:**
```python
from pydantic import BaseModel, Field

class SentimentAnalysis(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "I love this!"}],
    response_format=SentimentAnalysis
)

result = response.choices[0].message.parsed
# Guaranteed to have:
# - result.label in ["positive", "negative", "neutral"]
# - result.confidence between 0 and 1
# - result.reasoning as string
```

**Benefits:**
1. **Type safety**: Pydantic validates types automatically
2. **No parsing logic**: Direct access to structured data
3. **Database ready**: Can insert directly into DB
4. **API integration**: Can return as JSON API response
5. **Validation**: Built-in field validation
6. **Documentation**: Schema serves as documentation

**Production impact:**
```python
# Without structured outputs
try:
    # Parse response text
    # Handle variations
    # Validate manually
    # 50+ lines of parsing code
    # Still fragile
except Exception:
    # What went wrong?
    pass

# With structured outputs
result = response.choices[0].message.parsed
# 1 line, guaranteed correct
```

---

#### Q2: How do function calling and structured outputs differ?

**Answer:**

Both enable structured interaction, but serve different purposes:

**Structured Outputs (Response Format):**
- LLM returns structured data **as the response**
- Use for: classification, extraction, analysis
- LLM is the **final step**

```python
class TicketAnalysis(BaseModel):
    category: str
    urgency: int
    summary: str

response = client.beta.chat.completions.parse(
    messages=[{"role": "user", "content": ticket_text}],
    response_format=TicketAnalysis
)

# LLM returns analysis directly
analysis = response.choices[0].message.parsed
save_to_db(analysis)
```

**Function Calling (Tools):**
- LLM decides **which function to call** and **with what arguments**
- Use for: actions, retrieving data, multi-step tasks
- LLM is a **decision maker**, you execute functions

```python
def get_ticket_history(ticket_id: str) -> dict:
    return db.query(f"SELECT * FROM tickets WHERE id={ticket_id}")

tools = [{
    "type": "function",
    "function": {
        "name": "get_ticket_history",
        "description": "Retrieve ticket history from database",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"}
            }
        }
    }
}]

# User asks: "What's the status of ticket #12345?"
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What's status of #12345?"}],
    tools=tools
)

# LLM responds with function call
tool_call = response.choices[0].message.tool_calls[0]
# tool_call.function.name = "get_ticket_history"
# tool_call.function.arguments = '{"ticket_id": "12345"}'

# You execute the function
result = get_ticket_history("12345")

# Send result back to LLM
messages.append({"role": "function", "name": "get_ticket_history", "content": str(result)})
response2 = client.chat.completions.create(messages=messages)
# LLM: "Ticket #12345 is currently open, assigned to John..."
```

**Comparison:**

| Aspect | Structured Outputs | Function Calling |
|--------|-------------------|------------------|
| **Purpose** | Get structured data | Trigger actions |
| **Control flow** | One-shot | Multi-turn |
| **LLM role** | Analyzer | Orchestrator |
| **Your role** | Consumer | Executor |
| **Example** | "Classify this text" | "Book a flight" |

**When to combine both:**
```python
# User: "Analyze ticket #12345 and categorize it"

# Step 1: Function call to get ticket
tool_response = get_ticket(ticket_id="12345")

# Step 2: Structured output to analyze
class Analysis(BaseModel):
    category: str
    priority: int

analysis = client.beta.chat.completions.parse(
    messages=[{"role": "user", "content": tool_response}],
    response_format=Analysis
)
```

---

#### Q3: How do you handle API rate limits in production?

**Answer:**

**Rate Limits (OpenAI):**
- **RPM**: Requests per minute (e.g., 3,500 for gpt-4o-mini)
- **TPM**: Tokens per minute (e.g., 200,000)
- **Batch limits**: Different limits for batch API

**Strategy 1: Exponential Backoff**

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def call_openai(messages):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

# Automatically retries with backoff:
# Attempt 1: immediate
# Attempt 2: wait 4s
# Attempt 3: wait 8s
# Attempt 4: wait 16s
# Attempt 5: wait 32s
```

**Strategy 2: Token Bucket Rate Limiter**

```python
import time
import threading

class RateLimiter:
    def __init__(self, rpm=3000, tpm=150000):
        self.rpm = rpm
        self.tpm = tpm
        self.request_tokens = rpm
        self.token_bucket = tpm
        self.lock = threading.Lock()
        self.last_update = time.time()

    def acquire(self, tokens_needed):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill buckets
            self.request_tokens = min(
                self.rpm,
                self.request_tokens + (self.rpm * elapsed / 60)
            )
            self.token_bucket = min(
                self.tpm,
                self.token_bucket + (self.tpm * elapsed / 60)
            )
            self.last_update = now

            # Check if we can proceed
            if self.request_tokens >= 1 and self.token_bucket >= tokens_needed:
                self.request_tokens -= 1
                self.token_bucket -= tokens_needed
                return True

            # Calculate wait time
            wait_time = max(
                (1 - self.request_tokens) / (self.rpm / 60),
                (tokens_needed - self.token_bucket) / (self.tpm / 60)
            )
            return wait_time

# Usage
limiter = RateLimiter(rpm=3000, tpm=150000)

def safe_call(messages, tokens_needed):
    result = limiter.acquire(tokens_needed)
    if result is not True:
        time.sleep(result)
        result = limiter.acquire(tokens_needed)

    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
```

**Strategy 3: Queue-Based Processing**

```python
from queue import Queue
from threading import Thread
import time

class APIWorker:
    def __init__(self, rpm=3000):
        self.queue = Queue()
        self.rpm = rpm
        self.interval = 60 / rpm  # seconds between requests

        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def _process_queue(self):
        while True:
            messages, callback = self.queue.get()

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                callback(response, None)
            except Exception as e:
                callback(None, e)

            time.sleep(self.interval)
            self.queue.task_done()

    def submit(self, messages, callback):
        self.queue.put((messages, callback))

# Usage
worker = APIWorker(rpm=3000)

def handle_response(response, error):
    if error:
        print(f"Error: {error}")
    else:
        print(response.choices[0].message.content)

worker.submit(messages, handle_response)
```

**Strategy 4: Batch API for Non-Urgent Requests**

```python
# For non-real-time processing (50% cheaper, 24h turnaround)
from openai import OpenAI

client = OpenAI()

# Create batch file
batch_requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Classify: {text}"}]
        }
    }
    for i, text in enumerate(texts)
]

# Upload and submit batch
batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Check status later
status = client.batches.retrieve(batch.id)
```

**Best Practices:**
1. Count tokens **before** calling API
2. Use exponential backoff for sporadic traffic
3. Use rate limiter for high-volume production
4. Monitor usage via OpenAI dashboard
5. Set up alerts for approaching limits
6. Consider batch API for overnight processing

---

#### Q4: How do you count tokens and estimate costs?

**Answer:**

**Why token counting matters:**
- API pricing is per token, not per character
- Models have context limits (e.g., 128K for gpt-4o)
- Need to estimate costs before calling API

**Token counting with tiktoken:**

```python
import tiktoken

# Get encoding for specific model
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# Count tokens in text
text = "How many tokens is this?"
tokens = encoding.encode(text)
print(f"Token count: {len(tokens)}")  # ~6 tokens

# Rough estimate: 1 token ≈ 4 characters (English)
# But this varies by language and content!

# Count tokens in messages
def count_message_tokens(messages, model="gpt-4o-mini"):
    """
    Count tokens in chat messages

    Note: Every message has overhead tokens for formatting:
    - <|im_start|>role<|im_sep|>content<|im_end|>
    """
    encoding = tiktoken.encoding_for_model(model)

    tokens_per_message = 3  # Overhead per message
    tokens_per_name = 1     # If name field present

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # Every reply is primed with assistant
    return num_tokens

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

tokens = count_message_tokens(messages)
print(f"Total tokens: {tokens}")
```

**Cost estimation:**

```python
class CostEstimator:
    # Prices per 1M tokens (as of 2024)
    PRICES = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)

    def estimate_cost(self, messages, max_tokens=1000):
        """Estimate cost for API call"""
        input_tokens = count_message_tokens(messages, self.model)
        output_tokens = max_tokens  # Worst case

        input_cost = (input_tokens / 1_000_000) * self.PRICES[self.model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.PRICES[self.model]["output"]

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }

# Usage
estimator = CostEstimator("gpt-4o-mini")

messages = [
    {"role": "user", "content": "Write a 500-word essay about Python"}
]

cost = estimator.estimate_cost(messages, max_tokens=700)
print(f"Estimated cost: ${cost['total_cost']:.4f}")
# ~$0.0002 for gpt-4o-mini
```

**Track actual usage:**

```python
class APIClient:
    def __init__(self):
        self.client = OpenAI()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0

    def chat(self, messages, model="gpt-4o-mini"):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )

        # Track usage
        usage = response.usage
        self.total_input_tokens += usage.prompt_tokens
        self.total_output_tokens += usage.completion_tokens

        # Calculate cost
        prices = CostEstimator.PRICES[model]
        cost = (
            (usage.prompt_tokens / 1_000_000) * prices["input"] +
            (usage.completion_tokens / 1_000_000) * prices["output"]
        )
        self.total_cost += cost

        return response, cost

    def get_stats(self):
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost
        }

# Usage
client = APIClient()

response, cost = client.chat(messages)
print(f"This call cost: ${cost:.6f}")

# After many calls
stats = client.get_stats()
print(f"Total spent: ${stats['total_cost']:.2f}")
```

**Cost optimization tips:**

```python
# 1. Use cheaper models when possible
# gpt-4o-mini is 16x cheaper than gpt-4o

# 2. Reduce max_tokens
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=100  # Limit output length
)

# 3. Use system message to enforce brevity
messages = [
    {"role": "system", "content": "Answer in 1-2 sentences max."},
    {"role": "user", "content": question}
]

# 4. Cache responses (avoid duplicate calls)
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classify(text: str) -> str:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content

# 5. Batch similar requests
texts = ["text1", "text2", "text3"]
prompt = f"Classify each:\n1. {texts[0]}\n2. {texts[1]}\n3. {texts[2]}"
# Process 3 items in 1 API call
```

---

#### Q5: What are the differences between OpenAI models and when to use each?

**Answer:**

| Model | Use Case | Cost | Speed | Context | Intelligence |
|-------|----------|------|-------|---------|--------------|
| **gpt-4o** | Complex reasoning | High | Medium | 128K | Highest |
| **gpt-4o-mini** | Most tasks | Low | Fast | 128K | High |
| **o1-preview** | Hard problems | Highest | Slow | 128K | Reasoning |
| **gpt-3.5-turbo** | Simple tasks | Medium | Fastest | 16K | Good |

**Detailed breakdown:**

**gpt-4o (Omni):**
```python
# Use for:
# - Complex analysis requiring deep reasoning
# - Multi-step problem solving
# - Code generation with intricate logic
# - When accuracy matters more than cost

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Analyze this complex legal document and extract key clauses..."
    }]
)

# Cost: $2.50 per 1M input tokens
# Best for: High-value tasks where errors are expensive
```

**gpt-4o-mini (Recommended for most):**
```python
# Use for:
# - Classification, extraction
# - Summarization
# - Most production tasks
# - High-volume processing

# 85% of use cases should use this
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Classify this support ticket..."
    }]
)

# Cost: $0.15 per 1M input tokens (16x cheaper than gpt-4o)
# Performance: Nearly as good as gpt-4o for most tasks
```

**o1-preview (Reasoning model):**
```python
# Use for:
# - Math, science, coding challenges
# - Problems requiring chain-of-thought
# - Research and analysis

response = client.chat.completions.create(
    model="o1-preview",
    messages=[{
        "role": "user",
        "content": "Prove that there are infinitely many prime numbers"
    }]
)

# Cost: Highest
# Note: Spends more tokens "thinking" before answering
# Slower but more accurate on hard problems
```

**gpt-3.5-turbo (Legacy):**
```python
# Use for:
# - Extremely simple tasks
# - Highest volume, lowest latency needs
# - When 16K context is enough

# Increasingly rare to use (gpt-4o-mini is better value)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
```

**Decision tree:**

```python
def choose_model(task_type, budget, latency_requirement):
    if task_type == "math_or_hard_reasoning":
        return "o1-preview"

    if budget == "unlimited" and task_type == "complex_analysis":
        return "gpt-4o"

    if latency_requirement == "ultra_low" and task_type == "simple":
        return "gpt-3.5-turbo"

    # Default: best value for money
    return "gpt-4o-mini"
```

**Real-world examples:**

```python
# Support ticket classification
model = "gpt-4o-mini"  # ✓ Fast, cheap, accurate enough

# Legal contract review
model = "gpt-4o"  # ✓ High stakes, need accuracy

# Simple FAQ chatbot
model = "gpt-4o-mini"  # ✓ Perfect fit

# Complex SQL query generation
model = "gpt-4o"  # ✓ Errors are expensive

# Sentiment analysis (1M texts)
model = "gpt-4o-mini"  # ✓ Volume play

# Research paper analysis
model = "o1-preview"  # ✓ Needs deep reasoning
```

**Cost comparison (100K requests):**

```python
# Scenario: Classify 100K support tickets
# Average: 100 input tokens, 20 output tokens per request

def calculate_cost(model, num_requests=100_000):
    input_tokens = 100 * num_requests
    output_tokens = 20 * num_requests

    prices = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
    }

    cost = (
        (input_tokens / 1_000_000) * prices[model]["input"] +
        (output_tokens / 1_000_000) * prices[model]["output"]
    )

    return cost

print(f"gpt-4o: ${calculate_cost('gpt-4o'):.2f}")          # $450
print(f"gpt-4o-mini: ${calculate_cost('gpt-4o-mini'):.2f}")  # $27
print(f"gpt-3.5-turbo: ${calculate_cost('gpt-3.5-turbo'):.2f}")  # $80

# gpt-4o-mini is 16x cheaper than gpt-4o!
```

**Rule of thumb:**
- Start with **gpt-4o-mini** for everything
- Only upgrade to **gpt-4o** if quality issues arise
- Use **o1-preview** only for genuinely hard reasoning tasks
- Avoid **gpt-3.5-turbo** (gpt-4o-mini is better value)

---

#### Q6: How do you implement streaming responses for real-time UX?

**Answer:**

**Streaming benefits:**
```python
# Without streaming: User waits for full response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem"}]
)
# Response: 1000 tokens = ~4 seconds wait
print(response.choices[0].message.content)

# With streaming: See tokens arrive in real-time
with client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)  # See immediate output
```

**Server-sent events (SSE) pattern:**
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(messages: list):
    async def generate():
        with client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    # Send as SSE
                    data = {
                        "token": chunk.choices[0].delta.content,
                        "finish_reason": chunk.choices[0].finish_reason
                    }
                    yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# Frontend JavaScript:
# const response = await fetch('/chat/stream', {method: 'POST'})
# const reader = response.body.getReader()
# while (true) {
#     const {value, done} = await reader.read()
#     if (done) break
#     console.log(new TextDecoder().decode(value))  // Show token
# }
```

**Client-side streaming (Python):**
```python
def stream_chat(messages):
    """Stream response token-by-token"""
    buffer = ""

    with client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    ) as stream:
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                buffer += token

                # Yield complete sentences
                if token.endswith(('.', '!', '?', '\n')):
                    yield buffer
                    buffer = ""

        if buffer:  # Final partial
            yield buffer

# Usage
for chunk in stream_chat(messages):
    print(chunk, end="", flush=True)
```

**Performance comparison:**
```
Streaming:
- First token: 200ms (appear on screen immediately)
- Full response: 4s (but perceived as instant)

Non-streaming:
- Full response: 4s (blank then all text)

Perceived latency reduction: 95%!
```

---

#### Q7: How do you maintain conversation history effectively?

**Answer:**

**Problem with naive history:**
```python
# Context grows unbounded
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "4"},
    # ... 100 more turns ...
    # Now context is 50K tokens, costs skyrocket!
]
```

**Solution 1: Sliding Window (last N turns)**
```python
class ConversationManager:
    def __init__(self, max_turns=20):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

        # Keep only recent turns
        if len(self.history) > self.max_turns * 2:  # *2 for user+assistant pairs
            self.history = self.history[-self.max_turns*2:]

    def get_messages(self):
        return [
            {"role": "system", "content": "You are helpful"}
        ] + self.history

    def chat(self, user_message):
        self.add_turn("user", user_message)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.get_messages()
        )

        assistant_response = response.choices[0].message.content
        self.add_turn("assistant", assistant_response)

        return assistant_response
```

**Solution 2: Summarization (compress old history)**
```python
def compress_history(messages, keep_turns=5):
    """Keep recent turns, summarize old ones"""
    if len(messages) <= keep_turns * 2:
        return messages

    # Split into old and recent
    old_messages = messages[:-(keep_turns*2)]
    recent_messages = messages[-(keep_turns*2):]

    # Summarize old conversation
    summary_prompt = f"""Summarize this conversation concisely:

{format_messages(old_messages)}

Summary (2-3 sentences):"""

    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}]
    )

    return [
        {"role": "user", "content": f"[Previous conversation summary: {summary}]"},
        *recent_messages
    ]
```

**Solution 3: Token Budget**
```python
import tiktoken

def build_messages_within_budget(history, max_tokens=2000):
    """Include as much history as fits in token budget"""
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    selected = [{"role": "system", "content": "You are helpful"}]
    token_count = len(encoding.encode("You are helpful"))

    # Add messages from most recent backwards
    for message in reversed(history):
        msg_tokens = len(encoding.encode(message["content"]))

        if token_count + msg_tokens > max_tokens:
            break

        selected.insert(1, message)  # Insert after system
        token_count += msg_tokens

    return selected
```

**Conversation context example:**
```python
# User asks question referencing past context
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "I'm building an e-commerce site"},
    {"role": "assistant", "content": "Great! What tech stack?"},
    {"role": "user", "content": "Python with Django"},
    {"role": "assistant", "content": "Good choice for rapid development"},
    {"role": "user", "content": "How do I handle payments?"}  # Implicitly: with Python/Django
]

# Model understands context from history automatically
```

---

#### Q8: How do you validate LLM outputs with Pydantic?

**Answer:**

**Multi-level validation:**
```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List
import re

class SocialMediaPost(BaseModel):
    platform: Literal["twitter", "linkedin", "instagram"]
    text: str = Field(min_length=5, max_length=280)
    hashtags: List[str] = Field(max_items=10)
    engagement_prediction: float = Field(ge=0, le=1)

    @field_validator('platform')
    def validate_platform(cls, v):
        if v not in ["twitter", "linkedin", "instagram"]:
            raise ValueError(f"Invalid platform: {v}")
        return v

    @field_validator('text')
    def validate_text(cls, v):
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError("Text must contain letters")
        return v

    @field_validator('hashtags')
    def validate_hashtags(cls, v):
        for tag in v:
            if not tag.startswith('#'):
                raise ValueError(f"Hashtags must start with #")
        return v

# LLM generates post
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Create a Twitter post about Python"
    }],
    response_format=SocialMediaPost
)

# Automatically validated by Pydantic
post = response.choices[0].message.parsed
print(f"Platform: {post.platform}")  # Guaranteed valid
```

**Custom validators:**
```python
from datetime import datetime

class EventAnalysis(BaseModel):
    event_name: str
    date: str
    severity: Literal["low", "medium", "high", "critical"]
    impact_score: int = Field(ge=0, le=100)

    @field_validator('date')
    def validate_date(cls, v):
        try:
            parsed = datetime.fromisoformat(v)
            if parsed < datetime.now():
                raise ValueError("Date cannot be in the past")
            return v
        except ValueError:
            raise ValueError("Invalid date format, use ISO 8601")

    @field_validator('impact_score')
    def validate_score(cls, v, info):
        severity = info.data.get('severity')

        if severity == "critical" and v < 70:
            raise ValueError("Critical events must have score >= 70")

        return v
```

**Fallback handling:**
```python
def parse_with_fallback(response_text: str, model: type):
    """Try to parse, fallback to default if invalid"""
    try:
        return model.model_validate_json(response_text)
    except Exception as e:
        print(f"Parse failed: {e}")

        # Return valid instance with defaults
        return model(
            **{
                f.name: f.default
                for f in model.model_fields.values()
                if f.default is not None
            }
        )
```

---

#### Q9: How do you handle errors and retries in production?

**Answer:**

**Error types and handling:**
```python
from openai import (
    APIError, RateLimitError, APIConnectionError,
    AuthenticationError
)
from tenacity import retry, stop_after_attempt, wait_exponential
import time

def classify_with_retries(text: str, max_retries=3):
    """Classify with specific error handling"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}],
                timeout=30
            )
            return response.choices[0].message.content

        except RateLimitError:
            # API rate limit: exponential backoff
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        except APIConnectionError:
            # Network issue: retry immediately
            if attempt < max_retries - 1:
                print(f"Connection error, retrying...")
                time.sleep(1)

        except AuthenticationError:
            # Invalid API key: don't retry
            print("Invalid API key!")
            raise

        except APIError as e:
            # Other API errors: retry with backoff
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise Exception("Failed after retries")
```

**Tenacity decorator:**
```python
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def robust_api_call(messages):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
```

**Circuit breaker pattern:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            self.state = "closed"
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker opened after {self.failure_count} failures")

            raise

# Usage
breaker = CircuitBreaker(failure_threshold=3)

try:
    response = breaker.call(
        lambda: client.chat.completions.create(...)
    )
except Exception as e:
    print(f"Request failed: {e}")
```

---

#### Q10: How do you design prompts with templates for reusability?

**Answer:**

**Prompt template pattern:**
```python
from string import Template

class PromptTemplate:
    def __init__(self, template: str, variables: list):
        self.template = Template(template)
        self.variables = variables

    def format(self, **kwargs):
        """Substitute variables"""
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return self.template.substitute(**kwargs)

# Define templates
classification_template = PromptTemplate(
    """Classify the following text into one category:
$category_list

Text: $text

Classification:""",
    variables=["category_list", "text"]
)

# Reuse with different inputs
categories = "positive, negative, neutral"
text1 = "I love this product!"
text2 = "This is broken"

prompt1 = classification_template.format(
    category_list=categories,
    text=text1
)
# Output: "Classify the following text... Text: I love this product!"
```

**Advanced templating with context:**
```python
class PromptBuilder:
    def __init__(self, role: str, style: str):
        self.role = role
        self.style = style

    def build(self, task: str, **context) -> str:
        system = f"""You are a {self.role}.
Writing style: {self.style}.
Be concise and accurate."""

        # Dynamic context injection
        context_str = "\n".join([
            f"- {k}: {v}" for k, v in context.items()
        ])

        return f"""{system}

Context:
{context_str}

Task: {task}

Response:"""

# Usage
builder = PromptBuilder(
    role="technical writer",
    style="professional but approachable"
)

prompt = builder.build(
    task="Explain what RAG is",
    audience="beginners",
    length="2-3 sentences",
    include_examples="yes"
)
```

**Few-shot examples:**
```python
def build_few_shot_prompt(examples: list, new_input: str) -> str:
    """Few-shot learning with examples"""
    prompt = "Classify text as sentiment (positive/negative):\n\n"

    # Add examples
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Text: {example['text']}\n"
        prompt += f"Classification: {example['label']}\n\n"

    # New input
    prompt += f"Now classify:\n"
    prompt += f"Text: {new_input}\n"
    prompt += f"Classification:"

    return prompt

examples = [
    {"text": "This is amazing!", "label": "positive"},
    {"text": "Terrible experience", "label": "negative"},
    {"text": "It's okay", "label": "neutral"}
]

prompt = build_few_shot_prompt(examples, "I really enjoyed it")
```

---

### Implementation & Coding (11-20)

#### Q11: Implement a production-ready classification system with structured outputs

**Answer:**

```python
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum
import time
from functools import lru_cache

# Define schema with Pydantic
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TicketClassification(BaseModel):
    """Classification result for support ticket"""

    category: Literal[
        "technical",
        "billing",
        "account",
        "feature_request",
        "bug_report",
        "general"
    ] = Field(description="Main category of the ticket")

    priority: Priority = Field(description="Urgency level")

    subcategory: str = Field(
        description="Specific subcategory",
        max_length=50
    )

    requires_human_review: bool = Field(
        description="Whether ticket needs human escalation"
    )

    estimated_resolution_time: int = Field(
        description="Estimated hours to resolve",
        ge=0,
        le=720  # Max 30 days
    )

    confidence_score: float = Field(
        description="Confidence in classification",
        ge=0,
        le=1
    )

    reasoning: str = Field(
        description="Explanation of classification",
        min_length=10,
        max_length=500
    )

    @field_validator('subcategory')
    def validate_subcategory(cls, v, info):
        """Ensure subcategory matches category"""
        category = info.data.get('category')

        valid_subcategories = {
            "technical": ["login_issue", "performance", "crash", "integration"],
            "billing": ["payment_failed", "subscription", "refund", "invoice"],
            "account": ["password_reset", "profile_update", "deletion"],
            "feature_request": ["new_feature", "enhancement"],
            "bug_report": ["ui_bug", "data_bug", "security"],
            "general": ["question", "feedback", "other"]
        }

        if category and v not in valid_subcategories.get(category, []):
            raise ValueError(
                f"Invalid subcategory '{v}' for category '{category}'"
            )

        return v


class TicketClassifier:
    """Production ticket classifier with caching and error handling"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.total_calls = 0
        self.total_cost = 0
        self.cache_hits = 0

    @lru_cache(maxsize=1000)
    def _classify_cached(self, ticket_text: str) -> TicketClassification:
        """Cached classification to avoid duplicate API calls"""
        return self._classify(ticket_text)

    def _classify(self, ticket_text: str) -> TicketClassification:
        """Internal classification method"""

        system_prompt = """You are an expert support ticket classifier.
Analyze the ticket and provide structured classification.

Category guidelines:
- technical: System issues, bugs, performance, integrations
- billing: Payments, subscriptions, invoices, refunds
- account: Login, profile, permissions, account management
- feature_request: Requests for new features or enhancements
- bug_report: Software bugs and errors
- general: Questions, feedback, other

Priority rules:
- urgent: System down, data loss, security breach
- high: Major functionality broken, paying customer affected
- medium: Feature not working, workaround exists
- low: Cosmetic issues, general questions
"""

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this ticket:\n\n{ticket_text}"}
                ],
                response_format=TicketClassification
            )

            # Track usage
            self.total_calls += 1
            usage = response.usage
            cost = (
                (usage.prompt_tokens / 1_000_000) * 0.15 +
                (usage.completion_tokens / 1_000_000) * 0.60
            )
            self.total_cost += cost

            return response.choices[0].message.parsed

        except Exception as e:
            # Log error and return default classification
            print(f"Classification error: {e}")
            return TicketClassification(
                category="general",
                priority=Priority.MEDIUM,
                subcategory="other",
                requires_human_review=True,
                estimated_resolution_time=24,
                confidence_score=0.0,
                reasoning=f"Error during classification: {str(e)}"
            )

    def classify(self, ticket_text: str, use_cache: bool = True) -> TicketClassification:
        """
        Classify a support ticket

        Args:
            ticket_text: The ticket content
            use_cache: Whether to use cache for identical tickets

        Returns:
            TicketClassification object
        """
        if use_cache:
            try:
                result = self._classify_cached(ticket_text)
                self.cache_hits += 1
                return result
            except:
                pass

        return self._classify(ticket_text)

    def classify_batch(
        self,
        tickets: list[str],
        batch_size: int = 10,
        delay: float = 0.1
    ) -> list[TicketClassification]:
        """
        Classify multiple tickets with rate limiting

        Args:
            tickets: List of ticket texts
            batch_size: Number of tickets to process before delay
            delay: Seconds to wait between batches

        Returns:
            List of classifications
        """
        results = []

        for i, ticket in enumerate(tickets):
            result = self.classify(ticket)
            results.append(result)

            # Rate limiting
            if (i + 1) % batch_size == 0:
                time.sleep(delay)

        return results

    def get_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "total_cost": self.total_cost,
            "avg_cost_per_call": self.total_cost / max(self.total_calls, 1)
        }


# Usage
if __name__ == "__main__":
    classifier = TicketClassifier(api_key="your-key-here")

    # Single ticket
    ticket = """
    Subject: Can't login to my account

    I've been trying to reset my password for the past hour but the reset
    email never arrives. I've checked spam folder. This is blocking my work,
    I need access urgently.
    """

    result = classifier.classify(ticket)

    print(f"Category: {result.category}")
    print(f"Priority: {result.priority}")
    print(f"Subcategory: {result.subcategory}")
    print(f"Human review needed: {result.requires_human_review}")
    print(f"Est. resolution: {result.estimated_resolution_time}h")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Reasoning: {result.reasoning}")

    # Batch processing
    tickets = [ticket1, ticket2, ticket3]
    results = classifier.classify_batch(tickets)

    # Stats
    stats = classifier.get_stats()
    print(f"\nTotal cost: ${stats['total_cost']:.4f}")
    print(f"Cache hits: {stats['cache_hits']}")
```

---

#### Q12: How do you implement function calling with tool use?

**Answer:**

**Basic function calling:**
```python
def get_weather(location: str, unit: str = "C") -> str:
    """Get weather for location"""
    return f"Weather in {location}: 22{unit}, sunny"

def search_web(query: str) -> str:
    """Search the web"""
    return f"Results for '{query}': [result1, result2]"

# Define tools for LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["C", "F"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

# User asks question
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "What's the weather in Paris and the capital of France?"
    }],
    tools=tools
)

# LLM decides to call function
tool_calls = response.choices[0].message.tool_calls
for call in tool_calls:
    func_name = call.function.name
    func_args = json.loads(call.function.arguments)

    # Execute function
    if func_name == "get_weather":
        result = get_weather(**func_args)
    elif func_name == "search_web":
        result = search_web(**func_args)

    # Send result back
    messages.append({"role": "assistant", "content": str(response.choices[0].message)})
    messages.append({"role": "tool", "content": result, "tool_call_id": call.id})

# Get final answer
final_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(final_response.choices[0].message.content)
```

**Tool registry pattern:**
```python
from typing import Callable, Dict

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.definitions: list = []

    def register(self, func: Callable, description: str, parameters: dict):
        """Register a tool"""
        self.tools[func.__name__] = func

        self.definitions.append({
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters
            }
        })

    def execute(self, tool_name: str, args: dict):
        """Execute registered tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        return self.tools[tool_name](**args)

# Usage
registry = ToolRegistry()

registry.register(
    get_weather,
    "Get weather for location",
    {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=registry.definitions
)

# Execute
for call in response.choices[0].message.tool_calls:
    args = json.loads(call.function.arguments)
    result = registry.execute(call.function.name, args)
```

---

#### Q13: How do you extract structured data from unstructured text?

**Answer:**

**Information extraction:**
```python
from pydantic import BaseModel, Field
from typing import List

class Person(BaseModel):
    name: str
    age: int
    email: str
    phone: str = None

class ContactList(BaseModel):
    people: List[Person]
    extracted_date: str

text = """
John Smith, 28 years old, john@company.com, 555-1234
Jane Doe, 34, jane.doe@email.com
Bob Johnson, age 45, bob@example.com
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": f"Extract people from this text:\n\n{text}"
    }],
    response_format=ContactList
)

contacts = response.choices[0].message.parsed
for person in contacts.people:
    print(f"{person.name}: {person.email}")
```

**Named entity recognition:**
```python
class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: str
    items: List[dict]
    total_amount: float

invoice_text = """
ACME Corp Invoice #INV-2024-001
Date: January 15, 2024

Items:
- Widget A: $50.00 x 2 = $100.00
- Widget B: $30.00 x 1 = $30.00

Total: $130.00
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": f"Extract invoice data:\n\n{invoice_text}"
    }],
    response_format=InvoiceData
)

invoice = response.choices[0].message.parsed
print(f"Vendor: {invoice.vendor_name}")
print(f"Total: ${invoice.total_amount}")
```

---

#### Q14: How do you implement retry logic for transient failures?

**Answer:**

**Exponential backoff with jitter:**
```python
import random
import time

def retry_with_backoff(func, max_retries=5, base_delay=1):
    """Retry with exponential backoff + jitter"""

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff: 1, 2, 4, 8, 16
            delay = base_delay * (2 ** attempt)

            # Add jitter: ±0-50% random
            jitter = random.uniform(0, delay * 0.5)
            wait_time = delay + jitter

            print(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s")
            time.sleep(wait_time)

# Usage
response = retry_with_backoff(
    lambda: client.chat.completions.create(...)
)
```

**Structured retry:**
```python
class RetryConfig:
    def __init__(self):
        self.attempt = 0
        self.max_attempts = 3
        self.base_delay = 1
        self.max_delay = 60

    def should_retry(self, error: Exception) -> bool:
        """Check if error is retryable"""
        retryable = [
            RateLimitError,
            APIConnectionError,
            TimeoutError
        ]
        return any(isinstance(error, e) for e in retryable)

    def wait(self):
        """Calculate wait time"""
        delay = self.base_delay * (2 ** self.attempt)
        delay = min(delay, self.max_delay)
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter

def call_with_retry(messages):
    config = RetryConfig()

    while config.attempt < config.max_attempts:
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except Exception as e:
            if not config.should_retry(e):
                raise

            config.attempt += 1
            wait_time = config.wait()
            print(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    raise Exception("Max retries exceeded")
```

---

#### Q15: How do you implement response caching?

**Answer:**

**In-memory caching:**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_classify(text: str) -> str:
    """Cache classification results"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify: {text}"}]
    )
    return response.choices[0].message.content

# Same input = cached result (no API call)
result1 = cached_classify("This is great!")  # API call
result2 = cached_classify("This is great!")  # Cached result
```

**Redis caching:**
```python
import redis
import json
import hashlib

class CachedLLMClient:
    def __init__(self, redis_host="localhost"):
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
        self.ttl = 3600  # 1 hour

    def _get_cache_key(self, messages: list) -> str:
        """Generate cache key from messages"""
        msg_str = json.dumps(messages, sort_keys=True)
        return f"llm:{hashlib.md5(msg_str.encode()).hexdigest()}"

    def chat(self, messages: list, use_cache=True) -> str:
        """Chat with optional caching"""
        cache_key = self._get_cache_key(messages)

        # Try cache first
        if use_cache:
            cached = self.redis.get(cache_key)
            if cached:
                print("Cache hit!")
                return cached

        # Call API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        result = response.choices[0].message.content

        # Store in cache
        self.redis.setex(cache_key, self.ttl, result)

        return result
```

---

#### Q16: How do you parse JSON from LLM responses reliably?

**Answer:**

**Robust JSON parsing:**
```python
import json
import re

def extract_json(text: str) -> dict:
    """Extract JSON from LLM response with robustness"""

    # Try direct parsing first
    try:
        return json.loads(text)
    except:
        pass

    # Look for JSON blocks
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            return json.loads(match)
        except:
            continue

    # Try fixing common issues
    # Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Quote unquoted keys
    text = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', text)

    try:
        return json.loads(text)
    except:
        raise ValueError("Could not parse JSON from response")

# Usage
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Return JSON: {...}"}]
)

data = extract_json(response.choices[0].message.content)
```

---

#### Q17: How do you process multiple requests asynchronously?

**Answer:**

**Async processing:**
```python
import asyncio
from openai import AsyncOpenAI

async def process_batch_async(texts: list) -> list:
    """Process multiple texts concurrently"""
    client = AsyncOpenAI()

    async def classify_one(text: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Classify: {text}"}]
        )
        return response.choices[0].message.content

    # Run concurrently
    results = await asyncio.gather(*[
        classify_one(text) for text in texts
    ])

    return results

# Usage
texts = ["text1", "text2", "text3"]
results = asyncio.run(process_batch_async(texts))
```

**Concurrent with rate limiting:**
```python
import asyncio
from datetime import datetime, timedelta

class RateLimitedAsyncClient:
    def __init__(self, rpm: int = 3000):
        self.rpm = rpm
        self.min_interval = 60 / rpm
        self.last_request_time = None

    async def call(self, messages: list) -> str:
        """Rate-limited API call"""
        # Wait if needed
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

        self.last_request_time = datetime.now()

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return response.choices[0].message.content
```

---

#### Q18: How do you track and monitor API usage?

**Answer:**

**Usage tracking:**
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class UsageRecord:
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float

class UsageTracker:
    PRICES = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60}
    }

    def __init__(self):
        self.records = []

    def track(self, response, model: str):
        """Track API response usage"""
        usage = response.usage
        cost = (
            (usage.prompt_tokens / 1_000_000) * self.PRICES[model]["input"] +
            (usage.completion_tokens / 1_000_000) * self.PRICES[model]["output"]
        )

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost
        )

        self.records.append(record)
        return record

    def get_stats(self) -> dict:
        """Get usage statistics"""
        total_tokens = sum(r.total_tokens for r in self.records)
        total_cost = sum(r.cost for r in self.records)

        return {
            "total_calls": len(self.records),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_call": total_cost / max(len(self.records), 1)
        }

# Usage
tracker = UsageTracker()

response = client.chat.completions.create(...)
tracker.track(response, "gpt-4o-mini")

stats = tracker.get_stats()
print(f"Total cost: ${stats['total_cost']:.4f}")
```

---

#### Q19: How do you implement batch processing with OpenAI Batch API?

**Answer:**

**Batch API for cost savings (50% cheaper):**
```python
import jsonl
import time

def create_batch(requests: list) -> str:
    """Create batch request file"""
    batch_requests = []

    for i, req in enumerate(requests):
        batch_requests.append({
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": req["messages"]
            }
        })

    # Write to JSONL
    with open("batch_requests.jsonl", "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    # Upload file
    with open("batch_requests.jsonl", "rb") as f:
        file_response = client.files.create(
            file=f,
            purpose="batch"
        )

    file_id = file_response.id

    # Create batch
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch.id

def get_batch_results(batch_id: str) -> list:
    """Get results from completed batch"""
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"Batch status: {batch.status}")
        return []

    # Download results
    results_file = client.files.content(batch.output_file_id)

    results = []
    for line in results_file.text.split('\n'):
        if line:
            results.append(json.loads(line))

    return results
```

---

#### Q20: How do you handle long context efficiently?

**Answer:**

**Context optimization:**
```python
def optimize_context(documents: list, query: str, max_tokens: int = 3000):
    """Select most relevant docs within token limit"""
    # Rank docs by relevance
    scores = []
    for doc in documents:
        similarity = calculate_similarity(query, doc["content"])
        token_count = len(encoding.encode(doc["content"]))
        scores.append((doc, similarity, token_count))

    # Sort by relevance
    scores.sort(key=lambda x: x[1], reverse=True)

    # Select until token limit
    selected = []
    total_tokens = 0

    for doc, score, tokens in scores:
        if total_tokens + tokens > max_tokens:
            break

        selected.append(doc)
        total_tokens += tokens

    return selected

# LongContext pattern: don't add everything
context_docs = optimize_context(all_docs, query)

prompt = f"""Answer based on:
{format_docs(context_docs)}

Question: {query}

Answer:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
```

---

### Debugging & Troubleshooting (21-25)

#### Q21: How do you debug timeout and connection errors?

**Answer:**

**Timeout scenarios:**
```python
import requests
from openai import APIConnectionError, Timeout

# Problem: Response too slow
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Long task..."}],
        timeout=5  # 5 second timeout
    )
except Timeout:
    print("Request timed out after 5s")
    # Solutions:
    # 1. Increase timeout
    # 2. Use streaming for long responses
    # 3. Break into smaller requests

# Debugging checklist:
# - Is network stable? (ping api.openai.com)
# - Is API down? (check status.openai.com)
# - Is timeout too short?
# - Is message too large?

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    timeout=30  # Increase timeout for complex tasks
)
```

**Connection debugging:**
```python
import logging
from openai import APIConnectionError

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

try:
    response = client.chat.completions.create(...)
except APIConnectionError as e:
    print(f"Connection failed: {e}")
    print(f"Error code: {e.status_code}")

    # Diagnose:
    if "HTTPSConnectionPool" in str(e):
        print("→ SSL/TLS issue")
    elif "Connection refused" in str(e):
        print("→ Server not responding")
    elif "Name or service not known" in str(e):
        print("→ DNS resolution failed")
```

---

#### Q22: How do you handle rate limit errors in production?

**Answer:**

**Rate limit signals:**
```python
from openai import RateLimitError
import time

def handle_rate_limit():
    """Understand rate limit headers"""
    try:
        response = client.chat.completions.create(...)
    except RateLimitError as e:
        # Headers contain retry info
        retry_after = e.response.headers.get("retry-after")
        if retry_after:
            wait_time = int(retry_after)
            print(f"Rate limited, wait {wait_time}s")
            time.sleep(wait_time)

# Recommended: RPM (requests per minute) not TPM
# Track both metrics
class RateLimitMonitor:
    def __init__(self, rpm_limit=3000, tpm_limit=150000):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.minute_start = time.time()

    def can_request(self, estimated_tokens: int) -> bool:
        """Check if request would exceed limits"""
        now = time.time()

        # Reset if new minute
        if now - self.minute_start > 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = now

        # Check limits
        if self.requests_this_minute >= self.rpm_limit:
            return False

        if self.tokens_this_minute + estimated_tokens > self.tpm_limit:
            return False

        return True

    def record(self, tokens_used: int):
        """Record API usage"""
        self.requests_this_minute += 1
        self.tokens_this_minute += tokens_used
```

---

#### Q23: How do you handle token limit exceeded errors?

**Answer:**

**Context window management:**
```python
import tiktoken

def safe_chat_with_truncation(messages: list, model="gpt-4o-mini", max_tokens=120000):
    """Call API with automatic context truncation"""
    encoding = tiktoken.encoding_for_model(model)

    # Count tokens
    total_tokens = 0
    for msg in messages:
        tokens = len(encoding.encode(msg["content"]))
        total_tokens += tokens

    # Leave room for response
    response_budget = 2000

    if total_tokens + response_budget > max_tokens:
        print(f"Context too large ({total_tokens}), truncating...")

        # Remove oldest messages keeping system
        truncated = [m for m in messages if m["role"] == "system"]

        # Add recent messages up to limit
        for msg in reversed(messages):
            if msg["role"] == "system":
                continue

            tokens = len(encoding.encode(msg["content"]))
            if total_tokens + tokens > max_tokens - response_budget:
                break

            truncated.insert(1, msg)  # After system
            total_tokens += tokens

        messages = truncated

    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=response_budget
        )
    except Exception as e:
        if "context_length_exceeded" in str(e):
            raise ValueError(f"Message still too long: {total_tokens} tokens")
        raise
```

---

#### Q24: How do you debug inconsistent LLM outputs?

**Answer:**

**Nondeterministic outputs:**
```python
# Problem: Same input → different output
def unstable_classification(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify: {text}"}]
    )
    return response.choices[0].message.content

# Run twice, get different results!
result1 = unstable_classification("Good product")  # "positive"
result2 = unstable_classification("Good product")  # "positive" (mostly)

# Causes of instability:
# 1. High temperature → increase randomness
# 2. Ambiguous prompt → needs clarification
# 3. Underspecified output → needs examples

# Solutions:
def stable_classification(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classify as positive/negative/neutral.

Examples:
- "Great product!" → positive
- "Broken item" → negative
- "It's okay" → neutral

Text: {text}
Classification:"""
        }],
        temperature=0  # Deterministic!
    )
    return response.choices[0].message.content
```

**Consistency testing:**
```python
def test_consistency(text: str, runs=5):
    """Test output consistency"""
    results = []

    for i in range(runs):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            temperature=0  # Should be deterministic
        )
        results.append(response.choices[0].message.content)

    # Check if all same
    if len(set(results)) == 1:
        print("✓ Consistent")
    else:
        print("✗ Inconsistent outputs:")
        for i, r in enumerate(results):
            print(f"  Run {i+1}: {r}")

test_consistency("Classify: Amazing product", runs=5)
```

---

#### Q25: How do you handle API response parsing failures?

**Answer:**

**Graceful degradation:**
```python
def robust_parse_response(response, fallback_value=None):
    """Parse response with fallback"""
    try:
        # Try to parse
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"Parse failed: {e}")

        # Fallback strategies
        if fallback_value is not None:
            return fallback_value

        # Return empty/default model
        return get_default_model()

# Usage
try:
    result = robust_parse_response(response, fallback_value={"status": "error"})
except Exception as e:
    log_error(e)
    return default_response()
```

---

### Trade-offs & Decisions (26-30)

#### Q26: Structured outputs vs Function calling - when to use each?

**Answer:**

**Quick comparison:**
```python
# Use Structured Outputs when:
# ✓ LLM is the final step (data extraction)
# ✓ Need guaranteed valid schema
# ✓ One-shot processing

class Analysis(BaseModel):
    category: str
    confidence: float

response = client.beta.chat.completions.parse(
    messages=[{"role": "user", "content": "Analyze this..."}],
    response_format=Analysis
)
result = response.choices[0].message.parsed


# Use Function Calling when:
# ✓ LLM needs to trigger actions
# ✓ Multi-step workflows
# ✓ Conditonal tool use

tools = [{
    "type": "function",
    "function": {
        "name": "book_flight",
        "parameters": {...}
    }
}]

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Book a flight to Paris"}],
    tools=tools
)

# LLM decides whether to call
tool_call = response.choices[0].message.tool_calls[0]
execute_tool(tool_call)
```

**Workflow decision tree:**
```
Query arrives
├─ Is LLM the data consumer?
│  └─ YES → Structured Outputs
│
└─ Does LLM need to trigger actions?
   └─ YES → Function Calling
      ├─ Multi-step?
      │  └─ YES → Agentic loop
      │
      └─ Single tool?
         └─ YES → Simple function call
```

---

#### Q27: How do you choose between different embedding/reranking models?

**Answer:**

**Model trade-offs:**
```python
# Embedding models (for semantic search):
from sentence_transformers import SentenceTransformer

# Small, fast (good for mobile)
model = SentenceTransformer("all-MiniLM-L6-v2")  # 22M params
embeddings = model.encode(texts)

# Medium, accurate (good for RAG)
model = SentenceTransformer("all-mpnet-base-v2")  # 109M params

# Large, very accurate (good for research)
model = SentenceTransformer("all-t5-3b")  # 3B params

# Reranking models (for precision):
from sentence_transformers import CrossEncoder

# Fast reranker
reranker = CrossEncoder("ms-marco-MiniLM-L-6-v2")

# More accurate reranker
reranker = CrossEncoder("ms-marco-ELECTRA-Base")

# Performance table:
# Model | Speed | Accuracy | Memory | Use Case
# --------------|-------|----------|--------|----------
# MiniLM | ✓✓✓ | ✓✓ | ✓✓✓ | Production RAG
# mpnet | ✓✓ | ✓✓✓ | ✓✓ | Balanced
# T5-3B | ✓ | ✓✓✓✓ | ✓ | Research
```

---

#### Q28: Caching strategy - what to cache and for how long?

**Answer:**

**Cache decisions:**
```python
class SmartCache:
    def __init__(self):
        self.cache = {}
        self.ttls = {}

    def cache_with_ttl(self, key: str, value: any, ttl: int):
        """Cache with appropriate TTL"""
        self.cache[key] = value
        self.ttls[key] = ttl

    def should_cache(self, query: str, response: str) -> bool:
        """Decide if worth caching"""
        # Cache if:
        # ✓ Query is deterministic (same = same result)
        # ✓ Query is repeated (worth storing)
        # ✓ Response is long (expensive to recompute)

        # Don't cache:
        # ✗ Personalized responses (different per user)
        # ✗ Current events (quickly outdated)
        # ✗ Short responses (cheap to recompute)

        # Heuristics:
        is_static_query = not any(word in query.lower() for word in
            ["today", "now", "current", "latest"])
        is_expensive = len(response) > 1000  # Long response

        return is_static_query and is_expensive

# TTL decisions:
# FAQ answers: 1 week (static content)
product_info = {"ttl": 7 * 24 * 3600}

# Classification results: 1 day (might update)
classification = {"ttl": 24 * 3600}

# Current events: 1 hour
current_events = {"ttl": 3600}

# User preferences: per-session
user_prefs = {"ttl": None}  # Never auto-expire
```

---

#### Q29: Async vs sync - when to use each?

**Answer:**

**When to use sync:**
```python
# Single request, script-like
response = client.chat.completions.create(...)

# Low volume (<100 requests/hour)
for doc in docs[:10]:
    result = classify(doc)  # Sequential is fine

# High latency tolerance
response = requests.get(api_url)  # Don't care if slow
```

**When to use async:**
```python
import asyncio

# High volume (100s-1000s of requests)
async def process_many(items):
    tasks = [process_one(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results

# Low latency requirements
# User waiting for response - use streaming instead
# But if need multiple parallel calls: async

# Example: Multi-stage pipeline
async def complex_processing(query):
    # Stage 1: Retrieve docs (async)
    docs = await retrieve_docs_async(query)

    # Stage 2: Extract info from each doc (parallel)
    extractions = await asyncio.gather(*[
        extract_info_async(doc) for doc in docs
    ])

    # Stage 3: Summarize (async)
    summary = await summarize_async(extractions)

    return summary
```

**Overhead comparison:**
```python
# Sync: simple, no overhead
result = call_api()  # Direct

# Async: adds overhead
async def call_async():
    return await call_api()  # Event loop, coroutines

# Use async only if parallelism gain > overhead
# Rule of thumb: async if >50 parallel requests
```

---

#### Q30: Self-hosted vs API - cost and quality trade-offs?

**Answer:**

**Self-hosted (open-source models):**
```python
# Setup cost: One-time GPU purchase (1-10K USD)
# Running cost: Electricity + maintenance (100-500 USD/month)
# Quality: Generally lower than GPT-4
# Latency: Lower (no network)
# Privacy: Higher (data stays local)

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Your responsibility:
# - Keep server running
# - Handle scaling
# - Monitor performance
# - Security patches

# Good for:
# - High volume (1M+ requests/day)
# - Sensitive data (can't send to cloud)
# - Custom model (fine-tune for domain)
```

**API (OpenAI, Claude):**
```python
# Setup cost: Zero (just API key)
# Running cost: Per-token (0.15-10 USD per 1M tokens)
# Quality: Very high (frontier models)
# Latency: Higher (network), but
# Privacy: Cloud-hosted (data privacy concerns)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# They handle:
# - Infrastructure scaling
# - Model updates
# - Security/availability
# - Performance optimization

# Good for:
# - Low-medium volume (<100K req/day)
# - Quality critical (want best models)
# - Flexible usage (don't want capital cost)
```

**Cost comparison example:**
```python
# Scenario: Process 100K texts/day at 50 tokens each

# OpenAI gpt-4o-mini
api_cost = 100_000 * 50 * (0.15 / 1_000_000)
# = $0.75/day = $22/month

# Self-hosted (Llama 2 7B on GPU)
gpu_cost = 5000 / 36  # 3-year amortization
monthly_electricity = 300

# Breakeven: 100+ days
# After breakeven: self-hosted cheaper

# But consider:
# - Llama quality << GPT-4o-mini quality
# - Maintenance time cost?
# - Scaling complexity?
```

**Decision framework:**
```
High volume? (>100K req/day)
├─ YES + quality critical → Self-hosted + fine-tune
├─ YES + generic tasks → Self-hosted (open model)
└─ NO → API (OpenAI/Claude)

Sensitive data?
├─ YES → Self-hosted (local)
└─ NO → API

Quality non-negotiable?
├─ YES → API (GPT-4o)
└─ NO → Self-hosted or gpt-4o-mini
```

---

