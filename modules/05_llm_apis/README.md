# Module 5: LLM APIs

> Production-ready OpenAI API usage with structured outputs and error handling

## Why This Matters

LLMs are powerful, but raw text output is unreliable for production systems. Structured outputs (JSON with validation) enable reliable integration with databases, APIs, and business logic. This module teaches production patterns for LLM API usage.

## Key Concepts

### API Basics

```python
from openai import OpenAI

client = OpenAI(api_key="your-key-here")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
)

print(response.choices[0].message.content)
```

### Structured Outputs with Pydantic

**Problem with raw text:**
```python
# Unreliable - output format can vary
response = "The sentiment is positive with confidence 0.85"
# How do you parse this reliably?
```

**Solution with structured outputs:**
```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "I love this product!"}],
    response_format=Sentiment
)

result = response.choices[0].message.parsed
# result.label = "positive"
# result.confidence = 0.95
```

### Function Calling

Enable LLM to call your functions:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)
```

## Common Patterns

### 1. Classification with Enums

```python
from enum import Enum

class Category(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"

class TicketClassification(BaseModel):
    category: Category
    priority: int  # 1-5
    requires_escalation: bool
```

### 2. Extraction

```python
class ContactInfo(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None

# Extract structured data from unstructured text
```

### 3. Validation

```python
from pydantic import Field, field_validator

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, le=1000000)

    @field_validator('name')
    def name_must_not_contain_special_chars(cls, v):
        if not v.replace(' ', '').isalnum():
            raise ValueError('Name must be alphanumeric')
        return v
```

## Cost Management

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Speed |
|-------|---------------------|----------------------|-------|
| gpt-4o | $2.50 | $10.00 | Medium |
| gpt-4o-mini | $0.15 | $0.60 | Fast |
| gpt-3.5-turbo | $0.50 | $1.50 | Fast |

**Token counting:**
```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = encoding.encode("Your text here")
cost = len(tokens) * 0.00000015  # $0.15 per 1M tokens
```

## Error Handling

```python
from openai import OpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_openai_with_retry(prompt):
    try:
        response = client.chat.completions.create(...)
        return response
    except RateLimitError:
        # Wait and retry
        raise
    except APIError as e:
        # Log and handle
        raise
```

## Documentation & Resources

- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
- [Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Token Counting](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)

## Self-Assessment Checklist

- [ ] I can make basic chat completion requests
- [ ] I understand structured outputs with Pydantic
- [ ] I can implement function calling
- [ ] I know how to count tokens and estimate costs
- [ ] I can handle API errors and rate limits
- [ ] I understand when to use streaming vs non-streaming

---

## Deep Dive Q&A (30 Questions)

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

[Q6-Q10 continue with architecture questions...]

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

[Q12-Q20 continue with implementation questions about function calling, streaming, error handling, etc.]

### Debugging & Troubleshooting (21-25)

[Questions about debugging API issues, handling errors, etc.]

### Trade-offs & Decisions (26-30)

[Questions about model selection, cost optimization, etc.]

---

## Additional Resources

- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Pydantic Best Practices](https://docs.pydantic.dev/latest/concepts/models/)
- [Token Optimization Guide](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)
- [Production Patterns](https://github.com/openai/openai-cookbook/tree/main/examples)
