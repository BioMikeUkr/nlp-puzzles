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


## Practice Questions

See [QUESTIONS.md](./QUESTIONS.md) for 30 deep-dive questions with detailed answers covering:
- Architecture & Design (Q1-Q10)
- Implementation & Coding (Q11-Q20)
- Debugging & Troubleshooting (Q21-Q25)
- Trade-offs & Decisions (Q26-Q30)

---

## Additional Resources

- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Pydantic Best Practices](https://docs.pydantic.dev/latest/concepts/models/)
- [Token Optimization Guide](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)
- [Production Patterns](https://github.com/openai/openai-cookbook/tree/main/examples)
