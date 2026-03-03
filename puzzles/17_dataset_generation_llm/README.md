# Module 17: LLM-Driven Dataset Generation for GLiNER & GLiClass

## Overview

Training NER and text classification models requires annotated data — which is expensive to create manually. This module covers generating training datasets automatically using LLMs.

**Domain**: Cybersecurity threat intelligence

You generate two datasets from 20 raw security reports:
1. **NER dataset** for GLiNER — entity spans with types (`malware`, `vulnerability`, etc.)
2. **Classification dataset** for GLiClass — attack type labels (`ransomware`, `phishing`, etc.)

---

## Data Formats

### GLiNER Training Format
```json
{
  "tokenized_text": ["The", "LockBit", "3", ".", "0", "ransomware", ...],
  "ner": [[1, 4, "threat_actor"], [8, 8, "vulnerability"], ...]
}
```
- `tokenized_text`: word-level tokens (regex: `\w+(?:[-_]\w+)*|\S`)
- `ner`: `[start_token_idx, end_token_idx, entity_type]` — **inclusive, 0-indexed**

### GLiClass Training Format
```json
{
  "text": "The LockBit ransomware group exploited...",
  "all_labels": ["ransomware", "zero_day", "phishing", "ddos"],
  "true_labels": ["ransomware", "zero_day"]
}
```
- `all_labels`: true labels + false labels (negative examples required for training)
- `true_labels`: labels that actually apply

---

## NER Entity Types

| Type | Examples |
|------|----------|
| `malware` | LockBit, Emotet, SUNBURST, TrickBot |
| `vulnerability` | CVE-2023-4966, Log4Shell, ProxyNotShell |
| `attack_technique` | phishing, SQL injection, lateral movement |
| `affected_software` | Citrix NetScaler, Windows Server, Apache Log4j |
| `threat_actor` | APT29, Lazarus Group, FIN7, Killnet |

## Classification Labels

`ransomware` · `phishing` · `apt_attack` · `ddos` · `data_breach` · `supply_chain_attack` · `zero_day`

---

## Quick Start

```python
import re, json
from openai import OpenAI

client = OpenAI(api_key="your-key")

# Tokenizer (must match GLiNER's tokenizer)
def tokenize(text):
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

# NER: LLM returns "entity_text <> entity_type" lines
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract entities. Output: entity_text <> entity_type"},
        {"role": "user", "content": text}
    ],
    temperature=0
)

# Classification: LLM returns JSON with true/false labels
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Classify. Return JSON: {true_labels: [...], false_labels: [...]}"},
        {"role": "user", "content": text}
    ],
    temperature=0,
    response_format={"type": "json_object"}
)
```

---

## Key Patterns

### Finding Token Spans
```python
def find_token_span(tokenized_text, entity_text):
    entity_tokens = tokenize(entity_text)
    n = len(entity_tokens)
    for i in range(len(tokenized_text) - n + 1):
        if [t.lower() for t in tokenized_text[i:i+n]] == [t.lower() for t in entity_tokens]:
            return (i, i + n - 1)  # inclusive end index
    return None  # not found — skip this entity
```

### Building GLiClass Examples
```python
def build_gliclass_example(client, text):
    data = call_llm_cls(client, text)  # {"true_labels": [...], "false_labels": [...]}
    true_labels = [l for l in data["true_labels"] if l in ALL_LABELS]
    false_labels = [l for l in data["false_labels"] if l in ALL_LABELS]
    all_labels = list(set(true_labels + false_labels))  # deduplicate
    return {"text": text, "all_labels": all_labels, "true_labels": true_labels}
```

---

## Tasks

| Task | Description |
|------|-------------|
| `task_01_ner_annotation.ipynb` | Prompt LLM → parse entities → find token spans → GLiNER dataset |
| `task_02_classification_annotation.ipynb` | Prompt LLM → parse JSON → build all_labels → GLiClass dataset |

---

## Tips

- Use `temperature=0` for reproducible annotations
- Use `response_format={"type": "json_object"}` for classification to guarantee valid JSON
- Always validate spans: `0 <= start <= end < len(tokenized_text)`
- Entities not found in text (after tokenization) are silently skipped — verify LLM doesn't hallucinate
- `all_labels` must contain both true and false labels — GLiClass needs negatives to learn discrimination
- For production: add retry logic with `tenacity` for API rate limits
