# Module 17: LLM-Driven Dataset Generation — 30 Deep Q&A

---

## Architecture & Design (10 Questions)

**Q1. Why use LLMs for dataset generation instead of manual annotation?**

Manual annotation is expensive, slow, and requires domain expertise. LLMs offer:
- **Speed**: Annotate 1000 examples in minutes vs weeks for human annotators
- **Cost**: $1–10 for a full dataset vs $500–5000 for human annotation
- **Consistency**: Same model, same prompt = consistent annotation style
- **Scale**: Easy to regenerate with improved prompts or more examples

Tradeoff: LLM annotations may have errors (~5-15% noise), but for fine-tuning that's often acceptable since the base model already has strong priors.

---

**Q2. What is the GLiNER tokenization format and why does it matter?**

GLiNER uses **word-level tokenization** via regex `\w+(?:[-_]\w+)*|\S`:
- Words with hyphens/underscores stay together: `CVE-2023-4966` → `["CVE-2023-4966"]`
- Regular punctuation splits: `3.0` → `["3", ".", "0"]`
- All non-word characters are single tokens: `!`, `,`, `@`

This matters because entity spans are token indices, not character offsets. If you tokenize differently from GLiNER's actual tokenizer, all spans will be misaligned and training will fail.

---

**Q3. Why must `all_labels` contain both true and false labels in GLiClass training data?**

GLiClass is trained with contrastive loss: it learns to score true labels higher than false labels for a given text. If `all_labels` contains only true labels, there's nothing to contrast against — the model can achieve zero loss by assigning high scores to everything.

False labels (negative examples) are essential for the model to learn discrimination: "this text is about ransomware, NOT about phishing or DDoS." Without negatives, fine-tuning will degrade zero-shot performance.

---

**Q4. What are the failure modes of LLM-based annotation for NER?**

1. **Entity not in text**: LLM hallucinates an entity that doesn't appear verbatim → `find_token_span` returns `None` → span is silently dropped (correct behavior)
2. **Partial matches**: LLM extracts "LockBit 3.0" but the text has "LockBit 3.0 ransomware" — both are valid, but we only capture exact span
3. **Wrong entity type**: LLM classifies "phishing" (attack_technique) as "malware" → incorrect annotation
4. **Merged entities**: LLM outputs "Citrix NetScaler ADC vulnerability" when the entity is just "Citrix NetScaler ADC" → span search fails
5. **Missing entities**: LLM misses some entities, especially rare/specialized ones

---

**Q5. How does `temperature=0` affect LLM annotation quality?**

`temperature=0` makes generation deterministic (greedy decoding):
- **Pros**: Reproducible annotations; same text always produces same output; easier to debug
- **Cons**: May miss valid entities that require slight reasoning variation

For annotation tasks, `temperature=0` is almost always preferred because:
1. Consistency matters more than creativity
2. You want to reproduce results (same input → same output)
3. Slightly lower recall is acceptable if it improves precision

---

**Q6. How do you handle LLM responses that contain invalid entity types?**

Filter them during parsing:
```python
if entity_type in ENTITY_TYPES:
    entities.append((entity_text, entity_type))
# else: silently drop
```

Never let invalid types into the dataset — GLiNER model only knows the entity types it was trained on. Out-of-vocabulary types would confuse the model. Similarly for GLiClass, filter out any labels not in `ALL_LABELS`.

---

**Q7. What is the difference between span-level and character-level annotation?**

| Format | Used by | Example |
|--------|---------|---------|
| Character offsets | spaCy, Hugging Face NER | `{"start": 4, "end": 11, "label": "malware"}` |
| Token spans (inclusive) | GLiNER | `[1, 4, "threat_actor"]` |
| Token spans (exclusive end) | HF tokenizers | `[1, 5, "threat_actor"]` |

GLiNER uses **inclusive token spans**: both start and end indices are part of the entity. `[1, 4]` means tokens 1, 2, 3, 4 are all part of the entity. This must be precise — off-by-one errors corrupt training data.

---

**Q8. How do you generate negative examples programmatically?**

The LLM generates negatives by asking for "labels that do NOT apply." Alternative approaches:
1. **LLM-generated** (this module): reliable but costs API calls
2. **Random sampling**: pick K random labels not in true_labels — simple but may create trivially easy negatives
3. **Hard negatives**: use a classifier to find labels that score high but are wrong — requires a trained model
4. **Frequency-based**: sample from the most common labels in the dataset — creates realistic negatives

For GLiClass fine-tuning, LLM-generated negatives are best because they're semantically plausible (the model must learn real distinctions).

---

**Q9. What is `response_format={"type": "json_object"}` and when to use it?**

This parameter forces the OpenAI model to output syntactically valid JSON. Use it when:
- You need structured data (dicts, lists) from the LLM
- You don't want to handle JSON parse errors from malformed output
- The output structure is predictable (known keys)

Do NOT use it when:
- You need free-text output (NER entity lines in `entity_text <> entity_type` format)
- You're using models that don't support it (older models)
- The structure varies per response

Note: You must also instruct the model to return JSON in the system prompt — the parameter alone doesn't guarantee the right keys.

---

**Q10. How would you scale this pipeline to annotate 10,000 reports?**

Optimizations for large-scale annotation:
1. **Async batching**: Use `asyncio` + `client.chat.completions.acreate()` — process 50+ requests concurrently
2. **Rate limiting**: Implement exponential backoff with `tenacity` for 429 errors
3. **Caching**: Cache LLM responses (hash text → response) to avoid re-annotating
4. **Batch validation**: Run quality checks periodically, not after every example
5. **Cost estimation**: `tiktoken` to count tokens upfront; `gpt-4o-mini` at ~$0.15/1M input tokens
6. **Deduplication**: Remove near-duplicate texts before annotation

For 10K examples at ~300 tokens/text: ~3M tokens → ~$0.45 with `gpt-4o-mini`.

---

## Implementation & Coding (10 Questions)

**Q11. How do you implement `find_token_span`?**

```python
def find_token_span(tokenized_text, entity_text):
    entity_tokens = tokenize(entity_text)  # same tokenizer!
    n = len(entity_tokens)
    for i in range(len(tokenized_text) - n + 1):
        window = [t.lower() for t in tokenized_text[i:i + n]]
        if window == [t.lower() for t in entity_tokens]:
            return (i, i + n - 1)  # inclusive
    return None
```

Key details: (1) use the same tokenizer for text and entity; (2) case-insensitive matching for robustness; (3) return `None` when not found — never guess.

---

**Q12. How do you parse the `entity_text <> entity_type` LLM output format?**

```python
def parse_ner_output(raw_output, valid_types):
    entities = []
    for line in raw_output.strip().split('\n'):
        if '<>' not in line:
            continue  # skip headers, empty lines, explanations
        parts = line.split('<>')
        if len(parts) == 2:
            entity_text = parts[0].strip()
            entity_type = parts[1].strip().lower()
            if entity_text and entity_type in valid_types:
                entities.append((entity_text, entity_type))
    return entities
```

---

**Q13. How do you build a full GLiNER example from a raw text?**

```python
def build_gliner_example(client, text):
    tokenized_text = tokenize(text)
    raw_output = call_llm_ner(client, text)
    entities = parse_ner_output(raw_output)
    ner_spans = []
    for entity_text, entity_type in entities:
        span = find_token_span(tokenized_text, entity_text)
        if span:  # skip entities not found in text
            ner_spans.append([span[0], span[1], entity_type])
    if not ner_spans:
        return None  # skip examples with no found entities
    return {"tokenized_text": tokenized_text, "ner": ner_spans}
```

---

**Q14. How do you validate a NER dataset before training?**

```python
def validate_ner_dataset(dataset, valid_entity_types):
    errors = []
    for i, ex in enumerate(dataset):
        n = len(ex['tokenized_text'])
        for span in ex['ner']:
            s, e, label = span
            if not (0 <= s <= e < n):
                errors.append(f"Ex {i}: invalid span [{s},{e}] for {n} tokens")
            if label not in valid_entity_types:
                errors.append(f"Ex {i}: unknown type '{label}'")
    return errors  # empty = all valid
```

---

**Q15. How do you add retry logic for OpenAI API calls?**

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError

@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5)
)
def call_llm_ner(client, text):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        temperature=0
    )
```

---

**Q16. How do you process many texts in parallel with async OpenAI calls?**

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(api_key=api_key)

async def annotate_async(text):
    resp = await async_client.chat.completions.acreate(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
        temperature=0
    )
    return resp.choices[0].message.content

async def build_dataset_async(texts, concurrency=20):
    semaphore = asyncio.Semaphore(concurrency)
    async def limited(text):
        async with semaphore:
            return await annotate_async(text)
    return await asyncio.gather(*[limited(t) for t in texts])

results = asyncio.run(build_dataset_async(texts))
```

---

**Q17. How do you estimate the cost of annotating a dataset?**

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
INPUT_PRICE = 0.15 / 1_000_000   # $ per token
OUTPUT_PRICE = 0.60 / 1_000_000  # $ per token

total_input = sum(len(enc.encode(SYSTEM_PROMPT + text)) for text in texts)
total_output = len(texts) * 50  # estimated output tokens per call

cost = total_input * INPUT_PRICE + total_output * OUTPUT_PRICE
print(f"Estimated cost: ${cost:.4f} for {len(texts)} texts")
```

---

**Q18. How do you handle duplicate or near-duplicate entities in a single NER example?**

```python
def deduplicate_spans(ner_spans):
    """Remove duplicate spans (same start, end, label)."""
    seen = set()
    unique = []
    for span in ner_spans:
        key = (span[0], span[1], span[2])
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique
```

Also handle overlapping spans — GLiNER typically handles them internally, but clean training data is better.

---

**Q19. How do you merge NER datasets from multiple annotation runs?**

```python
def merge_ner_datasets(datasets):
    """Merge multiple NER datasets, deduplicating by tokenized_text."""
    seen_texts = set()
    merged = []
    for dataset in datasets:
        for ex in dataset:
            key = tuple(ex['tokenized_text'])
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(ex)
    return merged
```

---

**Q20. How do you split a generated dataset into train/eval for GLiNER training?**

```python
import random

def split_dataset(dataset, eval_ratio=0.1, seed=42):
    random.seed(seed)
    data = dataset.copy()
    random.shuffle(data)
    split = int(len(data) * (1 - eval_ratio))
    return data[:split], data[split:]

train_data, eval_data = split_dataset(ner_dataset, eval_ratio=0.1)
```

For GLiNER, ensure both splits have examples of each entity type.

---

## Debugging & Troubleshooting (5 Questions)

**Q21. Why do all `find_token_span` calls return `None` for a multi-word entity?**

Most common causes:
1. **Tokenizer mismatch**: You tokenized the entity differently from the text. Always use the same `tokenize()` function for both.
2. **Whitespace in entity**: Extra space before/after → `tokenize("  LockBit ")` vs `tokenize("LockBit")` — both produce `["LockBit"]` with the regex, so this is usually fine.
3. **LLM hallucinated entity**: The entity doesn't appear in the source text. Debug with `print(entity_text, tokenize(entity_text))`.
4. **Case mismatch without lowering**: If not using `.lower()` in comparison.

Debug: `print(tokenize("exact entity from LLM"))` and check if the tokens appear consecutively in `tokenize(full_text)`.

---

**Q22. Why does `response_format={"type": "json_object"}` still sometimes fail?**

The model returns valid JSON but with wrong keys. Solutions:
```python
data = json.loads(response.content)
true_labels = data.get('true_labels', [])  # use .get() not data['true_labels']
if not isinstance(true_labels, list):
    true_labels = [true_labels]  # handle string instead of list
```

Also: the model might use `labels` instead of `true_labels`. Always validate keys before using.

---

**Q23. Why is the NER dataset small after annotation (many `None` returns)?**

`build_gliner_example` returns `None` when no entity spans are found. Check:
1. Are all `find_token_span` returning `None`? → tokenizer issue
2. Is `parse_ner_output` returning empty? → prompt issue or LLM outputting wrong format
3. Is the entity type filtering too strict? → relax `if entity_type in ENTITY_TYPES`

Debug: add logging in `build_gliner_example` to print raw LLM output and parsed entities.

---

**Q24. Why does the GLiClass validation fail with `true_labels not in all_labels`?**

The LLM returned a label in `true_labels` that it also listed in `false_labels` — or the label got filtered out from `all_labels` due to being invalid. Fix:
```python
# Ensure true_labels ⊆ all_labels regardless of LLM inconsistency
all_labels = list(set(true_labels + false_labels))
# If LLM contradicts itself, true_labels always win (they're in all_labels by construction)
```

---

**Q25. How do you detect annotation quality issues in a generated dataset?**

Quality signals to monitor:
- **Span coverage**: avg entities per example < 1 → prompt is too strict or texts too short
- **Type imbalance**: one type dominates (>70%) → prompts may be biased
- **Short entities**: many 1-token entities → LLM may not be capturing full phrases
- **False label overlap**: `false_labels` contains items from other examples' `true_labels` → good sign (hard negatives)
- **Empty all_labels**: `all_labels` has fewer than 3 labels → weak supervision signal

---

## Trade-offs & Decisions (5 Questions)

**Q26. When should you use `gpt-4o-mini` vs `gpt-4o` for annotation?**

| | gpt-4o-mini | gpt-4o |
|---|---|---|
| Cost | ~15x cheaper | Expensive |
| NER quality | Good for standard entities | Better for complex/nested |
| Classification | Very good | Best |
| Speed | Faster | Slower |
| Recommendation | First choice for annotation | Use for ambiguous/complex cases |

Start with `gpt-4o-mini`. If validation shows >15% error rate, upgrade to `gpt-4o` for a subset of difficult examples.

---

**Q27. How do you choose what to include in `all_labels` for GLiClass training?**

Too few labels (≤2): the model doesn't learn real discrimination — easy task.
Too many labels (≥10): the model struggles to identify which labels are relevant per text.

Best practice:
- **True labels**: 1-3 per example (LLM decides)
- **False labels**: 2-4 semantically plausible but incorrect labels per example
- **Total per example**: 4-7 labels

This gives enough signal for discrimination without overwhelming the model. The LLM should choose false labels that are "almost right" — e.g., for a ransomware attack, `data_breach` is a good false label (often co-occurs) while `cooking_recipe` is too easy.

---

**Q28. Should you generate annotations once and save them, or regenerate each training run?**

Save generated annotations. Reasons:
1. **Reproducibility**: same dataset for all experiments
2. **Cost**: avoid paying for the same API calls repeatedly
3. **Debugging**: inspect what the model was trained on
4. **Iteration**: manually correct bad annotations without regenerating everything

Save to JSON, version-control the fixture file. Only regenerate when you change the prompt significantly or add new raw texts.

---

**Q29. How do you handle domain shift when adapting this pipeline to a new domain?**

1. **Update entity types**: replace cybersecurity types with domain-specific ones (e.g., medical: `drug`, `disease`, `symptom`)
2. **Update system prompt**: add domain context, examples of correct annotations
3. **Adjust false label strategy**: domain-specific "hard negatives" — similar categories that require real understanding to distinguish
4. **Validate with domain expert**: sample 50 examples and have a human check quality before training
5. **Iterative refinement**: run annotation → train model → find errors → improve prompt → repeat

---

**Q30. When is LLM-generated data good enough vs when do you need human annotations?**

| Use LLM-generated | Use human annotations |
|---|---|
| Prototyping, proof-of-concept | Production-critical models |
| Standard entity types (well-known) | Rare/ambiguous entities |
| High-resource languages (English) | Low-resource languages |
| Clear, objective annotations | Subjective annotations (sentiment) |
| > 500 examples needed quickly | < 200 examples, high quality needed |
| Budget < $50 | Budget for $500+ annotation |

A hybrid approach often works best: LLM generates 500 examples (cheap), human annotators review and correct 50 (quality check), train on the cleaned set.
