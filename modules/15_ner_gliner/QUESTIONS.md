# Module 15: NER with GLiNER — Interview Questions

## Architecture & Design (10 questions)

**Q1. What is GLiNER and how does it differ from traditional NER models like spaCy or BERT-NER?**

Traditional NER models are trained to predict from a fixed set of entity types (PERSON, ORG, LOC). They cannot recognize entity types not seen during training — adding a new category requires full retraining. GLiNER is a zero-shot model: instead of classifying spans into a closed label set, it takes entity type descriptions as input at inference time and finds spans that semantically match. This means you can extract `["spacecraft", "chemical compound", "legislation"]` without any fine-tuning.

---

**Q2. Explain the difference between uni-encoder and bi-encoder GLiNER architectures.**

In the **uni-encoder** architecture, the text and all entity labels are concatenated into a single sequence and processed jointly by a transformer. This gives accurate cross-attention between text tokens and label tokens, but means every inference pass must re-encode all labels — computational cost scales linearly with label count.

In the **bi-encoder** architecture, two independent transformers process text and labels separately. The text encoder produces contextualized token embeddings; the label encoder produces label embeddings independently. Entity matching uses dot-product similarity between token and label embeddings. Because labels are independent of the text, they can be **pre-computed once** and cached. Inference time then depends only on text length, not label count.

---

**Q3. What is the key computational advantage of pre-computing label embeddings in GLiNER-bi?**

When processing millions of documents against a fixed taxonomy (e.g., medical ontologies with thousands of entity types), you encode each label once instead of on every document. The label encoder forward pass (MiniLM) is expensive relative to the matching step. With pre-computed embeddings, the bottleneck is purely the text encoder. Benchmarks show ~130× throughput improvement at 1024 entity types vs. uni-encoder, with near-constant inference time from 1 to 1024 labels.

---

**Q4. What are `ner_labels` and `ner_negatives` in the GLiNER training format, and why are they useful for bi-encoders?**

`ner_labels` specifies the pool of entity type candidates the model sees for a given training example — positives (labels that appear in `ner`) plus extra types for the model to practice rejecting. `ner_negatives` specifies **hard negatives**: entity types semantically similar to the positives that the model should distinguish (e.g., `person` vs. `character`, `organization` vs. `institution`). For bi-encoders this is especially valuable because the label encoder must produce discriminative embeddings — hard negatives push the model to separate similar concepts in embedding space rather than just learning easy positives.

---

**Q5. Why does GLiNER use word-level (whitespace) token indices in training data instead of subword indices?**

GLiNER's span detection operates at the word level — it predicts which words constitute an entity, then maps back to the original text. Using word-level indices in training data keeps the format human-readable and annotation-tool-agnostic. Most annotation tools produce character-level or word-level spans rather than subword spans. Subword tokenization (BPE/WordPiece) happens internally and is an implementation detail of the encoder — the training format abstracts over it.

---

**Q6. What are the main PII categories GLiNER-pii is designed for, and under which compliance regimes are they relevant?**

The model targets three compliance domains:
- **PII** (Personally Identifiable Information) — GDPR, CCPA: name, email, phone, address, SSN, DOB
- **PHI** (Protected Health Information) — HIPAA: medical record number, condition, drug, blood type, healthcare facility
- **PCI** (Payment Card Industry) — PCI-DSS: credit card number, CVV, expiration date, bank account, routing number

The model uses a single unified interface for all three via zero-shot label specification, making it useful for cross-domain compliance pipelines.

---

**Q7. Why must entity spans be replaced end-to-start during text anonymization?**

`predict_entities` returns character offsets into the original string. When you replace a span with a placeholder of different length (e.g., `"John Smith"` → `"<NAME>"`), all character offsets *after* that replacement shift. If you replace left-to-right, the offsets for later entities (which still point into the original string) are now wrong relative to the modified string. Replacing from the rightmost span first leaves all left-side offsets unchanged, making subsequent replacements accurate.

---

**Q8. How does GLiNER handle overlapping entity spans?**

GLiNER uses a non-overlapping span selection algorithm (NMS-style). For each token position, it considers all possible spans up to a maximum width (`max_span_width`, default 12 tokens) and scores them. During post-processing, it selects the highest-scoring spans and suppresses overlapping lower-confidence spans. This is a design choice — true nested entities (e.g., `[organization [location]]`) are not supported in the base model; for nested NER, multi-task GLiNER variants are used.

---

**Q9. What is the role of `threshold` in `predict_entities` and how should you set it for different use cases?**

`threshold` is the minimum cosine similarity (or dot-product score) between a token span embedding and a label embedding for the span to be returned as a detected entity. Lower threshold → higher recall, more false positives. Higher threshold → higher precision, fewer false positives. For compliance/PII use cases, prefer lower thresholds (0.2–0.3) since missing a sensitive entity (false negative) is riskier than over-redacting (false positive). For structured information extraction where precision matters (e.g., populating a knowledge base), use 0.4–0.6.

---

**Q10. How does GLiNER differ from using an LLM (GPT-4, Claude) for NER?**

| Dimension | GLiNER | LLM |
|-----------|--------|-----|
| Speed | ~10–30 ms per doc (CPU) | 200–2000 ms per doc |
| Cost | Free, local | Per-token API cost |
| Privacy | Fully on-premise | Data sent to cloud |
| Output format | Structured spans with offsets | Free-form text, needs parsing |
| Hallucination | None — span detection only | Can hallucinate entities |
| Entity count limit | 1000+ with bi-encoder | Prompt window limited |
| Zero-shot flexibility | High | Very high (instructions) |

For production PII anonymization and entity extraction at scale, GLiNER is preferable. LLMs are better for complex extraction tasks requiring reasoning about context.

---

## Implementation & Coding (10 questions)

**Q11. Write code to load `gliner-bi-edge-v2.0` and extract `person` and `organization` entities from a text.**

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-bi-edge-v2.0")

text = "Sam Altman leads OpenAI, which is backed by Microsoft."
labels = ["person", "organization"]

entities = model.predict_entities(text, labels, threshold=0.3)
for e in entities:
    print(f"{e['text']!r} => {e['label']} ({e['score']:.2f})")
    # Verify offset
    assert text[e["start"]:e["end"]] == e["text"]
```

---

**Q12. Implement batch NER using pre-computed label embeddings.**

```python
def batch_ner_cached(model, texts, labels, threshold=0.3):
    # Encode labels once
    embeddings = model.encode_labels(labels, batch_size=8)
    # Inference with cached embeddings
    return model.batch_predict_with_embeds(texts, embeddings, labels, threshold=threshold)
```

This is the production-recommended pattern when processing many documents against a static label set.

---

**Q13. Implement `anonymize_text` that replaces PII with `<LABEL>` placeholders.**

```python
def anonymize_text(model, text, labels, threshold=0.3):
    entities = model.predict_entities(text, labels, threshold=threshold)
    entities.sort(key=lambda e: e["start"], reverse=True)  # end-to-start
    result = text
    for e in entities:
        tag = "<" + e["label"].upper().replace(" ", "_") + ">"
        result = result[:e["start"]] + tag + result[e["end"]:]
    return result
```

---

**Q14. Convert character-level entity annotations to GLiNER training format.**

```python
import re

def to_gliner_format(text, char_entities):
    token_matches = list(re.finditer(r'\S+', text))
    tokens = [m.group() for m in token_matches]
    token_spans = [(m.start(), m.end()) for m in token_matches]
    ner = []
    for ent in char_entities:
        toks = [i for i, (ts, te) in enumerate(token_spans)
                if ts >= ent["start"] and te <= ent["end"]]
        if toks:
            ner.append([toks[0], toks[-1], ent["label"]])
    return {"tokenized_text": tokens, "ner": ner}
```

---

**Q15. Write the `model.train_model()` call for bi-encoder fine-tuning.**

```python
train_split = train_data[:int(len(train_data) * 0.8)]
eval_split  = train_data[int(len(train_data) * 0.8):]

trainer = model.train_model(
    train_split,
    eval_split,
    output_dir="./checkpoints",
    max_steps=2000,
    learning_rate=1e-5,       # lower for encoder
    others_lr=3e-5,           # higher for span/matching heads
    per_device_train_batch_size=8,
    negatives=1.5,            # higher negative sampling for bi-encoder
    warmup_ratio=0.1,
    save_steps=500,
    save_total_limit=2,
)
trainer.model.save_pretrained("./checkpoints")  # save final model (not just checkpoint)
```

---

**Q16. How do you load and use a fine-tuned GLiNER model?**

```python
# After training saves to ./checkpoints/
finetuned = GLiNER.from_pretrained("./checkpoints")
entities = finetuned.predict_entities(text, labels, threshold=0.3)
```

`from_pretrained` works for both HuggingFace Hub model IDs and local directory paths.

---

**Q17. How do you validate that a GLiNER training dataset is correctly formatted?**

```python
def validate(examples):
    errors = []
    for i, ex in enumerate(examples):
        n = len(ex.get("tokenized_text", []))
        for span in ex.get("ner", []):
            if len(span) != 3:
                errors.append(f"Ex {i}: span must have 3 elements")
            else:
                start, end, label = span
                if not (0 <= start <= end < n):
                    errors.append(f"Ex {i}: [{start},{end}] out of bounds (n={n})")
    return errors
```

---

**Q18. How do you batch-anonymize documents efficiently?**

```python
def anonymize_batch(model, texts, labels, threshold=0.3):
    all_entities = model.run(texts, labels, threshold=threshold)
    results = []
    for text, entities in zip(texts, all_entities):
        entities.sort(key=lambda e: e["start"], reverse=True)
        result = text
        for e in entities:
            tag = "<" + e["label"].upper().replace(" ", "_") + ">"
            result = result[:e["start"]] + tag + result[e["end"]:]
        results.append(result)
    return results
```

Using `model.run()` processes all texts in a single batched forward pass instead of N separate calls.

---

**Q19. How do you enrich training examples with `ner_labels` and `ner_negatives`?**

```python
def enrich(example, all_labels, n_neg=3):
    pos = list({span[2] for span in example["ner"]})
    neg = [l for l in all_labels if l not in pos]
    return {
        **example,
        "ner_labels": pos + neg[:n_neg],
        "ner_negatives": neg[n_neg:n_neg*2],
    }
```

---

**Q20. How do you use GLiNER to extract entities with custom domain-specific labels?**

```python
# Medical domain
entities = model.predict_entities(
    medical_note,
    labels=["medical procedure", "drug name", "dosage", "diagnosis", "anatomy"],
    threshold=0.3
)

# Legal domain
entities = model.predict_entities(
    contract,
    labels=["party name", "obligation", "effective date", "jurisdiction", "penalty"],
    threshold=0.3
)
```

Label descriptions are encoded by the label encoder — descriptive phrases work better than cryptic abbreviations.

---

## Debugging & Troubleshooting (5 questions)

**Q21. The model finds no entities even though the text clearly contains them. What do you check?**

1. **Threshold too high** — lower to 0.2 and check if entities appear with low scores
2. **Label mismatch** — try synonyms: `"company"` vs. `"organization"` vs. `"corporation"`; the label encoder is semantic but specific phrasings matter
3. **Text too long** — GLiNER has a max context length (1024 tokens for bi-edge-v2.0); truncate or chunk long documents
4. **Wrong model variant** — PII model for general NER, or general model for PII (the PII model is fine-tuned on PII examples and may give different scores)
5. **Span width** — entities longer than `max_span_width` (default 12 tokens) are never returned; compound names or long titles may be missed

---

**Q22. After fine-tuning, the model performs worse than the base model. What are the likely causes?**

1. **Too many steps / learning rate too high** — overfitting to 20 examples; reduce `max_steps` or lower `learning_rate`
2. **Not enough label diversity** — if training data only has `person`/`org`, the model may lose ability on other types; add `ner_labels` with diverse candidates
3. **Missing negatives** — without `ner_negatives`, the model doesn't learn to discriminate similar types; add hard negatives
4. **Wrong tokenization** — token indices in training data don't match actual spans; validate with `validate_training_data()`
5. **Catastrophic forgetting** — freeze the text encoder with `freeze_components=["text_encoder"]` to preserve general representations

---

**Q23. The anonymized text has garbled characters after some replacements. What's wrong?**

The replacements are being done left-to-right instead of right-to-left. Each replacement changes the string length, shifting the character offsets of subsequent entities. Fix: sort entities by `start` in descending order before replacing.

```python
entities.sort(key=lambda e: e["start"], reverse=True)  # end-to-start
```

---

**Q24. Training throws an `IndexError` about span indices being out of range. How do you diagnose it?**

The `ner` span indices reference token positions outside `tokenized_text`. Common causes:
- Character-to-token conversion bug (entity boundary doesn't align with a token boundary)
- Off-by-one in end index (GLiNER uses inclusive end; if you subtract 1 by mistake, the span is too short)
- Entity text contains leading/trailing whitespace that shifts the match

Run `validate_training_data(examples)` to find the specific example and span, then print `tokenized_text` and the failing span to inspect visually.

---

**Q25. `model.encode_labels()` returns embeddings but `batch_predict_with_embeds()` gives different results than `predict_entities()`. Why?**

Small differences are expected due to internal batching and numerical precision. However, large differences indicate:
1. **Threshold mismatch** — ensure you pass the same `threshold` value to both calls
2. **Label order mismatch** — `batch_predict_with_embeds` requires that `entity_embeddings` rows correspond to `labels` in the same order as passed to `encode_labels`
3. **Device mismatch** — embeddings on CPU, model on GPU (or vice versa); ensure both are on the same device

---

## Trade-offs & Decisions (5 questions)

**Q26. When would you use GLiNER-bi vs. GLiNER uni-encoder for a production NER pipeline?**

Use **bi-encoder** when:
- You have more than ~32 entity types
- The label set is static (can pre-compute embeddings)
- Throughput is the primary concern (millions of documents)
- You need to add new entity types without recomputing anything per-document

Use **uni-encoder** when:
- You have very few entity types (< 16) and the joint cross-attention helps accuracy
- Maximum precision on a small fixed schema matters more than throughput
- The entity types change frequently per document (no caching benefit)

---

**Q27. For GDPR-compliant log anonymization, should you prioritize precision or recall? What threshold does this imply?**

**Recall must be prioritized**. A missed SSN or email address in a log file is a compliance violation; over-redacting a non-PII word is an acceptable false positive. Use thresholds of 0.2–0.3 and accept some false positives. Pair with a human review step for edge cases. Also prefer the larger PII model variants (`gliner-pii-base-v1.0` or larger) if latency allows — higher F1 at lower thresholds.

---

**Q28. When is fine-tuning GLiNER worth the investment vs. staying with zero-shot inference?**

Fine-tuning is worth it when:
- Zero-shot misses consistently fail on a specific entity type that appears frequently in your domain (e.g., proprietary product names, internal org unit names)
- You have at least 100–500 labeled examples of the failing type
- The entity types are highly domain-specific (rare in general web text used for pre-training)

Zero-shot is sufficient when:
- The entity types map cleanly to common English descriptions
- Accuracy gaps are tolerable
- Your entity taxonomy changes frequently (retraining overhead is prohibitive)

---

**Q29. How does GLiNER-PII compare to rule-based approaches (regex) for PII detection?**

| Dimension | GLiNER-PII | Regex/Rules |
|-----------|-----------|-------------|
| Name detection | ✓ Semantic, handles variations | ✗ Requires name lists |
| Structured PII (SSN, phone) | ✓ Good | ✓ Excellent (precise patterns) |
| Informal PII | ✓ Handles paraphrases | ✗ Misses non-standard formats |
| False positives | Some at low threshold | Low for well-crafted patterns |
| Maintenance | Model update, no rule changes | Manual pattern updates |
| New entity types | Zero-shot | New regex per type |

**Best practice**: use GLiNER-PII for semantic entities (names, conditions, organizations) and complement with regex for structured patterns (credit cards, phone numbers, SSNs) where format is deterministic.

---

**Q30. What are the risks of using char-level offset replacement for anonymization, and how do you mitigate them?**

**Risks:**
1. **Overlapping entities** — if two detected spans overlap (rare but possible), naive end-to-start replacement may skip one or corrupt text. Mitigation: remove overlapping spans by keeping the highest-confidence one before replacement.
2. **Unicode characters** — Python string indices are by Unicode code point, but some tokenizers use byte offsets. Verify that your GLiNER version and the text encoding are consistent.
3. **Whitespace inconsistency** — if the input text was normalized (tabs→spaces, etc.) before inference but original text is used for replacement, offsets may be off. Always run inference on the exact string used for replacement.
4. **Incomplete detection** — anonymization is only as good as the model's recall; always validate anonymized output in sensitive pipelines. Consider adding regex-based post-processing for high-value patterns (SSN, credit card).
