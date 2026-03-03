# Module 13 Fixtures

## `input/tickets.json`
20 support tickets with ground truth labels:
- `id`: unique integer
- `text`: ticket content
- `category`: `billing` | `technical` | `account` | `shipping`
- `priority`: `high` | `medium` | `low`

Distribution: 5 tickets per category, mixed priorities.

## `input/extraction_samples.json`
10 samples for entity extraction tasks:
- `id`: unique integer
- `text`: customer message
- `expected`: `{"product": str, "issue": str|null, "sentiment": "positive"|"negative"}`

## `input/edge_cases.json`
10 examples for testing prompt injection defenses:
- `id`: unique integer
- `text`: user message
- `is_injection`: `true` | `false`
- `category`: attack type or "normal"

Attack types: `direct_injection`, `role_override`, `memory_override`, `jailbreak`, `hidden_injection`
