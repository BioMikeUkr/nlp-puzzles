# Module 14 Fixtures

## `input/tickets.json`
20 labeled support tickets (same format as module 13):
- `id`, `text`, `category` (billing/technical/account/shipping), `priority` (high/medium/low)

## `input/knowledge_base.json`
10 company knowledge base articles:
- `id`, `title`, `content` (~200 words each)
- Topics: refunds, password reset, API rate limits, billing, shipping, security, mobile app, cancellation, team management, GDPR

## `input/test_questions.json`
5 RAG evaluation questions:
- `id`, `question`, `expected_keywords` (list of strings that should appear in a good answer), `relevant_article_id`
