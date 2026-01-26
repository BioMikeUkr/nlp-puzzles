# System Design Problems

50 end-to-end scenarios covering complex ML/NLP system design.

## How to Use

Each problem file contains:
1. **Problem Statement** - what you need to build
2. **Requirements** - scale, performance, constraints
3. **Questions to Answer** - architecture, tech choices, trade-offs
4. **Solution** - detailed answer with diagrams

## Practice Approach

1. Read the problem and requirements
2. Take 5-10 minutes to sketch your solution
3. Compare with the provided solution
4. Note trade-offs you missed

---

## Problem Index

### Search & Retrieval (1-10)
| # | Problem | Key Topics |
|---|---------|------------|
| 01 | Ticket Search System | Semantic search, vector DB, embeddings |
| 02 | Document Similarity Detection | Deduplication, similarity threshold |
| 03 | Duplicate Detection Pipeline | LSH, near-duplicate, batch processing |
| 04 | Multi-language Search | Cross-lingual embeddings, language detection |
| 05 | Hybrid Search System | BM25 + semantic, fusion ranking |
| 06 | Real-time Search Indexing | Streaming updates, index refresh |
| 07 | Personalized Search Ranking | User preferences, learning to rank |
| 08 | Query Understanding System | Intent detection, query expansion |
| 09 | Auto-complete with Semantics | Prefix search, semantic suggestions |
| 10 | Cross-document Linking | Entity linking, knowledge graph |

### Classification & NLP (11-20)
| # | Problem | Key Topics |
|---|---------|------------|
| 11 | Ticket Routing System | Multi-class classification, confidence |
| 12 | Sentiment Analysis Pipeline | Fine-tuning, aspect sentiment |
| 13 | Intent Classification | Few-shot, out-of-scope detection |
| 14 | Named Entity Extraction | NER, custom entities, validation |
| 15 | Topic Modeling at Scale | LDA, BERTopic, dynamic topics |
| 16 | Spam/Fraud Detection | Imbalanced data, real-time scoring |
| 17 | Priority Classification | Ordinal classification, business rules |
| 18 | Multi-label Classification | Threshold tuning, label correlation |
| 19 | Language Detection | Short text, code-switching |
| 20 | Content Moderation | Toxicity, PII detection, appeals |

### RAG & Generation (21-30)
| # | Problem | Key Topics |
|---|---------|------------|
| 21 | Documentation Q&A System | RAG, chunking, citation |
| 22 | Automated Response Generation | Templates, personalization |
| 23 | Report Summarization | Long document, extractive + abstractive |
| 24 | Chat with Documents | Conversation memory, follow-ups |
| 25 | Knowledge Base Assistant | Multi-source, disambiguation |
| 26 | Code Documentation Generator | Code understanding, docstrings |
| 27 | Meeting Notes Summarizer | Action items, speakers |
| 28 | Email Response Suggester | Context, tone matching |
| 29 | Compliance Checker | Rules engine, explanations |
| 30 | Contract Analysis | Clause extraction, risk scoring |

### Data Pipelines (31-40)
| # | Problem | Key Topics |
|---|---------|------------|
| 31 | Document Processing Pipeline | PDF, OCR, quality checks |
| 32 | Real-time Data Enrichment | Stream processing, lookups |
| 33 | ETL for ML Features | Feature engineering, freshness |
| 34 | Data Quality Monitoring | Drift detection, alerts |
| 35 | Streaming Analytics | Windows, aggregations |
| 36 | Batch Embedding Generation | GPU utilization, checkpointing |
| 37 | Data Versioning System | Lineage, reproducibility |
| 38 | Feature Store Design | Online/offline, consistency |
| 39 | Log Analysis Pipeline | Parsing, anomaly detection |
| 40 | Anomaly Detection Stream | Time series, adaptive thresholds |

### MLOps & Infrastructure (41-50)
| # | Problem | Key Topics |
|---|---------|------------|
| 41 | Model Serving Architecture | Latency, scaling, versioning |
| 42 | A/B Testing Framework | Experiment design, metrics |
| 43 | Model Monitoring System | Drift, performance degradation |
| 44 | Retraining Pipeline | Triggers, validation, rollback |
| 45 | Multi-model Orchestration | Ensemble, fallback, routing |
| 46 | Cost Optimization System | Caching, batching, model selection |
| 47 | Caching Layer Design | Embedding cache, TTL, invalidation |
| 48 | Rate Limiting for ML APIs | Token buckets, fair queuing |
| 49 | Disaster Recovery for ML | Backup, failover, data recovery |
| 50 | Multi-region Deployment | Latency, consistency, compliance |

---

## Template

Each problem follows this structure:

```markdown
# Problem XX: [Name]

## Problem Statement
[Description of what needs to be built]

## Requirements
- Scale: [expected load]
- Latency: [SLA]
- Accuracy: [target metrics]

## Questions to Answer
1. What is your high-level architecture?
2. Which ML models/techniques would you use?
3. How do you handle [specific challenge]?
4. How do you measure success?

---

## Solution

### Architecture Diagram
[ASCII diagram]

### Data Flow
[Step by step]

### Technology Choices
| Component | Technology | Why |
|-----------|------------|-----|

### Trade-offs
[Discussion]

### Metrics & Monitoring
[What to track]
```
