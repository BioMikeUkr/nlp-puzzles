# Curriculum Plan: Practical NLP, ML & GenAI Engineering

## Overview

### Project Structure:
```
nlp-puzzles/
├── README.md                      # Project overview
├── curriculum_plan.md             # This file
│
├── system_design_problems/        # 50 complex end-to-end problems
│   ├── README.md                  # Index of all problems
│   ├── 01_ticket_search_system.md
│   ├── 02_document_processing_pipeline.md
│   ├── 03_realtime_classification.md
│   └── ...                        # Each with diagrams, data flow, tech choices
│
├── datasets/                      # Shared datasets (real or from HuggingFace)
│   ├── README.md                  # Dataset descriptions and sources
│   ├── download_datasets.py       # Script to download from HF
│   ├── tickets/                   # Support tickets dataset
│   ├── documents/                 # Financial documents
│   ├── embeddings/                # Pre-computed embeddings
│   └── sql/                       # SQL dumps for local DB
│
└── modules/
    └── module_name/
        ├── README.md              # Theory + 30 deep Q&A
        ├── fixtures/              # Module-specific sample data
        ├── learning/              # Learning notebooks
        └── tasks/                 # Livecoding tasks
```

### Module Structure:
```
module_name/
├── README.md                      # Theory, key concepts, 30 deep Q&A
├── requirements.txt               # Module-specific dependencies
├── fixtures/                      # Sample data in various formats
│   ├── input/
│   ├── expected/
│   └── edge_cases/
├── learning/                      # Learning notebooks with explanations
│   ├── 01_basics.ipynb
│   └── 02_advanced.ipynb
├── tasks/                         # Livecoding tasks (empty cells + asserts)
│   ├── task_01.ipynb
│   └── task_02.ipynb
└── solutions/                     # Solved versions of tasks (for self-check)
    ├── task_01_solution.ipynb
    └── task_02_solution.ipynb
```

---

## Module 1: Python & Data Fundamentals
> Core skills required for all positions

### Topics:
- Pandas: filtering, grouping, merge, pivot
- JSON parsing and metadata handling
- Regex for data cleaning
- File formats (CSV, JSON, Parquet)

### Documentation:
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Python re Module](https://docs.python.org/3/library/re.html)
- [Regex101 - Interactive Tester](https://regex101.com/)
- [Real Python - Pandas Tutorial](https://realpython.com/pandas-python-explore-dataset/)

### Learning Notebooks:
1. `pandas_essentials.ipynb` - core operations
2. `data_cleaning.ipynb` - regex, text processing
3. `json_metadata.ipynb` - parsing complex structures

### Livecoding Tasks:
- Filter tickets by metadata status (see examples in tasks/)
- Data cleaning: phone numbers, emails, PII removal (critical for fintech!)
- Data aggregation by categories

---

## Module 2: Text Embeddings & Semantic Search
> Core for GenAI and AI Research positions

### Topics:
- Sentence Transformers: architecture, models (gte-base, all-MiniLM, etc.)
- Embeddings: what they are, dimensionality, normalization
- Similarity metrics: cosine similarity, dot product, euclidean distance
- When to use which metric and why

### Documentation:
- [Sentence Transformers Docs](https://www.sbert.net/)
- [SBERT Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
- [HuggingFace MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence Embeddings Paper](https://arxiv.org/abs/1908.10084)
- [Understanding Embeddings - OpenAI](https://platform.openai.com/docs/guides/embeddings)

### Learning Notebooks:
1. `embeddings_intro.ipynb` - what are embeddings, visualization
2. `sentence_transformers.ipynb` - using the library
3. `similarity_metrics.ipynb` - comparing metrics, normalization effects

### Livecoding Tasks:
- Find similar documents (see examples in tasks/)
- Text deduplication using similarity threshold
- Ticket clustering using embeddings

---

## Module 3: Vector Databases
> Practical skill for production systems

### Topics:
- Why vector DBs vs simple numpy arrays
- Main solutions: FAISS, ChromaDB, Pinecone, Weaviate
- Index types: Flat, IVF, HNSW
- Filtering and metadata in vector search

### Documentation:
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)

### Learning Notebooks:
1. `faiss_basics.ipynb` - local vector search
2. `chromadb_intro.ipynb` - with persistence and metadata
3. `scaling_search.ipynb` - indexes and performance

### Livecoding Tasks:
- Build a search system for ticket database
- Add filtering by category/status
- Hybrid search: keyword + semantic

---

## Module 4: Cross-Encoders, NLI & Reranking
> Improving search quality and text understanding

### Topics:
- Bi-encoders vs Cross-encoders: architecture, trade-offs
- `sentence-transformers` CrossEncoder class
- NLI (Natural Language Inference): entailment, contradiction, neutral
- Zero-shot classification with NLI models
- Two-stage retrieval: retrieve + rerank
- Reranking models: `cross-encoder/ms-marco-*`, `cross-encoder/nli-*`
- Semantic textual similarity with cross-encoders

### Learning Notebooks:
1. `cross_encoder_basics.ipynb` - CrossEncoder class, how it differs from bi-encoder
2. `nli_models.ipynb` - NLI task, entailment for classification
3. `zero_shot_classification.ipynb` - classify without training data
4. `two_stage_retrieval.ipynb` - retrieve + rerank pipeline
5. `cross_encoder_training.ipynb` - fine-tuning cross-encoder

### Livecoding Tasks:
- Improve ticket search with reranking
- Zero-shot ticket classification using NLI
- Compare bi-encoder vs cross-encoder quality
- Build entailment-based fact checker

### Documentation:
- [SBERT Cross-Encoders](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [Cross-Encoder Pretrained Models](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
- [NLI Explained - HuggingFace](https://huggingface.co/tasks/text-classification)
- [Zero-Shot Classification](https://huggingface.co/tasks/zero-shot-classification)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)

### Key Models:
| Model | Use Case |
|-------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking search results |
| `cross-encoder/nli-deberta-v3-base` | NLI, zero-shot classification |
| `cross-encoder/stsb-roberta-base` | Semantic similarity scoring |

---

## Module 5: RAG (Retrieval-Augmented Generation)
> Key skill for GenAI Platform position

### Topics:
- RAG architecture: retrieval + generation
- Chunking strategies
- Prompt engineering for RAG
- Evaluation metrics: faithfulness, relevance, groundedness

### Documentation:
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/en/stable/understanding/rag/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [RAGAS Evaluation](https://docs.ragas.io/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

### Learning Notebooks:
1. `rag_basics.ipynb` - simple RAG pipeline
2. `chunking_strategies.ipynb` - how to split documents
3. `rag_evaluation.ipynb` - quality metrics

### Livecoding Tasks:
- RAG system for documentation Q&A
- Q&A bot over ticket database (find solution + generate answer)

---

## Module 6: LLM APIs & Application Development
> Working with OpenAI/Anthropic APIs

### Topics:
- OpenAI API: chat completions, function calling
- Structured outputs with Pydantic
- Error handling, rate limits, retries
- Cost optimization strategies

### Documentation:
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [Instructor Library](https://python.useinstructor.com/)
- [Tenacity (Retries)](https://tenacity.readthedocs.io/)

### Learning Notebooks:
1. `openai_basics.ipynb` - basic usage
2. `function_calling.ipynb` - structured outputs
3. `pydantic_integration.ipynb` - response validation

### Livecoding Tasks:
- Ticket classification using LLM
- Extraction pipeline: extract structured data from text
- Automatic resolution summarization

---

## Module 7: Fine-tuning
> For AI Research position

### Topics:
- Sentence Transformers fine-tuning
- Dataset preparation: pairs, triplets
- Loss functions: Contrastive, Triplet, InfoNCE, MSE
- Evaluation: recall@k, MRR, NDCG

### Documentation:
- [SBERT Training Overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html)
- [SBERT Loss Functions](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html)
- [HuggingFace Trainer](https://huggingface.co/docs/transformers/training)
- [SetFit (Few-shot)](https://huggingface.co/docs/setfit/)
- [Contrastive Learning Paper](https://arxiv.org/abs/2002.05709)

### Learning Notebooks:
1. `dataset_preparation.ipynb` - creating training datasets
2. `losses_explained.ipynb` - different loss functions
3. `finetuning_sbert.ipynb` - practical fine-tuning

### Livecoding Tasks:
- Prepare dataset from tickets for fine-tuning
- Fine-tune model on domain data
- Evaluate quality improvement

---

## Module 8: ML Fundamentals & Metrics
> Theoretical foundation

### Topics:
- Classification metrics: Precision, Recall, F1, Confusion Matrix, ROC-AUC
- Regression metrics: MSE, MAE, R²
- Cross-validation strategies
- Overfitting/underfitting detection

### Documentation:
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [StatQuest YouTube](https://www.youtube.com/c/joshstarmer)

### Learning Notebooks:
1. `classification_metrics.ipynb` - with examples
2. `model_evaluation.ipynb` - proper evaluation practices
3. `cross_validation.ipynb` - practical implementation

### Livecoding Tasks:
- Evaluate a ticket classifier
- Select threshold for binary classification
- A/B testing models

---

## Module 9: MLOps Basics
> For DevOps and Platform positions

### Topics:
- MLflow: tracking experiments, model registry
- Docker basics for ML
- CI/CD for ML pipelines
- AWS basics: S3, Lambda, SageMaker concepts

### Documentation:
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [Docker for Data Science](https://docs.docker.com/)
- [AWS ML Services](https://aws.amazon.com/machine-learning/)
- [DVC (Data Version Control)](https://dvc.org/doc)
- [Weights & Biases](https://docs.wandb.ai/)

### Learning Notebooks:
1. `mlflow_tracking.ipynb` - logging experiments
2. `mlflow_models.ipynb` - saving and loading models
3. `docker_ml.ipynb` - containerization

### Livecoding Tasks:
- Log experiments in MLflow
- Create reproducible pipeline

---

## Module 10: Document Automation
> For Intelligent Automation position

### Topics:
- Document parsing: PDF, DOCX
- OCR basics
- Document classification
- Information extraction

### Documentation:
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [Unstructured.io](https://unstructured-io.github.io/unstructured/)
- [Tesseract OCR](https://tesseract-ocr.github.io/)
- [DocTR](https://mindee.github.io/doctr/)
- [LayoutLM Paper](https://arxiv.org/abs/1912.13318)

### Learning Notebooks:
1. `document_parsing.ipynb` - text extraction
2. `document_classification.ipynb` - type classification
3. `information_extraction.ipynb` - NER, regex, LLM-based

### Livecoding Tasks:
- Classify financial documents
- Extract key fields from document

---

## Fintech-specific Topics (integrate into modules above)

- PII detection and removal (Module 1)
- Compliance and audit trails (Module 9)
- Financial document types (Module 10)
- Risk assessment classification (Module 8)

---

## Recommended Learning Order

```
Week 1-2:   Module 1  (Python/Data) + Module 11 (Data Formats & SQL)
Week 3-4:   Module 2  (Embeddings) + Module 3 (Vector DB) + Module 8 (ML Metrics)
Week 5-6:   Module 4  (Cross-encoders) + Module 5 (RAG)
Week 7-8:   Module 6  (LLM APIs) + Module 15 (Prompt Engineering)
Week 9-10:  Module 7  (Fine-tuning) + Module 16 (LangChain)
Week 11-12: Module 9  (MLOps) + Module 14 (Spark)
Week 13-14: Module 10 (Document Automation) + Module 12 (FastAPI)
Week 15-16: Module 13 (Testing) + System Design Problems (50 scenarios)
```

### Daily Practice:
- Morning: 1-2 learning notebooks
- Afternoon: 1 livecoding task
- Evening: 2-3 deep Q&A from README (explain out loud)

---

## README.md Template for Each Module (with 30 Deep Q&A)

```markdown
# Module Name

## Why This Matters
Practical applications in Fintech/fintech context

## Key Concepts
- Concept 1: explanation
- Concept 2: explanation

## Documentation & Resources

### Official Docs:
- [Library Name - Official Docs](https://...)
- [API Reference](https://...)

### Tutorials & Articles:
- [Tutorial 1](https://...)
- [Deep Dive Article](https://...)

### Papers (optional):
- [Original Paper](https://arxiv.org/...)

## Self-Assessment Checklist
- [ ] I understand X
- [ ] I can explain Y
- [ ] I can implement Z from scratch

---

## Deep Dive Q&A (30 Questions)

### Architecture & Design (10 questions)

#### Q1: You need to build a semantic search system for 10M documents. How would you approach this?

**Answer:**
1. **Architecture Overview:**
   - Use bi-encoder (Sentence Transformers) for embedding generation
   - Store embeddings in vector database (FAISS/Milvus for scale)
   - Implement two-stage retrieval: fast ANN search + cross-encoder reranking

2. **Technology Stack:**
   - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (fast) or `gte-base` (quality)
   - Vector DB: FAISS with IVF index for 10M scale
   - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

3. **Data Flow:**
   ```
   Query → Embed → FAISS (top 100) → Cross-encoder rerank → Top 10
   ```

4. **Considerations:**
   - Index update strategy (batch vs real-time)
   - Caching frequent queries
   - Fallback to keyword search

#### Q2: [Next question...]

### Implementation & Coding (10 questions)

#### Q11: How would you implement efficient batch processing for embeddings?

**Answer:**
...

### Debugging & Troubleshooting (5 questions)

#### Q21: Your semantic search returns irrelevant results. How do you debug?

**Answer:**
...

### Trade-offs & Decisions (5 questions)

#### Q26: When would you choose cross-encoder over bi-encoder?

**Answer:**
...
```

---

## System Design Problems (50 End-to-End Scenarios)

Located in `system_design_problems/` directory. Each problem is a separate markdown file with:

### Problem File Template:
```markdown
# Problem 01: Intelligent Ticket Routing System

## Problem Statement
Design a system that automatically routes incoming support tickets to the
appropriate team based on content, urgency, and historical resolution data.

## Requirements
- Handle 10,000 tickets/day
- Route within 5 seconds
- 95% routing accuracy
- Support for 15 categories

## Questions to Answer
1. What is your high-level architecture?
2. Which ML models would you use?
3. How do you handle edge cases?
4. How do you measure success?
5. How do you handle model updates?

---

## Solution

### Architecture Diagram
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Ticket    │────▶│   API        │────▶│  Classifier │
│   Input     │     │   Gateway    │     │   Service   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐            │
                    │   Vector DB  │◀───────────┤
                    │   (similar   │            │
                    │   tickets)   │            ▼
                    └──────────────┘     ┌─────────────┐
                                         │   Routing   │
                                         │   Engine    │
                                         └─────────────┘
```

### Data Flow
1. Ticket received via API
2. Text preprocessing and embedding generation
3. Classification + similar ticket lookup
4. Confidence-based routing or human escalation
5. Feedback loop for model improvement

### Technology Choices
| Component | Technology | Why |
|-----------|------------|-----|
| Embeddings | gte-base | Good quality for support domain |
| Vector DB | FAISS | Scale, cost, self-hosted |
| Classifier | Fine-tuned BERT | 15 classes, need accuracy |
| API | FastAPI | Async, fast, Pydantic |
| Queue | Redis | Simple, fast routing |

### Trade-offs Discussed
- Real-time vs batch classification
- Accuracy vs latency
- Custom model vs LLM API

### Metrics & Monitoring
- Routing accuracy (weekly review)
- P95 latency
- Escalation rate
- Team satisfaction score
```

### 50 Problem Categories:

**Search & Retrieval (10):**
1. Ticket search system (semantic)
2. Document similarity detection
3. Duplicate detection pipeline
4. Multi-language search
5. Hybrid search (keyword + semantic)
6. Real-time search indexing
7. Personalized search ranking
8. Query understanding system
9. Auto-complete with semantics
10. Cross-document linking

**Classification & NLP (10):**
11. Ticket routing system
12. Sentiment analysis pipeline
13. Intent classification
14. Named entity extraction
15. Topic modeling at scale
16. Spam/fraud detection
17. Priority classification
18. Multi-label classification
19. Language detection
20. Content moderation

**RAG & Generation (10):**
21. Q&A system over documentation
22. Automated response generation
23. Report summarization
24. Chat with documents
25. Knowledge base assistant
26. Code documentation generator
27. Meeting notes summarizer
28. Email response suggester
29. Compliance checker
30. Contract analysis

**Data Pipelines (10):**
31. Document processing pipeline
32. Real-time data enrichment
33. ETL for ML features
34. Data quality monitoring
35. Streaming analytics
36. Batch embedding generation
37. Data versioning system
38. Feature store design
39. Log analysis pipeline
40. Anomaly detection stream

**MLOps & Infrastructure (10):**
41. Model serving architecture
42. A/B testing framework
43. Model monitoring system
44. Retraining pipeline
45. Multi-model orchestration
46. Cost optimization system
47. Caching layer design
48. Rate limiting for ML APIs
49. Disaster recovery for ML
50. Multi-region deployment

---

## Datasets Strategy

### Shared Datasets (`datasets/` directory)

#### Real Datasets from HuggingFace:

| Dataset | Source | Size | Use Case |
|---------|--------|------|----------|
| Support Tickets | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | 27K | Classification, search |
| Financial News | `ashraq/financial-news` | 300K | Sentiment, NER |
| SEC Filings | `JanosAudworx/SEC-10K-Filings` | 10K | Document processing |
| Stack Overflow | `koutch/stackoverflow_python` | 2M | Q&A, search |
| Banking77 | `PolyAI/banking77` | 13K | Intent classification |
| Contracts | `lexlms/legal-contracts-nli` | 10K | Document automation |
| Emails | `aeslc` | 18K | Summarization |
| ArXiv Papers | `ccdv/arxiv-summarization` | 200K | RAG, summarization |

#### download_datasets.py:
```python
from datasets import load_dataset
import os

DATASETS = {
    "tickets": ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None),
    "banking": ("PolyAI/banking77", None),
    "financial_news": ("ashraq/financial-news", "train[:50000]"),
    # ... more datasets
}

def download_all():
    for name, (path, split) in DATASETS.items():
        ds = load_dataset(path, split=split)
        ds.to_parquet(f"datasets/{name}/data.parquet")
        ds.to_json(f"datasets/{name}/data.jsonl")
        # Also save as CSV for SQL loading
        ds.to_csv(f"datasets/{name}/data.csv")
```

#### Pre-computed Resources:
- Embeddings for all datasets (`.npy` files)
- FAISS indexes (ready to load)
- SQL dumps (PostgreSQL, SQLite)
- Sample subsets for quick testing

#### Fixture Sizes:
| Type | Size | Purpose |
|------|------|---------|
| Tiny | 100 rows | Unit tests, quick iteration |
| Small | 1K rows | Learning notebooks |
| Medium | 10K rows | Livecoding tasks |
| Full | 100K+ rows | Performance testing, real scenarios |

---

## Livecoding Task Format

### Task Notebooks (`tasks/`)
1. Markdown cell with task description and requirements
2. Setup code (data loading) - provided
3. Empty cells for solution - student fills in
4. Assert statements for validation - tests correctness
5. Hints in markdown cells (optional)

### Solution Notebooks (`solutions/`)
1. Same structure as task notebooks
2. All cells filled with working solutions
3. Explanations of approach and key decisions
4. Common pitfalls section at the end
5. Named `task_XX_solution.ipynb`

### Workflow:
1. Student attempts `tasks/task_01.ipynb`
2. Runs assert cells to check correctness
3. If stuck, checks `solutions/task_01_solution.ipynb`
4. Reviews solution approach even if passed

---

---

## Module 11: Data Formats, Storage & SQL
> Working with real-world data infrastructure

### Topics:
- File formats: CSV, JSON, JSONL, Parquet, Avro, ORC
- Compression: gzip, snappy, lz4, zstd
- JSON handling: nested structures, streaming JSON
- Parquet: columnar storage, partitioning, predicate pushdown
- SQL fundamentals: joins, aggregations, window functions
- Setting up local databases: PostgreSQL, SQLite
- ORM basics: SQLAlchemy
- Data serialization: pickle, joblib, msgpack

### Documentation:
- [Apache Parquet](https://parquet.apache.org/docs/)
- [PyArrow](https://arrow.apache.org/docs/python/)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)
- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
- [SQLite Docs](https://www.sqlite.org/docs.html)
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)

### Learning Notebooks:
1. `file_formats_comparison.ipynb` - pros/cons of each format
2. `compression_strategies.ipynb` - when to use which
3. `parquet_deep_dive.ipynb` - partitioning, optimization
4. `sql_fundamentals.ipynb` - core SQL operations
5. `local_database_setup.ipynb` - PostgreSQL with Docker
6. `sqlalchemy_orm.ipynb` - Python database interaction

### Livecoding Tasks:
- Convert dataset between formats, optimize storage
- Write SQL queries for ticket analytics
- Design database schema for ticket system
- Query optimization challenge

### Fixtures:
- Same dataset in all formats (CSV, JSON, Parquet, SQL dump)
- Nested JSON with metadata
- Partitioned Parquet dataset
- SQLite database file

---

## Module 12: API Development for ML (FastAPI)
> Building production-ready ML services

### Topics:
- FastAPI basics: endpoints, request/response models
- Async Python for concurrent processing
- Pydantic models for validation
- Error handling patterns
- Streaming responses for LLMs
- Background tasks and workers

### Documentation:
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Starlette (underlying)](https://www.starlette.io/)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)
- [SSE (Server-Sent Events)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [Celery Docs](https://docs.celeryq.dev/)

### Learning Notebooks:
1. `fastapi_basics.ipynb` - building simple API
2. `async_python.ipynb` - async/await patterns
3. `streaming_responses.ipynb` - SSE for LLM output

### Livecoding Tasks:
- Build API endpoint for ticket search
- Add input validation with Pydantic
- Implement streaming chat endpoint

### Fixtures:
- `requests.json` - sample API requests
- `responses.json` - expected responses

---

## Module 13: Testing & Code Quality
> Professional development practices

### Topics:
- pytest for ML code
- Mocking external APIs
- Testing with fixtures
- Code quality: type hints, linting
- Pre-commit hooks

### Documentation:
- [pytest Docs](https://docs.pytest.org/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [Ruff (linter)](https://docs.astral.sh/ruff/)
- [mypy (type checking)](https://mypy.readthedocs.io/)
- [pre-commit](https://pre-commit.com/)

### Learning Notebooks:
1. `pytest_basics.ipynb` - testing fundamentals
2. `testing_ml.ipynb` - testing ML pipelines
3. `mocking_apis.ipynb` - testing with external services

### Livecoding Tasks:
- Write tests for embedding search
- Mock OpenAI API in tests
- Add type hints to existing code

---

## Module 14: Spark Basics
> Big data processing

### Topics:
- PySpark DataFrame basics
- Spark vs Pandas: when to use which
- Reading/writing data formats
- Basic transformations and aggregations

### Documentation:
- [PySpark Docs](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Databricks Learning](https://www.databricks.com/learn)
- [Spark By Examples](https://sparkbyexamples.com/)

### Learning Notebooks:
1. `spark_intro.ipynb` - getting started
2. `spark_vs_pandas.ipynb` - comparison
3. `spark_transformations.ipynb` - common operations

### Livecoding Tasks:
- Process ticket data with Spark (see examples in tasks/)
- Convert Pandas solution to Spark

### Fixtures:
- Same data in formats optimized for Spark

---

## Module 15: Prompt Engineering
> Critical for GenAI positions

### Topics:
- Prompt patterns: few-shot, chain-of-thought, role-based
- System prompts design
- Prompt injection awareness
- Prompt versioning and testing

### Documentation:
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Learn Prompting](https://learnprompting.org/)
- [Prompt Injection - OWASP](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)

### Learning Notebooks:
1. `prompt_patterns.ipynb` - common techniques
2. `few_shot_examples.ipynb` - when and how
3. `prompt_security.ipynb` - injection prevention

### Livecoding Tasks:
- Design prompt for ticket classification
- Improve extraction accuracy with better prompts
- Handle edge cases with prompt engineering

---

## Module 16: LangChain & Orchestration
> Building complex LLM pipelines

### Topics:
- LangChain basics: chains, agents, tools
- LlamaIndex for document processing
- When to use frameworks vs custom code
- Debugging LLM pipelines

### Documentation:
- [LangChain Docs](https://python.langchain.com/docs/)
- [LangChain Expression Language](https://python.langchain.com/docs/concepts/lcel/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [LangSmith (debugging)](https://docs.smith.langchain.com/)
- [Haystack](https://haystack.deepset.ai/overview/intro)

### Learning Notebooks:
1. `langchain_basics.ipynb` - core concepts
2. `building_agents.ipynb` - tool use
3. `custom_vs_framework.ipynb` - trade-offs

### Livecoding Tasks:
- Build RAG pipeline with LangChain
- Create agent with tools for ticket resolution

---

## Additional Suggestions

### Soft Skills / Practice Prep:
- System design for ML systems (how would you design X?)
- Explaining technical concepts simply
- Trade-off discussions (accuracy vs latency, cost vs quality)
- Debugging approach walkthrough

### Fintech-specific Knowledge:
- Basic financial terms (useful for domain understanding)
- Compliance requirements awareness (GDPR, data handling)
- Audit logging patterns
- Data quality and validation

### Topics to Practice During Livecoding:
- Thinking out loud while coding
- Asking clarifying questions
- Testing incrementally with assert statements
- Handling edge cases
- Time management

### Question Types for Practice:
1. **Coding**: Implement X algorithm/pipeline
2. **System Design**: Design a ticket search system
3. **ML Theory**: Explain precision vs recall, when to use which loss
4. **Debugging**: Here's broken code, fix it
5. **Trade-offs**: Compare approaches A vs B

---

## Fixtures Strategy

Each module should have relevant data formats:

| Module | Suggested Fixtures |
|--------|-------------------|
| Python/Data | CSV, JSON, Parquet, nested JSON with metadata |
| Embeddings | Pre-computed embeddings (.npy), text corpus |
| Vector DB | Sample vector index, metadata JSON |
| RAG | PDF documents, chunked text, Q&A pairs |
| LLM APIs | Request/response examples, error cases |
| Fine-tuning | Training pairs, triplets, evaluation set |
| ML Metrics | Predictions CSV, ground truth labels |
| MLOps | MLflow artifacts, model files |
| Document Automation | Sample PDFs, DOCX, scanned images |
| Spark | Large CSV, Parquet partitioned data |

### Fixture Naming Convention:
```
fixtures/
├── input/           # Raw input data
├── expected/        # Expected outputs for validation
├── edge_cases/      # Tricky examples
└── large/           # Performance testing (gitignored)
```

---

## Topics Summary

### Extended Topics:
- Vector databases (critical for production)
- RAG evaluation metrics
- Pydantic for structured outputs
- Document automation
- PII handling (compliance)
- Cloud concepts (AWS)
- Two-stage retrieval (retrieve + rerank)
- API development (FastAPI)
- Testing & code quality
- Spark for big data

### Core Topics:
- ✅ Sentence transformers fine-tuning
- ✅ Vector db for search
- ✅ Cosine similarity, normalization, dot product
- ✅ Cross encoders
- ✅ LLM fine-tuning (included in Module 7)
- ✅ Dataset preparation
- ✅ Regex
- ✅ RAG
- ✅ Different losses
- ✅ Metrics (F1, precision, recall)
- ✅ OpenAI API + Pydantic
- ✅ Custom models pipelines
- ✅ Data analysis
- ✅ MLflow setup
