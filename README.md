# Curriculum Plan: Practical NLP, ML & GenAI Engineering

## Overview

### Project Structure:
```
nlp-puzzles/
├── README.md                      # Project overview
├── curriculum_plan.md             # This file
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

## Topics Summary

### Module Overview

| Module | Focus Area | Key Technologies | Priority |
|--------|------------|------------------|----------|
| 1 | Python & Data | Pandas, Regex, JSON parsing | 🔴 Critical |
| 2 | Embeddings & Search | sentence-transformers, similarity metrics | 🔴 Critical |
| 3 | Vector Databases | FAISS, ChromaDB, Pinecone | 🟡 Important |
| 4 | Cross-Encoders & NLI | Reranking, zero-shot classification | 🟡 Important |
| 5 | LLM APIs | OpenAI, Anthropic, Pydantic, structured outputs | 🔴 Critical |
| 6 | RAG | Retrieval-augmented generation, chunking | 🔴 Critical |
| 7 | Fine-tuning | SBERT, loss functions, dataset preparation | 🟡 Important |
| 8 | ML Metrics | Precision, Recall, F1, evaluation | 🔴 Critical |
| 9 | MLOps | MLflow, Docker, experiment tracking | 🟡 Important |
| 10 | Document Automation | PDF parsing, OCR, extraction | 🟢 Useful |
| 11 | Data Formats & SQL | Parquet, SQL, database design | 🟡 Important |
| 12 | FastAPI | API development, async, validation | 🟡 Important |
| 13 | Testing & Quality | pytest, type hints, code quality | 🟡 Important |
| 14 | Spark | Big data processing with PySpark | 🟢 Useful |
| 15 | Prompt Engineering | Patterns, optimization, security | 🔴 Critical |
| 16 | LangChain | Orchestration, agents, tools | 🟢 Useful |

### Skills by Category

**Foundation (Must-Know):**
- Data manipulation with pandas (Module 1)
- Text embeddings and semantic similarity (Module 2)
- LLM API usage with structured outputs (Module 5)
- RAG implementation (Module 6)
- ML evaluation metrics (Module 8)
- Prompt engineering patterns (Module 15)

**Production Engineering:**
- Vector databases for scale (Module 3)
- Reranking with cross-encoders (Module 4)
- API development with FastAPI (Module 12)
- MLOps and experiment tracking (Module 9)
- Testing and code quality (Module 13)

**Advanced/Specialized:**
- Model fine-tuning (Module 7)
- Document automation (Module 10)
- Data formats and SQL optimization (Module 11)
- Big data with Spark (Module 14)
- LLM orchestration frameworks (Module 16)

### Learning Path Recommendations

**Path 1: NLP/ML Engineering Focus** (6-8 weeks)
```
Modules 1 → 2 → 5 → 6 → 3 → 8 → 15 → 12
```

**Path 2: Full-Stack ML Engineer** (10-12 weeks)
```
Modules 1 → 2 → 5 → 6 → 3 → 4 → 7 → 8 → 9 → 12 → 13
```

**Path 3: Interview Prep Sprint** (3-4 weeks)
```
Modules 1 → 2 → 5 → 8 → 15
```

---

## How to Use This Curriculum

### Study Workflow for Each Module

```
1. Read README.md
   ├─ Understand theory and key concepts
   ├─ Review documentation links
   └─ Read through 30 Q&A pairs

2. Experiment in Learning Notebooks
   ├─ Open learning/*.ipynb files
   ├─ Run code cells and observe outputs
   ├─ Modify parameters and experiment
   └─ Try variations to build intuition

3. Solve Livecoding Tasks
   ├─ Open tasks/*.ipynb files
   ├─ Read problem description
   ├─ Implement solution in empty cells
   ├─ Run assert statements to validate
   └─ Debug until all tests pass

4. Check Solutions (if stuck)
   ├─ Review solutions/*.ipynb
   ├─ Understand the approach
   ├─ Try to implement again from scratch
   └─ Compare your solution with provided one
```

### Time Estimates per Module

| Activity | Time Required |
|----------|---------------|
| README.md + Q&A | 1-2 hours |
| Learning notebooks | 2-3 hours |
| Livecoding tasks | 2-4 hours |
| **Total per module** | **5-9 hours** |

### Best Practices

**Do:**
- ✅ Type code manually, don't copy-paste
- ✅ Experiment with parameters in learning notebooks
- ✅ Read error messages carefully
- ✅ Try to solve tasks before looking at solutions
- ✅ Re-implement solutions from scratch after understanding them

**Don't:**
- ❌ Skip the Q&A section in README
- ❌ Rush through learning notebooks without experimenting
- ❌ Look at solutions immediately when stuck
- ❌ Skip testing your code with assert statements
- ❌ Ignore documentation links

### When You Get Stuck

1. **Re-read the theory** - Often the answer is in the module README
2. **Check documentation** - Official docs have examples
3. **Review learning notebooks** - Similar patterns might be there
4. **Debug systematically** - Use print statements, check shapes, types
5. **Look at solution** - But then close it and re-implement yourself

### Progress Tracking

Track your progress with this simple system:
- 📖 **Reading** - Currently studying theory
- 🧪 **Experimenting** - Working through learning notebooks
- 💻 **Coding** - Solving livecoding tasks
- ✅ **Completed** - Finished all tasks with passing tests

---

## Module 1: Python & Data Fundamentals
> Foundation for all ML/NLP pipelines - data loading, cleaning, and transformation

### Topics:
- Pandas: loading data, filtering, grouping, merge, transformations
- JSON parsing from DataFrame columns
- Regex for data cleaning and PII removal
- Text normalization pipelines

### Documentation:
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Python re Module](https://docs.python.org/3/library/re.html)
- [Regex101 - Interactive Tester](https://regex101.com/)
- [Real Python - Pandas Tutorial](https://realpython.com/pandas-python-explore-dataset/)

### Learning Notebooks:
1. `01_pandas_essentials.ipynb` - loading CSV/JSON, filtering (boolean, query, isin), JSON column parsing, apply/transform, groupby/agg, merge/map
2. `02_regex_data_cleaning.ipynb` - regex basics, phone/email/URL patterns, PII removal, text normalization, cleaning pipelines

### Livecoding Tasks:
- `task_01_ticket_processing.ipynb` - Complete pipeline: extract status from JSON metadata, filter resolved tickets, remove PII (phones, emails), remove category prefix, aggregate by category

---

## Module 2: Text Embeddings & Semantic Search
> Foundation for semantic search, RAG, and similarity-based applications

### Topics:
- Sentence Transformers: loading models, encoding text
- Embeddings: dense vectors, dimensionality, normalization
- Similarity metrics: cosine similarity, dot product, euclidean distance
- Semantic search implementation

### Documentation:
- [Sentence Transformers Docs](https://www.sbert.net/)
- [SBERT Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
- [HuggingFace MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence Embeddings Paper](https://arxiv.org/abs/1908.10084)
- [Understanding Embeddings - OpenAI](https://platform.openai.com/docs/guides/embeddings)

### Learning Notebooks:
1. `01_embeddings_intro.ipynb` - what are embeddings, SentenceTransformer usage, encoding single/batch, cosine similarity, semantic search function
2. `02_similarity_metrics.ipynb` - cosine vs dot product vs euclidean, normalization effects, when to use which metric, performance comparison
3. `03_clustering_embeddings.ipynb` - K-Means clustering, Elbow/Silhouette for choosing K, hierarchical clustering with dendrograms, DBSCAN for outliers, t-SNE visualization, cluster quality metrics (ARI, NMI)

### Livecoding Tasks:
- `task_01_semantic_search.ipynb` - create normalized embeddings, implement search function with top-k, find near-duplicate documents by threshold, cluster documents with KMeans

---

## Module 3: FAISS for Vector Search
> Production-ready vector search with Facebook AI Similarity Search

### Topics:
- Why FAISS vs simple numpy arrays
- Index types: Flat (exact), IVF (fast), HNSW (balanced)
- Building and saving indexes
- Batch search and filtering
- Performance optimization

### Documentation:
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [FAISS Index Types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)

### Learning Notebooks:
1. `01_faiss_basics.ipynb` - IndexFlatIP, IndexFlatL2, basic search
2. `02_faiss_indexes.ipynb` - IVF, HNSW, index selection
3. `03_faiss_optimization.ipynb` - batch search, memory optimization, GPU acceleration

### Livecoding Tasks:
- Build FAISS index for ticket database
- Compare Flat vs IVF vs HNSW performance
- Implement search with metadata filtering

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

## Module 5: LLM APIs & Application Development
> Working with OpenAI/Anthropic APIs

### Topics:
- OpenAI API: chat completions, function calling
- Structured outputs with Pydantic
- Error handling, rate limits, retries
- Cost optimization strategies
- Streaming responses

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

## Module 6: RAG (Retrieval-Augmented Generation)
> Combining vector search with LLMs for knowledge-grounded responses

### Topics:
- RAG architecture: retrieval + generation
- Building embeddings with sentence-transformers
- Vector database integration with FAISS
- Chunking strategies for documents
- Prompt engineering for RAG
- Context selection and relevance filtering
- Evaluation metrics: faithfulness, relevance, groundedness
- Citation and source tracking

### Documentation:
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers for RAG](https://www.sbert.net/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/en/stable/understanding/rag/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [RAGAS Evaluation](https://docs.ragas.io/)

### Learning Notebooks:
1. `01_rag_basics.ipynb` - RAG pipeline: embeddings (sentence-transformers) + FAISS + LLM
2. `02_chunking_strategies.ipynb` - fixed-size, semantic, recursive chunking
3. `03_rag_with_sources.ipynb` - citation tracking and source references
4. `04_rag_evaluation.ipynb` - quality metrics and testing

### Livecoding Tasks:
- Build end-to-end RAG system: sentence-transformers → FAISS → OpenAI
- Implement chunking pipeline with overlap
- Add source citation to generated answers
- Q&A bot over ticket database with context retrieval

---

## Module 7: Fine-tuning
> Adapting models to specific domains for better performance

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
> Experiment tracking, model versioning, and deployment infrastructure

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
> Processing PDFs, extracting information, and document classification

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
> Maximizing LLM output quality through effective prompting techniques

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
2. **ML Theory**: Explain precision vs recall, when to use which loss
3. **Debugging**: Here's broken code, fix it
4. **Trade-offs**: Compare approaches A vs B
