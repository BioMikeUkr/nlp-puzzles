# NLP Puzzles
![alt text](static/logo.png)
A hands-on NLP/ML curriculum for **trainees and juniors** who already know basic Python, ML and want to get production-ready in AI/ML engineering from embeddings to production APIs.

---

## How to Study

Each puzzle follows the same flow:

**1. Read `README.md`** — start with theory. Understand the concepts before touching any code.

**2. Work through `learning/` notebooks** — run cells, tweak parameters, break things. The goal is intuition, not just output.

**3. Solve `tasks/` notebooks** — cells are empty, you fill them in. Each task has `assert` statements at the bottom — you're done when they all pass. Don't look at solutions until you've genuinely tried.

**4. Do the Q&A** — open `QUESTIONS.md`, try to answer each question out loud or in writing before reading the answer. These are the questions you'll get in interviews.

> Solutions are in `solutions/` — use them to compare approaches, not as a shortcut.

---

## Setup

```bash
# Clone and create venv
git clone https://github.com/BioMikeUkr/nlp-puzzles.git
cd nlp-puzzles
python -m venv .venv && source .venv/bin/activate

# Install dependencies for a specific puzzle
pip install -r puzzles/01_python_data_fundamentals/requirements.txt

# Launch Jupyter
jupyter notebook puzzles/01_python_data_fundamentals/learning/
```

---

## Puzzles

| # | Topic | Key Tools |
|---|-------|-----------|
| | **Foundation** | |
| 01 | Python & Data Fundamentals | pandas, numpy, regex |
| 02 | ML Metrics | scikit-learn, matplotlib, seaborn |
| | **Core NLP** | |
| 03 | Text Embeddings & Semantic Search | sentence-transformers, scikit-learn |
| 04 | FAISS for Vector Search | faiss-cpu, sentence-transformers |
| 05 | Cross-Encoders & NLI | sentence-transformers, transformers |
| | **LLM Stack** | |
| 06 | LLM APIs | openai, pydantic, tenacity |
| 07 | Prompt Engineering | openai, pydantic |
| 08 | RAG | openai, sentence-transformers, faiss-cpu |
| 09 | LangChain & Orchestration | langchain, langchain-openai, faiss-cpu |
| | **ML Engineering** | |
| 10 | Fine-tuning Sentence Transformers | sentence-transformers, datasets |
| 11 | Data Formats & SQL | pyarrow, sqlalchemy, pandas |
| 12 | FastAPI for ML | fastapi, uvicorn, pydantic |
| 13 | Testing & Code Quality | pytest, pytest-mock, ruff, mypy |
| 14 | Spark Basics | pyspark, pyarrow |
| | **Specialized** | |
| 15 | NER with GLiNER | gliner, transformers |
| 16 | Text Classification with GLiClass | gliclass, transformers |
| 17 | Dataset Generation for NER & Classification | openai, gliner, gliclass |
| 18 | Gradio Demos | gradio, gliner, gliclass |

---

## Puzzle Structure

```
puzzles/XX_name/
├── README.md        # theory and key concepts
├── QUESTIONS.md     # 30 deep Q&A
├── requirements.txt
├── fixtures/        # sample input/output data
├── learning/        # tutorial notebooks
├── tasks/           # livecoding tasks (you solve these)
└── solutions/       # reference solutions (check after solving)
```
