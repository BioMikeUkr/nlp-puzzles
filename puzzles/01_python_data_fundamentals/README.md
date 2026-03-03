# Module 1: Python & Data Fundamentals

> Foundation for all ML/NLP pipelines - data loading, cleaning, and transformation

## Why This Matters

Every ML pipeline starts with data. Before embeddings, models, or RAG - you need to load, clean, filter, and transform data. These skills are tested in every technical assessment.

## Key Concepts

### Pandas Core Operations
- `DataFrame.query()` - SQL-like filtering
- `DataFrame.apply()` - row/column transformations
- `groupby()` + `agg()` - aggregations
- `merge()` / `join()` - combining datasets

### JSON Handling
- `json.loads()` / `json.dumps()`
- Nested structure navigation
- Handling missing keys with `.get()`

### Regex Patterns
- `\d{7}` - digit sequences
- `[a-zA-Z]+` - word characters
- `|` - alternatives
- `()` - capture groups

## Documentation & Resources

- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Python re Module](https://docs.python.org/3/library/re.html)
- [Regex101](https://regex101.com/)
- [Real Python - Pandas](https://realpython.com/pandas-python-explore-dataset/)

## Self-Assessment Checklist

- [ ] Filter DataFrame by multiple conditions
- [ ] Parse nested JSON and extract values
- [ ] Write regex for phone/email patterns
- [ ] Use groupby with multiple aggregations
- [ ] Merge DataFrames on different keys

---

## Practice Questions

See [QUESTIONS.md](./QUESTIONS.md) for 30 deep-dive questions with detailed answers covering:
- Data Loading & Exploration (Q1-Q6)
- Filtering & Transformation (Q7-Q12)
- Aggregation & Grouping (Q13-Q18)
- Merging & Joining (Q19-Q24)
- Regex Deep Dive (Q25-Q30)
