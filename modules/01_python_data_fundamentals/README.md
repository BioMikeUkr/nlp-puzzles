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

## Deep Dive Q&A (30 Questions)

### Data Loading & Exploration (1-6)

#### Q1: You receive a CSV with 10M rows. How do you explore it efficiently?

**Answer:**
```python
# Sample first
df_sample = pd.read_csv("large.csv", nrows=1000)

# Chunked reading
for chunk in pd.read_csv("large.csv", chunksize=100_000):
    process(chunk)

# Select columns + optimize dtypes
df = pd.read_csv("large.csv",
    usecols=["id", "text"],
    dtype={"category": "category"})
```

---

#### Q2: Dataset has `metadata` column with JSON strings. Extract `status` field?

**Answer:**
```python
import json

# Method 1: apply
df["status"] = df["metadata"].apply(lambda x: json.loads(x).get("status"))

# Method 2: for list of dicts
df["status"] = df["metadata"].apply(
    lambda x: next((d.get("status") for d in json.loads(x) if "status" in d), None)
)
```

---

#### Q3: How handle different encodings when reading CSV?

**Answer:**
```python
import chardet

with open("file.csv", "rb") as f:
    result = chardet.detect(f.read(100000))

df = pd.read_csv("file.csv", encoding=result["encoding"])
```

---

#### Q4: DataFrame has duplicate rows. How identify and handle?

**Answer:**
```python
# Find all duplicates
duplicates = df[df.duplicated(keep=False)]

# Remove - keep first
df_clean = df.drop_duplicates(subset=["email", "date"], keep="first")

# Custom logic - keep row with most data
df_clean = df.sort_values("completeness", ascending=False).drop_duplicates(subset=["id"])
```

---

#### Q5: How read large JSON file with nested structures?

**Answer:**
```python
# JSONL format (one object per line) - best for large files
df = pd.read_json("data.jsonl", lines=True)

# Streaming for huge files
import ijson
with open("huge.json", "rb") as f:
    for record in ijson.items(f, "item"):
        process(record)
```

---

#### Q6: Difference between `loc` and `iloc`?

**Answer:**
```python
# iloc - position-based (integers)
df.iloc[0:5]      # First 5 rows (exclusive end)

# loc - label-based
df.loc[0:5]       # Rows with labels 0-5 (inclusive end!)
df.loc[df["age"] > 30]  # Boolean filtering
```

---

### Filtering & Transformation (7-12)

#### Q7: Filter DataFrame by multiple complex conditions?

**Answer:**
```python
# Boolean indexing
df_filtered = df[
    (df["status"] == "resolved") &
    (df["priority"] > 3) &
    ~df["category"].isin(["spam"])
]

# query() - cleaner
df_filtered = df.query("status == 'resolved' and priority > @min_priority")
```

---

#### Q8: Apply different transformations to different columns?

**Answer:**
```python
# Named aggregations
result = df.groupby("category").agg(
    avg_price=("price", "mean"),
    total=("price", "sum"),
    count=("id", "count")
)

# transform() - keeps original shape
df["group_mean"] = df.groupby("category")["price"].transform("mean")
```

---

#### Q9: Clean text by removing phone numbers and emails?

**Answer:**
```python
import re

def clean_pii(text: str) -> str:
    # Phone: +370 1234567 or 81234567
    text = re.sub(r'\+?\d{1,3}[\s-]?\d{7,}', '[PHONE]', text)
    # Email
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    return text

df["clean"] = df["text"].apply(clean_pii)
```

---

#### Q10: Handle missing values in different scenarios?

**Answer:**
```python
# Drop rows with missing in specific columns
df_clean = df.dropna(subset=["required_field"])

# Fill with group statistics
df["price"] = df.groupby("category")["price"].transform(
    lambda x: x.fillna(x.median())
)

# Flag missing then fill
df["price_missing"] = df["price"].isnull().astype(int)
df["price"] = df["price"].fillna(0)
```

---

#### Q11: Apply function that needs multiple columns?

**Answer:**
```python
# Vectorized (fast)
df["score"] = df["rating"] * df["weight"]

# Conditional
import numpy as np
df["score"] = np.where(df["premium"], df["rating"] * 2, df["rating"])

# Multiple conditions
conditions = [df["tier"] == "gold", df["tier"] == "silver"]
choices = [df["rating"] * 3, df["rating"] * 2]
df["score"] = np.select(conditions, choices, default=df["rating"])
```

---

#### Q12: Extract structured data from text using regex?

**Answer:**
```python
# Named groups
pattern = r"Ticket #(?P<id>TKT-\d+-\d+).*Priority:\s*(?P<priority>\w+)"
df["ticket_id"] = df["text"].str.extract(r"Ticket #(TKT-\d+-\d+)")

# Extract all matches
df["all_dates"] = df["text"].str.findall(r"\d{4}-\d{2}-\d{2}")
```

---

### Aggregation & Grouping (13-18)

#### Q13: Groupby with custom aggregation functions?

**Answer:**
```python
def percentile_90(x):
    return x.quantile(0.9)

result = df.groupby("category").agg(
    p90_time=("resolution_time", percentile_90),
    unique_users=("user_id", "nunique")
)
```

---

#### Q14: Create pivot table with multiple values?

**Answer:**
```python
pivot = pd.pivot_table(
    df,
    values=["revenue", "quantity"],
    index="category",
    columns="quarter",
    aggfunc={"revenue": "sum", "quantity": "mean"},
    fill_value=0,
    margins=True
)
```

---

#### Q15: Calculate running totals and moving averages?

**Answer:**
```python
df = df.sort_values("date")

# Running total
df["cumsum"] = df["amount"].cumsum()

# Moving average
df["ma_7"] = df["amount"].rolling(window=7).mean()

# Per group
df["group_cumsum"] = df.groupby("category")["amount"].cumsum()
```

---

#### Q16: Find top N items per group?

**Answer:**
```python
# Top 3 per category
top_3 = df.sort_values("revenue", ascending=False).groupby("category").head(3)

# With ranking
df["rank"] = df.groupby("category")["revenue"].rank(ascending=False)
top_3 = df[df["rank"] <= 3]
```

---

#### Q17: Compare each row to group statistics?

**Answer:**
```python
df["group_mean"] = df.groupby("category")["price"].transform("mean")
df["group_std"] = df.groupby("category")["price"].transform("std")
df["z_score"] = (df["price"] - df["group_mean"]) / df["group_std"]
df["is_outlier"] = df["z_score"].abs() > 2
```

---

#### Q18: Time-based grouping and resampling?

**Answer:**
```python
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

# Resample
monthly = df.resample("M").agg({"revenue": "sum", "orders": "count"})

# Rolling with time
df["revenue_7d"] = df["revenue"].rolling("7D").sum()
```

---

### Merging & Joining (19-24)

#### Q19: Different types of joins - when use each?

**Answer:**
```python
# INNER - only matching
pd.merge(df1, df2, on="key", how="inner")

# LEFT - keep all from left, add from right
pd.merge(df1, df2, on="key", how="left")

# OUTER - all from both
pd.merge(df1, df2, on="key", how="outer")
```

---

#### Q20: Handle duplicate keys when merging?

**Answer:**
```python
# Check first
print(df1["key"].duplicated().sum())

# Deduplicate before merge
df2_dedup = df2.drop_duplicates(subset="key", keep="first")

# Use validate to catch issues
pd.merge(df1, df2, on="key", validate="one_to_one")
```

---

#### Q21: Merge on multiple columns?

**Answer:**
```python
# Same names
pd.merge(df1, df2, on=["user_id", "date"])

# Different names
pd.merge(df1, df2, left_on=["user", "dt"], right_on=["customer_id", "date"])

# With suffixes
pd.merge(df1, df2, on="id", suffixes=("_new", "_old"))
```

---

#### Q22: Concatenate DataFrames?

**Answer:**
```python
# Vertical (stack rows)
combined = pd.concat([df1, df2, df3], ignore_index=True)

# Horizontal (add columns)
combined = pd.concat([df1, df2], axis=1)
```

---

#### Q23: Find rows in one DataFrame but not another?

**Answer:**
```python
# Using merge indicator
merged = pd.merge(df1, df2, on="key", how="outer", indicator=True)
only_in_df1 = merged[merged["_merge"] == "left_only"]

# Using isin
only_in_df1 = df1[~df1["key"].isin(df2["key"])]
```

---

#### Q24: Efficiently join large DataFrame with small lookup?

**Answer:**
```python
# map() for single column (fastest)
lookup_dict = lookup_df.set_index("code")["name"].to_dict()
df["name"] = df["code"].map(lookup_dict)

# Index-based join
lookup_df = lookup_df.set_index("code")
df = df.join(lookup_df, on="code")
```

---

### Regex Deep Dive (25-30)

#### Q25: Regex for various phone number formats?

**Answer:**
```python
pattern = r'''
    (?:\+\d{1,3}[\s.-]?)?   # International prefix
    (?:\(\d{1,4}\)[\s.-]?)? # Area code
    \d{3}[\s.-]?            # First group
    \d{3,4}[\s.-]?          # Second group
    \d{3,4}                 # Last group
'''
phone_regex = re.compile(pattern, re.VERBOSE)
```

---

#### Q26: Extract all URLs from text?

**Answer:**
```python
url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
urls = re.findall(url_pattern, text)

# Domain only
domain_pattern = r'https?://(?:www\.)?([^/]+)'
domains = re.findall(domain_pattern, text)
```

---

#### Q27: Regex groups and backreferences?

**Answer:**
```python
# Named groups
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
match = re.search(pattern, text)
print(match.groupdict())  # {'year': '2024', ...}

# Backreference - find repeated words
pattern = r'\b(\w+)\s+\1\b'  # "the the" -> matches

# Replace with groups
re.sub(r'(\w+)\s+(\w+)', r'\2, \1', "John Smith")  # "Smith, John"
```

---

#### Q28: Regex performance for large texts?

**Answer:**
```python
# 1. Compile for reuse
pattern = re.compile(r'\b\w+@\w+\.\w+\b')

# 2. Non-capturing groups when don't need value
# (?:https?://) instead of (https?://)

# 3. Be specific, avoid .*
# [^@]+@[^@]+ instead of .*@.*

# 4. Use finditer for memory efficiency
for match in pattern.finditer(huge_text):
    process(match.group())
```

---

#### Q29: Validate email addresses?

**Answer:**
```python
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

def is_valid_email(email):
    return bool(re.match(email_pattern, email))
```

---

#### Q30: Text cleaning and normalization with regex?

**Answer:**
```python
def normalize_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)           # Remove HTML
    text = re.sub(r'https?://\S+', '', text)      # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)           # Remove emails
    text = re.sub(r'[^\w\s.,!?-]', '', text)      # Keep basic chars
    text = re.sub(r'\s+', ' ', text)              # Normalize whitespace
    return text.strip()
```
