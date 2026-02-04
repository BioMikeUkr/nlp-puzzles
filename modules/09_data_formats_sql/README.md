# Module 9: Data Formats, Storage & SQL

> Working with real-world data infrastructure - file formats, compression, and databases

## Overview

Understanding data formats and SQL is essential for ML engineers. Most production systems involve moving data between different formats, optimizing storage, and querying databases. This module covers practical skills for working with various data formats and relational databases.

## Learning Objectives

By the end of this module, you will be able to:
- Choose the right file format for different use cases
- Work with Parquet for columnar storage
- Optimize data storage with compression
- Write efficient SQL queries (joins, aggregations, window functions)
- Use SQLAlchemy for database interactions
- Design database schemas for ML applications
- Optimize query performance

## Key Concepts

### 1. File Formats Comparison

| Format | Type | Use Case | Pros | Cons |
|--------|------|----------|------|------|
| **CSV** | Text | Simple tabular data | Human-readable, universal | Large size, slow, no types |
| **JSON** | Text | Nested/hierarchical data | Flexible, human-readable | Large size, slow parsing |
| **JSONL** | Text | Streaming JSON | Line-by-line processing | Not standard JSON |
| **Parquet** | Binary | Analytical queries | Columnar, compressed, fast | Not human-readable |
| **Avro** | Binary | Data exchange | Schema evolution, compact | Less common |
| **ORC** | Binary | Big data (Hive) | Optimized for Spark | Less portable |

### 2. Parquet Deep Dive

**What is Parquet?**
- Columnar storage format
- Optimized for analytical queries
- Built-in compression
- Schema embedded in file
- Supports nested data

**When to use Parquet:**
- ✅ Large datasets (>100MB)
- ✅ Analytical queries (aggregations, filtering)
- ✅ Long-term storage
- ✅ Need compression
- ✅ Reading subset of columns

**When NOT to use Parquet:**
- ❌ Small datasets (<1MB)
- ❌ Need to append rows frequently
- ❌ Need human readability
- ❌ Streaming data (use JSONL)

**Parquet Features:**

```python
import pandas as pd
import pyarrow.parquet as pq

# Write with compression
df.to_parquet('data.parquet',
              compression='snappy',  # or 'gzip', 'zstd'
              index=False)

# Read specific columns (predicate pushdown)
df = pd.read_parquet('data.parquet',
                     columns=['col1', 'col2'])

# Read with filters
df = pd.read_parquet('data.parquet',
                     filters=[('age', '>', 25)])

# Partitioned writes
df.to_parquet('data/',
              partition_cols=['year', 'month'])
```

### 3. Compression

**Compression Algorithms:**

| Algorithm | Speed | Ratio | Use Case |
|-----------|-------|-------|----------|
| **Snappy** | Very Fast | Low (2-3x) | Hot data, frequent access |
| **Gzip** | Slow | High (5-10x) | Cold storage, archival |
| **Zstd** | Fast | High (4-8x) | Best balance |
| **LZ4** | Very Fast | Low (2-3x) | Similar to Snappy |

**Trade-offs:**
- Fast compression = Lower ratio = Larger files = Faster read/write
- Slow compression = Higher ratio = Smaller files = Slower read/write

**Recommendation:**
- **Development**: Snappy (fast iteration)
- **Production**: Zstd (good balance)
- **Archival**: Gzip (maximum compression)

### 4. JSON Processing

**Formats:**

```python
# Standard JSON - entire object in memory
{"name": "Alice", "age": 30}

# JSONL (JSON Lines) - one object per line
{"name": "Alice", "age": 30}
{"name": "Bob", "age": 25}

# Nested JSON
{
  "user": {
    "name": "Alice",
    "address": {"city": "NYC"}
  }
}
```

**Processing:**

```python
import pandas as pd
import jsonlines

# Read JSON
df = pd.read_json('data.json')

# Read JSONL (streaming)
with jsonlines.open('data.jsonl') as reader:
    for obj in reader:
        process(obj)

# Normalize nested JSON
df = pd.json_normalize(data, sep='_')
```

### 5. SQL Fundamentals

**Key Operations:**

**SELECT with filtering:**
```sql
SELECT name, age, salary
FROM employees
WHERE age > 25 AND department = 'Engineering'
ORDER BY salary DESC
LIMIT 10;
```

**JOINs:**
```sql
-- INNER JOIN
SELECT e.name, d.department_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;

-- LEFT JOIN (keep all from left table)
SELECT e.name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;
```

**Aggregations:**
```sql
SELECT department,
       COUNT(*) as employee_count,
       AVG(salary) as avg_salary,
       MAX(salary) as max_salary
FROM employees
GROUP BY department
HAVING COUNT(*) > 5;
```

**Window Functions:**
```sql
-- Rank employees by salary within each department
SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;

-- Running total
SELECT date, amount,
       SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;
```

**Common Table Expressions (CTEs):**
```sql
WITH high_earners AS (
  SELECT * FROM employees WHERE salary > 100000
),
dept_stats AS (
  SELECT department, AVG(salary) as avg_salary
  FROM high_earners
  GROUP BY department
)
SELECT * FROM dept_stats WHERE avg_salary > 120000;
```

### 6. SQLAlchemy

**Core vs ORM:**

```python
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base

# Core - SQL-like
engine = create_engine('postgresql://user:pass@localhost/db')
metadata = MetaData()
users = Table('users', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
)
metadata.create_all(engine)

# ORM - object-oriented
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)

Base.metadata.create_all(engine)
```

**Querying:**

```python
from sqlalchemy.orm import Session

# ORM queries
with Session(engine) as session:
    # Select
    users = session.query(User).filter(User.age > 25).all()

    # Insert
    new_user = User(name='Alice', age=30)
    session.add(new_user)
    session.commit()

    # Update
    user = session.query(User).filter(User.name == 'Alice').first()
    user.age = 31
    session.commit()
```

**Pandas Integration:**

```python
import pandas as pd

# Read from database
df = pd.read_sql('SELECT * FROM users WHERE age > 25', engine)

# Write to database
df.to_sql('users', engine, if_exists='append', index=False)
```

### 7. Database Design for ML

**Schema Design Patterns:**

**1. Training Data Storage:**
```sql
CREATE TABLE training_data (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    features JSONB,  -- flexible feature storage
    label VARCHAR(50),
    dataset_version VARCHAR(20),
    split VARCHAR(10)  -- train/val/test
);

CREATE INDEX idx_dataset_split ON training_data(dataset_version, split);
```

**2. Predictions Log:**
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    input_data JSONB,
    prediction VARCHAR(100),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_model_time ON predictions(model_version, created_at);
```

**3. Feature Store:**
```sql
CREATE TABLE features (
    entity_id VARCHAR(100),
    feature_name VARCHAR(100),
    feature_value FLOAT,
    timestamp TIMESTAMP,
    PRIMARY KEY (entity_id, feature_name, timestamp)
);
```

### 8. Query Optimization

**Best Practices:**

1. **Use indexes:**
```sql
-- Add index on frequently filtered columns
CREATE INDEX idx_user_email ON users(email);

-- Composite index for multiple columns
CREATE INDEX idx_user_dept_age ON users(department, age);
```

2. **Limit data scanned:**
```sql
-- Bad: scanning entire table
SELECT * FROM large_table;

-- Good: only needed columns
SELECT id, name FROM large_table WHERE created_at > '2024-01-01';
```

3. **Use EXPLAIN to analyze:**
```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25;
```

4. **Avoid N+1 queries:**
```python
# Bad: N+1 queries
for user in users:
    department = session.query(Department).filter_by(id=user.dept_id).first()

# Good: single query with join
users_with_dept = session.query(User, Department)\
    .join(Department)\
    .all()
```

## Common Use Cases

### 1. ETL Pipeline
```
CSV → Pandas → Clean → Parquet → Database
```

### 2. Feature Engineering
```
Database → SQL query → Pandas → Feature engineering → Parquet
```

### 3. Model Training
```
Parquet → PyTorch DataLoader → Training
```

### 4. Prediction Serving
```
API request → Database query → Model prediction → Log to database
```

## Performance Tips

### 1. Batch Processing
```python
# Bad: row-by-row
for row in df.iterrows():
    insert_to_db(row)

# Good: batch insert
df.to_sql('table', engine, if_exists='append', chunksize=1000)
```

### 2. Lazy Loading
```python
# Use dask for larger-than-memory data
import dask.dataframe as dd

ddf = dd.read_parquet('large_data/')
result = ddf[ddf['age'] > 25].compute()
```

### 3. Connection Pooling
```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://...',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

## Documentation Links

- [Apache Parquet](https://parquet.apache.org/docs/)
- [PyArrow](https://arrow.apache.org/docs/python/)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)
- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
- [DuckDB](https://duckdb.org/docs/)
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)

## Next Steps

After completing this module:
1. Module 10: FastAPI - build APIs with database backends
2. Module 11: Testing & Code Quality - test database interactions
3. Module 12: Spark - big data processing
4. Module 8: ML Metrics - store evaluation results in database
