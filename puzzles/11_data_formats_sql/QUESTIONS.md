# Module 9: Data Formats & SQL - Deep Dive Questions

## Architecture & Design (Q1-Q10)

### Q1: You have 10GB of CSV transaction data that you query daily for ML feature extraction. Reading takes 5 minutes. Your manager asks you to optimize this. What format would you choose and why? Show the conversion and performance comparison.

**Answer:**

Convert to **Parquet with Snappy compression**. Parquet is columnar, allowing you to read only needed columns, and Snappy provides fast decompression.

**Conversion and comparison:**

```python
import pandas as pd
import time

# Read CSV
start = time.time()
df = pd.read_csv('transactions.csv')
csv_time = time.time() - start
csv_size = os.path.getsize('transactions.csv') / (1024**3)  # GB

print(f"CSV: {csv_time:.2f}s, {csv_size:.2f}GB")

# Convert to Parquet with different compressions
for compression in ['snappy', 'gzip', 'zstd']:
    output_file = f'transactions_{compression}.parquet'

    start = time.time()
    df.to_parquet(output_file, compression=compression, index=False)
    write_time = time.time() - start

    start = time.time()
    df_read = pd.read_parquet(output_file)
    read_time = time.time() - start

    file_size = os.path.getsize(output_file) / (1024**3)

    print(f"{compression}: write={write_time:.2f}s, read={read_time:.2f}s, size={file_size:.2f}GB")

# Column subset read (Parquet advantage)
start = time.time()
df_subset = pd.read_parquet('transactions_snappy.parquet',
                             columns=['user_id', 'amount', 'timestamp'])
subset_time = time.time() - start
print(f"Parquet column subset: {subset_time:.2f}s")
```

**Expected results:**
```
CSV: 300s, 10GB
snappy: write=45s, read=30s, size=2.5GB
gzip: write=180s, read=90s, size=1.2GB
zstd: write=60s, read=40s, size=1.5GB
Parquet column subset: 5s
```

**Recommendation:** Use Snappy for daily queries (10x faster reads). If storage cost is critical, use Zstd.

---

### Q2: Design a database schema for storing embeddings and metadata for 1M documents in a RAG system. Include tables for documents, embeddings, and query logs. Explain your indexing strategy.

**Answer:**

**Schema design:**

```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    title VARCHAR(500),
    source VARCHAR(200),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Embeddings table (separate for flexibility)
CREATE TABLE embeddings (
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL,
    embedding FLOAT8[] NOT NULL,  -- or use pgvector extension
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (document_id, model_version)
);

-- Query logs for monitoring and evaluation
CREATE TABLE query_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    retrieved_doc_ids INTEGER[],
    response_text TEXT,
    user_id VARCHAR(100),
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_embeddings_model ON embeddings(model_version);
CREATE INDEX idx_query_logs_user_time ON query_logs(user_id, created_at DESC);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);  -- for JSONB queries
```

**If using pgvector extension:**

```sql
CREATE EXTENSION vector;

CREATE TABLE embeddings (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id),
    model_version VARCHAR(50),
    embedding vector(1536),  -- for OpenAI embeddings
    created_at TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX embeddings_hnsw_idx ON embeddings
    USING hnsw (embedding vector_cosine_ops);
```

**Why this design:**
- Separate embeddings table allows multiple model versions per document
- JSONB metadata for flexible schema
- Query logs for A/B testing and evaluation
- CASCADE delete ensures no orphaned embeddings

---

### Q3: You need to process 100GB of JSONL logs daily. Each line is a user event. You need to extract events from the past week for a specific user_id. Compare: (1) Loading all data into pandas, (2) DuckDB, (3) Parquet partitioning. Which is best and why?

**Answer:**

**Option 1: Pandas (BAD)**
```python
import pandas as pd

# This will fail or be very slow
df = pd.read_json('events.jsonl', lines=True)  # Out of memory!
user_events = df[df['user_id'] == 'user123']
```
❌ Loads entire 100GB into memory → OOM crash

**Option 2: DuckDB (GOOD)**
```python
import duckdb

con = duckdb.connect()
result = con.execute("""
    SELECT *
    FROM read_json_auto('events.jsonl')
    WHERE user_id = 'user123'
      AND timestamp > current_date - interval '7 days'
""").df()
```
✅ Streaming, only scans needed rows, ~10x faster than pandas

**Option 3: Parquet Partitioning (BEST)**
```python
# One-time conversion with partitioning
df = pd.read_json('events.jsonl', lines=True, chunksize=100000)
for chunk in df:
    chunk['date'] = pd.to_datetime(chunk['timestamp']).dt.date
    chunk.to_parquet('events_parquet/',
                     partition_cols=['date'],
                     compression='snappy',
                     engine='pyarrow')

# Query specific partition (only reads relevant files)
df = pd.read_parquet('events_parquet/',
                     filters=[('date', '>=', '2024-01-27'),
                             ('user_id', '==', 'user123')])
```
✅ Best performance after initial conversion
✅ Predicate pushdown skips irrelevant files

**Recommendation:**
- **One-time query:** DuckDB (no preprocessing needed)
- **Repeated queries:** Parquet partitioning (100x faster for filtered queries)

---

### Q4: When would you choose JSONB in PostgreSQL over a normalized relational schema? Give a concrete example with trade-offs.

**Answer:**

**Use JSONB when:**
1. Schema is highly variable or evolving
2. Need flexible metadata storage
3. Querying is secondary to storage

**Example: User profiles in a multi-tenant SaaS**

**Option A: Normalized (traditional)**
```sql
CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));
CREATE TABLE user_attributes (
    user_id INTEGER REFERENCES users(id),
    attribute_name VARCHAR(50),
    attribute_value TEXT,
    PRIMARY KEY (user_id, attribute_name)
);
```

**Option B: JSONB**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    attributes JSONB
);

CREATE INDEX idx_users_attributes ON users USING GIN(attributes);
```

**Usage comparison:**

```python
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://...')

# Insert with JSONB
engine.execute(text("""
    INSERT INTO users (name, attributes)
    VALUES (:name, :attrs::jsonb)
"""), name='Alice', attrs='{"age": 30, "city": "NYC", "premium": true}')

# Query JSONB
result = engine.execute(text("""
    SELECT * FROM users
    WHERE attributes->>'city' = 'NYC'
      AND (attributes->>'age')::int > 25
"""))

# Update nested field
engine.execute(text("""
    UPDATE users
    SET attributes = jsonb_set(attributes, '{premium}', 'false'::jsonb)
    WHERE id = 1
"""))
```

**Trade-offs:**

| Aspect | Normalized | JSONB |
|--------|-----------|-------|
| Schema enforcement | ✅ Strong | ❌ Weak |
| Query performance | ✅ Fast with indexes | ⚠️ Slower, requires GIN index |
| Flexibility | ❌ Requires migrations | ✅ No schema changes |
| Storage | ✅ Efficient | ⚠️ Slightly larger |
| Joins | ✅ Easy | ❌ Harder |

**Recommendation:**
- Core data (user_id, email): Normalized
- Variable metadata (preferences, tags): JSONB

---

### Q5: You're building a feature store. Should you store features in wide format (one row per entity, many columns) or long format (one row per feature)? Explain with examples and query patterns.

**Answer:**

**Wide Format (Recommended for ML):**
```sql
CREATE TABLE features_wide (
    entity_id VARCHAR(100) PRIMARY KEY,
    feature_1 FLOAT,
    feature_2 FLOAT,
    feature_3 FLOAT,
    -- ... 100+ features
    updated_at TIMESTAMP
);

-- Query for training data (fast)
SELECT feature_1, feature_2, feature_3, label
FROM features_wide f
JOIN labels l ON f.entity_id = l.entity_id
WHERE updated_at > '2024-01-01';
```

**Long Format (Flexible but slower):**
```sql
CREATE TABLE features_long (
    entity_id VARCHAR(100),
    feature_name VARCHAR(100),
    feature_value FLOAT,
    updated_at TIMESTAMP,
    PRIMARY KEY (entity_id, feature_name)
);

-- Query for training data (slow, requires pivot)
SELECT entity_id,
       MAX(CASE WHEN feature_name = 'feature_1' THEN feature_value END) as feature_1,
       MAX(CASE WHEN feature_name = 'feature_2' THEN feature_value END) as feature_2,
       MAX(CASE WHEN feature_name = 'feature_3' THEN feature_value END) as feature_3
FROM features_long
GROUP BY entity_id;
```

**Hybrid Approach (Best):**
```sql
-- Wide table for serving (fast)
CREATE TABLE feature_vectors (
    entity_id VARCHAR(100) PRIMARY KEY,
    features FLOAT8[],  -- array of feature values
    feature_names TEXT[], -- parallel array of names
    updated_at TIMESTAMP
);

-- Long table for analytics/monitoring
CREATE TABLE feature_history (
    entity_id VARCHAR(100),
    feature_name VARCHAR(100),
    feature_value FLOAT,
    timestamp TIMESTAMP
);
```

**Recommendation:**
- **Serving (production):** Wide format or array
- **Analytics/debugging:** Long format
- **Best practice:** Maintain both, sync periodically

---

### Q6: You need to store time-series embeddings for anomaly detection (1M points per day, 384-dim vectors). Design a schema that supports: (1) Querying by time range, (2) Finding similar patterns, (3) Efficient storage. What tables, indexes, and partitioning strategy would you use?

**Answer:**

**Schema design with time-series optimization:**

```sql
-- Main time-series data table (partitioned by date)
CREATE TABLE embedding_timeseries (
    id BIGSERIAL,
    timestamp TIMESTAMP NOT NULL,
    sensor_id VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    metadata JSONB,
    embedding vector(384),  -- Using pgvector
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE embedding_timeseries_2024_01 PARTITION OF embedding_timeseries
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE embedding_timeseries_2024_02 PARTITION OF embedding_timeseries
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Indexes for different query patterns
CREATE INDEX idx_ts_sensor_time ON embedding_timeseries(sensor_id, timestamp DESC);
CREATE INDEX idx_ts_time ON embedding_timeseries(timestamp DESC);

-- HNSW index for vector similarity (per partition for better performance)
CREATE INDEX embedding_timeseries_2024_01_vec_idx
    ON embedding_timeseries_2024_01
    USING hnsw (embedding vector_cosine_ops);

-- Summary table for aggregated statistics
CREATE TABLE embedding_summaries (
    date DATE PRIMARY KEY,
    sensor_id VARCHAR(50),
    count INTEGER,
    avg_value FLOAT,
    anomaly_count INTEGER,
    centroid_embedding vector(384)
);

CREATE INDEX idx_summaries_sensor ON embedding_summaries(sensor_id, date DESC);
```

**Efficient storage with TimescaleDB:**

```sql
-- Alternative: Use TimescaleDB for automatic partitioning
CREATE EXTENSION timescaledb;

CREATE TABLE embedding_timeseries (
    timestamp TIMESTAMP NOT NULL,
    sensor_id VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    embedding vector(384)
);

-- Convert to hypertable (automatic time partitioning)
SELECT create_hypertable('embedding_timeseries', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

-- Compression policy (reduces storage by 10-20x)
ALTER TABLE embedding_timeseries SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'sensor_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('embedding_timeseries', INTERVAL '7 days');

-- Retention policy (auto-delete old data)
SELECT add_retention_policy('embedding_timeseries', INTERVAL '90 days');
```

**Query examples:**

```python
import psycopg2
from datetime import datetime, timedelta

conn = psycopg2.connect("postgresql://...")
cur = conn.cursor()

# Q1: Time range query (uses partition pruning)
cur.execute("""
    SELECT timestamp, sensor_id, value, embedding
    FROM embedding_timeseries
    WHERE timestamp >= %s AND timestamp < %s
      AND sensor_id = %s
    ORDER BY timestamp DESC
""", (datetime.now() - timedelta(days=1), datetime.now(), 'sensor_123'))

# Q2: Find similar patterns (vector search)
cur.execute("""
    SELECT timestamp, sensor_id,
           1 - (embedding <=> %s::vector) as similarity
    FROM embedding_timeseries
    WHERE timestamp >= %s
    ORDER BY embedding <=> %s::vector
    LIMIT 50
""", (query_embedding, datetime.now() - timedelta(days=7), query_embedding))

# Q3: Anomaly detection with window function
cur.execute("""
    SELECT timestamp, sensor_id, value,
           AVG(value) OVER (PARTITION BY sensor_id
                           ORDER BY timestamp
                           ROWS BETWEEN 100 PRECEDING AND CURRENT ROW) as moving_avg,
           STDDEV(value) OVER (PARTITION BY sensor_id
                              ORDER BY timestamp
                              ROWS BETWEEN 100 PRECEDING AND CURRENT ROW) as moving_std
    FROM embedding_timeseries
    WHERE timestamp >= %s
""", (datetime.now() - timedelta(hours=24),))
```

**Storage optimization:**

```python
# Daily aggregation job to reduce storage
def aggregate_old_data():
    cur.execute("""
        INSERT INTO embedding_summaries
        SELECT
            date_trunc('day', timestamp) as date,
            sensor_id,
            COUNT(*) as count,
            AVG(value) as avg_value,
            SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
            AVG(embedding) as centroid_embedding  -- Average embedding
        FROM embedding_timeseries
        WHERE timestamp < current_date - interval '30 days'
        GROUP BY date, sensor_id
        ON CONFLICT (date, sensor_id) DO UPDATE
        SET count = EXCLUDED.count,
            avg_value = EXCLUDED.avg_value
    """)

    # Delete raw data older than 30 days
    cur.execute("""
        DELETE FROM embedding_timeseries
        WHERE timestamp < current_date - interval '30 days'
    """)
    conn.commit()
```

**Why this design:**
- Partitioning by time enables fast range queries (partition pruning)
- Vector indexes per partition keep search fast
- Compression for old data (10-20x savings)
- Summary tables for long-term analytics
- Automatic retention policies prevent unbounded growth

---

### Q7: Compare row-oriented (CSV, PostgreSQL) vs columnar (Parquet, ClickHouse) storage for a 10TB dataset with 100 columns where you typically query 5 columns at a time. Show the performance difference with code.

**Answer:**

**Row-oriented storage:**
```
Row 1: [col1, col2, col3, ..., col100]
Row 2: [col1, col2, col3, ..., col100]
```
Reading any column requires reading entire row → wasteful

**Columnar storage:**
```
Col1: [row1_val, row2_val, row3_val, ...]
Col2: [row1_val, row2_val, row3_val, ...]
```
Reading specific columns doesn't touch others → efficient

**Performance comparison:**

```python
import pandas as pd
import numpy as np
import time
import psycopg2
from sqlalchemy import create_engine

# Generate test data (10M rows, 100 columns)
np.random.seed(42)
df = pd.DataFrame({
    f'col_{i}': np.random.randn(10_000_000) for i in range(100)
})
df['id'] = range(len(df))

print(f"Dataset: {len(df):,} rows, {len(df.columns)} columns")
print(f"Memory: {df.memory_usage().sum() / (1024**3):.2f} GB")

# 1. CSV (row-oriented)
print("\n=== CSV (Row-oriented) ===")
df.to_csv('data.csv', index=False)

start = time.time()
df_csv = pd.read_csv('data.csv', usecols=['col_0', 'col_1', 'col_2', 'col_3', 'col_4'])
csv_time = time.time() - start
csv_size = os.path.getsize('data.csv') / (1024**3)

print(f"Read 5 columns: {csv_time:.2f}s")
print(f"File size: {csv_size:.2f} GB")

# 2. Parquet (columnar)
print("\n=== Parquet (Columnar) ===")
df.to_parquet('data.parquet', compression='snappy', index=False)

start = time.time()
df_parquet = pd.read_parquet('data.parquet',
                              columns=['col_0', 'col_1', 'col_2', 'col_3', 'col_4'])
parquet_time = time.time() - start
parquet_size = os.path.getsize('data.parquet') / (1024**3)

print(f"Read 5 columns: {parquet_time:.2f}s")
print(f"File size: {parquet_size:.2f} GB")
print(f"Speedup vs CSV: {csv_time/parquet_time:.2f}x")

# 3. PostgreSQL (row-oriented database)
print("\n=== PostgreSQL (Row-oriented DB) ===")
engine = create_engine('postgresql://user:pass@localhost/testdb')

# Create table and insert data (in chunks)
df.to_sql('data_table', engine, if_exists='replace', index=False, chunksize=10000)

start = time.time()
df_pg = pd.read_sql(
    "SELECT col_0, col_1, col_2, col_3, col_4 FROM data_table",
    engine
)
pg_time = time.time() - start

print(f"Read 5 columns: {pg_time:.2f}s")

# 4. ClickHouse (columnar database)
print("\n=== ClickHouse (Columnar DB) ===")
from clickhouse_driver import Client

client = Client('localhost')
client.execute('CREATE DATABASE IF NOT EXISTS testdb')
client.execute('''
    CREATE TABLE IF NOT EXISTS testdb.data_table (
        id UInt64,
        col_0 Float64,
        col_1 Float64,
        col_2 Float64,
        col_3 Float64,
        col_4 Float64
        -- ... all 100 columns
    ) ENGINE = MergeTree()
    ORDER BY id
''')

# Insert data
client.execute('INSERT INTO testdb.data_table VALUES', df.values.tolist())

start = time.time()
result = client.execute('SELECT col_0, col_1, col_2, col_3, col_4 FROM testdb.data_table')
ch_time = time.time() - start

print(f"Read 5 columns: {ch_time:.2f}s")
print(f"Speedup vs PostgreSQL: {pg_time/ch_time:.2f}x")
```

**Expected results:**

```
Dataset: 10,000,000 rows, 101 columns
Memory: 7.63 GB

=== CSV (Row-oriented) ===
Read 5 columns: 45.3s
File size: 8.2 GB

=== Parquet (Columnar) ===
Read 5 columns: 1.2s
File size: 2.1 GB
Speedup vs CSV: 37.75x

=== PostgreSQL (Row-oriented DB) ===
Read 5 columns: 35.8s

=== ClickHouse (Columnar DB) ===
Read 5 columns: 0.8s
Speedup vs PostgreSQL: 44.75x
```

**Why columnar is faster:**

```python
# Simulate disk I/O for reading 5 columns

# Row-oriented: Must read ALL 100 columns
row_oriented_io = 10_000_000 * 100 * 8  # rows * cols * 8 bytes
print(f"Row-oriented I/O: {row_oriented_io / (1024**3):.2f} GB")

# Columnar: Only read 5 columns
columnar_io = 10_000_000 * 5 * 8  # rows * selected_cols * 8 bytes
print(f"Columnar I/O: {columnar_io / (1024**3):.2f} GB")

print(f"I/O reduction: {row_oriented_io / columnar_io:.0f}x")
```

**Output:**
```
Row-oriented I/O: 7.45 GB
Columnar I/O: 0.37 GB
I/O reduction: 20x
```

**Recommendation:**
- **Analytics queries (few columns):** Columnar (Parquet, ClickHouse)
- **Transactional updates:** Row-oriented (PostgreSQL, MySQL)
- **For 10TB + analytical workload:** Use columnar (ClickHouse or data lake with Parquet)

---

### Q8: You have nested JSON data from an API (max depth 5). Compare storing it as: (1) Flattened table, (2) JSONB column, (3) Multiple normalized tables. Show when each is appropriate.

**Answer:**

**Sample data:**
```json
{
  "user_id": "user_123",
  "name": "Alice",
  "address": {
    "street": "123 Main St",
    "city": "NYC",
    "coordinates": {"lat": 40.7, "lon": -74.0}
  },
  "orders": [
    {"order_id": "ord_1", "items": [{"sku": "A1", "qty": 2}]},
    {"order_id": "ord_2", "items": [{"sku": "B2", "qty": 1}]}
  ]
}
```

**Option 1: Flattened table**

```sql
CREATE TABLE users_flat (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    address_street VARCHAR(200),
    address_city VARCHAR(100),
    address_lat FLOAT,
    address_lon FLOAT,
    orders_json TEXT  -- Store orders as JSON string
);
```

```python
import pandas as pd
from pandas.json_normalize import json_normalize

# Flatten nested JSON
flat_data = json_normalize(
    data,
    max_level=2,
    sep='_',
    errors='ignore'
)

flat_data.to_sql('users_flat', engine, if_exists='replace', index=False)

# Query
df = pd.read_sql("SELECT * FROM users_flat WHERE address_city = 'NYC'", engine)
```

**Pros:**
- ✅ Fast queries (indexed columns)
- ✅ Easy to understand
- ✅ Works with any SQL database

**Cons:**
- ❌ Schema changes require migrations
- ❌ Can't handle variable structure
- ❌ Many NULL columns if sparse data

---

**Option 2: JSONB column**

```sql
CREATE TABLE users_jsonb (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    data JSONB  -- Store everything else as JSONB
);

-- Index for fast JSON queries
CREATE INDEX idx_users_jsonb_city ON users_jsonb ((data->'address'->>'city'));
CREATE INDEX idx_users_jsonb_gin ON users_jsonb USING GIN(data);
```

```python
import json
import psycopg2

conn = psycopg2.connect("postgresql://...")
cur = conn.cursor()

# Insert
cur.execute("""
    INSERT INTO users_jsonb (user_id, name, data)
    VALUES (%s, %s, %s::jsonb)
""", ('user_123', 'Alice', json.dumps(user_data)))

# Query nested fields
cur.execute("""
    SELECT user_id, name, data->'address'->>'city' as city
    FROM users_jsonb
    WHERE data->'address'->>'city' = 'NYC'
""")

# Query array elements
cur.execute("""
    SELECT user_id, jsonb_array_elements(data->'orders')->>'order_id' as order_id
    FROM users_jsonb
    WHERE user_id = 'user_123'
""")

# Update nested field
cur.execute("""
    UPDATE users_jsonb
    SET data = jsonb_set(data, '{address,city}', '"LA"')
    WHERE user_id = 'user_123'
""")
```

**Pros:**
- ✅ Flexible schema (no migrations)
- ✅ Preserves nested structure
- ✅ PostgreSQL JSONB is fast with GIN indexes

**Cons:**
- ❌ Slower than indexed columns
- ❌ Harder to enforce constraints
- ❌ Complex queries are verbose

---

**Option 3: Normalized tables**

```sql
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE addresses (
    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
    street VARCHAR(200),
    city VARCHAR(100),
    lat FLOAT,
    lon FLOAT
);

CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE order_items (
    order_id VARCHAR(50) REFERENCES orders(order_id),
    sku VARCHAR(50),
    quantity INTEGER,
    PRIMARY KEY (order_id, sku)
);

-- Indexes
CREATE INDEX idx_addresses_city ON addresses(city);
CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_order_items_order ON order_items(order_id);
```

```python
# Insert with relationships
cur.execute("INSERT INTO users VALUES (%s, %s)", ('user_123', 'Alice'))
cur.execute("""
    INSERT INTO addresses VALUES (%s, %s, %s, %s, %s)
""", ('user_123', '123 Main St', 'NYC', 40.7, -74.0))

cur.execute("INSERT INTO orders VALUES (%s, %s)", ('ord_1', 'user_123'))
cur.execute("""
    INSERT INTO order_items VALUES (%s, %s, %s)
""", ('ord_1', 'A1', 2))

# Query with JOINs
cur.execute("""
    SELECT u.name, a.city, COUNT(o.order_id) as order_count
    FROM users u
    LEFT JOIN addresses a ON u.user_id = a.user_id
    LEFT JOIN orders o ON u.user_id = o.user_id
    WHERE a.city = 'NYC'
    GROUP BY u.user_id, u.name, a.city
""")
```

**Pros:**
- ✅ Best query performance (proper indexes)
- ✅ Data integrity (foreign keys)
- ✅ No data duplication
- ✅ Easy to enforce constraints

**Cons:**
- ❌ Complex schema
- ❌ Requires migrations for changes
- ❌ Many JOINs can be slow

---

**Comparison table:**

| Aspect | Flattened | JSONB | Normalized |
|--------|-----------|-------|------------|
| Query speed | ⚠️ Medium | ⚠️ Medium | ✅ Fast |
| Schema flexibility | ❌ Low | ✅ High | ❌ Low |
| Data integrity | ⚠️ Medium | ❌ Low | ✅ High |
| Storage efficiency | ❌ Low (NULLs) | ✅ Good | ✅ Good |
| Ease of use | ✅ Easy | ⚠️ Medium | ❌ Complex |

**Recommendation:**

| Use Case | Best Choice |
|----------|-------------|
| Analytics on API logs | **Flattened** (easy to query) |
| Flexible metadata storage | **JSONB** (schema evolves) |
| Production transactional DB | **Normalized** (data integrity) |
| User preferences/settings | **JSONB** (variable fields) |
| E-commerce orders | **Normalized** (complex relationships) |

**Hybrid approach (best for most cases):**
```sql
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(200),  -- Core fields flattened
    metadata JSONB  -- Flexible fields in JSONB
);

CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    amount FLOAT,  -- Core fields flattened
    details JSONB  -- Variable fields in JSONB
);
```

This gives you fast queries on core fields + flexibility for variable data.

---

### Q9: Explain database connection pooling. Why is it important for ML applications? Show how to implement it with SQLAlchemy and diagnose common pooling issues.

**Answer:**

**What is connection pooling?**

Without pooling:
```python
# BAD: Create new connection for every query
for i in range(1000):
    conn = psycopg2.connect("postgresql://...")  # Slow!
    cur = conn.cursor()
    cur.execute("SELECT * FROM data WHERE id = %s", (i,))
    conn.close()
```

With pooling:
```python
# GOOD: Reuse connections from pool
pool = psycopg2.pool.SimpleConnectionPool(minconn=5, maxconn=20, ...)
for i in range(1000):
    conn = pool.getconn()  # Fast! Reuses existing connection
    cur = conn.cursor()
    cur.execute("SELECT * FROM data WHERE id = %s", (i,))
    pool.putconn(conn)
```

**Why it matters for ML:**
1. ML training queries database thousands of times
2. Creating connections is expensive (100-500ms each)
3. Database has max connection limit (typically 100)

**SQLAlchemy implementation:**

```python
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import time

# Configure connection pool
engine = create_engine(
    'postgresql://user:pass@localhost/mldb',
    poolclass=QueuePool,
    pool_size=10,          # Core pool size
    max_overflow=20,       # Additional connections if needed
    pool_timeout=30,       # Wait 30s for connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True,    # Check connection before using
    echo_pool=True         # Debug pool activity
)

# Usage in ML training loop
def get_batch_features(user_ids):
    """Get features for batch of users."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM features WHERE user_id = ANY(:ids)"),
            {"ids": user_ids}
        )
        return result.fetchall()

# Training loop
for epoch in range(10):
    for batch in batches:
        features = get_batch_features(batch['user_ids'])
        # Train model...
```

**Performance comparison:**

```python
import time
import psycopg2
from psycopg2 import pool

# Without pooling
def query_without_pool(n_queries=1000):
    start = time.time()
    for i in range(n_queries):
        conn = psycopg2.connect("postgresql://...")
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
    return time.time() - start

# With pooling
def query_with_pool(n_queries=1000):
    pool = psycopg2.pool.SimpleConnectionPool(
        minconn=5,
        maxconn=20,
        host="localhost",
        database="testdb",
        user="user",
        password="pass"
    )

    start = time.time()
    for i in range(n_queries):
        conn = pool.getconn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        pool.putconn(conn)

    pool.closeall()
    return time.time() - start

print(f"Without pool: {query_without_pool(1000):.2f}s")
print(f"With pool: {query_with_pool(1000):.2f}s")
```

**Output:**
```
Without pool: 85.3s
With pool: 2.1s
Speedup: 40.6x
```

---

**Common pooling issues and solutions:**

**Issue 1: Pool exhaustion**

```python
# Symptom: TimeoutError after pool_timeout seconds
# Cause: Not returning connections to pool

# BAD
def bad_code():
    conn = engine.connect()
    result = conn.execute("SELECT * FROM data")
    return result.fetchall()
    # Connection never returned!

# GOOD
def good_code():
    with engine.connect() as conn:  # Automatically returns to pool
        result = conn.execute("SELECT * FROM data")
        return result.fetchall()
```

**Issue 2: Stale connections**

```python
# Symptom: "server closed the connection unexpectedly"
# Cause: Database closes idle connections

# Solution: Use pool_pre_ping and pool_recycle
engine = create_engine(
    'postgresql://...',
    pool_pre_ping=True,      # Test connection before use
    pool_recycle=3600        # Recycle after 1 hour
)
```

**Issue 3: Connection leaks**

```python
# Diagnostic: Monitor pool stats
from sqlalchemy import event

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    print(f"New connection created: {id(dbapi_conn)}")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    print(f"Connection checked out: {id(dbapi_conn)}")
    print(f"Pool size: {engine.pool.size()}")
    print(f"Checked out: {engine.pool.checkedout()}")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    print(f"Connection returned: {id(dbapi_conn)}")
```

**Issue 4: Too many connections**

```python
# Check current connections
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT count(*) as connection_count,
               state,
               application_name
        FROM pg_stat_activity
        WHERE datname = 'mldb'
        GROUP BY state, application_name
    """))
    print(result.fetchall())

# If too many, reduce pool_size + max_overflow
engine = create_engine(
    'postgresql://...',
    pool_size=5,        # Reduce from 10
    max_overflow=10     # Reduce from 20
)
```

**Best practices for ML applications:**

```python
# 1. Use context managers
with engine.connect() as conn:
    result = conn.execute(...)

# 2. Set appropriate pool size
# Rule of thumb: pool_size = number of workers * 2
engine = create_engine(
    'postgresql://...',
    pool_size=20,  # For 10 worker processes
    max_overflow=10
)

# 3. Use connection pooling with multiprocessing
from multiprocessing import Pool as ProcessPool

def worker_init():
    """Create engine per worker process."""
    global engine
    engine = create_engine('postgresql://...', pool_size=2)

def process_batch(batch_id):
    with engine.connect() as conn:
        # Process batch...
        pass

if __name__ == '__main__':
    with ProcessPool(processes=10, initializer=worker_init) as pool:
        pool.map(process_batch, range(100))

# 4. Monitor pool health
def check_pool_health(engine):
    pool = engine.pool
    print(f"Pool size: {pool.size()}")
    print(f"Checked out: {pool.checkedout()}")
    print(f"Overflow: {pool.overflow()}")
    print(f"Checked in: {pool.size() - pool.checkedout()}")
```

---

### Q10: Design an ETL pipeline that reads from PostgreSQL, transforms in Spark, and writes to Parquet partitioned by date. Include error handling, monitoring, and incremental updates.

**Answer:**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, current_timestamp
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self, db_url, output_path):
        self.db_url = db_url
        self.output_path = output_path
        self.engine = create_engine(db_url, pool_size=10)

        # Initialize Spark
        self.spark = SparkSession.builder \
            .appName("ETL Pipeline") \
            .config("spark.jars", "postgresql-42.5.0.jar") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()

        # Monitoring table
        self._init_monitoring()

    def _init_monitoring(self):
        """Create monitoring table for tracking runs."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS etl_runs (
                    run_id SERIAL PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status VARCHAR(20),
                    rows_processed INTEGER,
                    error_message TEXT,
                    partition_date DATE
                )
            """))
            conn.commit()

    def get_last_successful_run(self):
        """Get last successful run timestamp for incremental loads."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MAX(partition_date)
                FROM etl_runs
                WHERE status = 'success'
            """))
            last_date = result.scalar()
            return last_date or datetime.now().date() - timedelta(days=30)

    def extract(self, start_date, end_date):
        """Extract data from PostgreSQL."""
        logger.info(f"Extracting data from {start_date} to {end_date}")

        try:
            # Read from PostgreSQL using Spark JDBC
            df = self.spark.read \
                .format("jdbc") \
                .option("url", self.db_url.replace('postgresql://', 'jdbc:postgresql://')) \
                .option("dbtable", f"""
                    (SELECT *
                     FROM transactions
                     WHERE created_at >= '{start_date}'
                       AND created_at < '{end_date}') as t
                """) \
                .option("user", "user") \
                .option("password", "pass") \
                .option("numPartitions", "10") \
                .option("fetchsize", "10000") \
                .load()

            logger.info(f"Extracted {df.count():,} rows")
            return df

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

    def transform(self, df):
        """Transform data."""
        logger.info("Transforming data")

        try:
            # Add derived columns
            df = df.withColumn("transaction_date", to_date(col("created_at")))
            df = df.withColumn("amount_usd", col("amount") * col("exchange_rate"))
            df = df.withColumn("processed_at", current_timestamp())

            # Filter out invalid records
            df = df.filter(col("amount") > 0)
            df = df.filter(col("user_id").isNotNull())

            # Deduplicate
            df = df.dropDuplicates(["transaction_id"])

            logger.info(f"Transformed {df.count():,} rows")
            return df

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise

    def load(self, df, partition_date):
        """Load data to Parquet."""
        logger.info(f"Loading data for partition {partition_date}")

        try:
            # Write to Parquet with partitioning
            df.write \
                .mode("overwrite") \
                .partitionBy("transaction_date") \
                .parquet(self.output_path, compression="snappy")

            rows_written = df.count()
            logger.info(f"Loaded {rows_written:,} rows to {self.output_path}")

            return rows_written

        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise

    def log_run(self, status, rows_processed, partition_date, error_message=None):
        """Log run status to monitoring table."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO etl_runs (start_time, end_time, status, rows_processed, error_message, partition_date)
                VALUES (:start, NOW(), :status, :rows, :error, :date)
            """), {
                "start": self.run_start_time,
                "status": status,
                "rows": rows_processed,
                "error": error_message,
                "date": partition_date
            })
            conn.commit()

    def run_incremental(self):
        """Run incremental ETL pipeline."""
        self.run_start_time = datetime.now()

        try:
            # Get date range for incremental load
            last_run_date = self.get_last_successful_run()
            today = datetime.now().date()

            logger.info(f"Running incremental ETL from {last_run_date} to {today}")

            # Extract
            df = self.extract(last_run_date, today)

            if df.count() == 0:
                logger.info("No new data to process")
                self.log_run("success", 0, today)
                return

            # Transform
            df_transformed = self.transform(df)

            # Load
            rows_processed = self.load(df_transformed, today)

            # Log success
            self.log_run("success", rows_processed, today)
            logger.info(f"ETL completed successfully. Processed {rows_processed:,} rows")

        except Exception as e:
            # Log failure
            self.log_run("failed", 0, today, str(e))
            logger.error(f"ETL failed: {e}")
            raise

        finally:
            self.spark.stop()

    def validate_output(self, partition_date):
        """Validate output data quality."""
        logger.info(f"Validating output for {partition_date}")

        # Read written Parquet
        df = self.spark.read.parquet(self.output_path)
        df_partition = df.filter(col("transaction_date") == partition_date)

        # Check 1: No null values in critical columns
        null_counts = df_partition.select([
            col(c).isNull().cast("int").alias(c)
            for c in ["transaction_id", "user_id", "amount"]
        ]).agg(*[sum(c).alias(c) for c in ["transaction_id", "user_id", "amount"]])

        null_result = null_counts.collect()[0].asDict()
        if any(null_result.values()):
            raise ValueError(f"Found null values: {null_result}")

        # Check 2: Amount is positive
        negative_count = df_partition.filter(col("amount") <= 0).count()
        if negative_count > 0:
            raise ValueError(f"Found {negative_count} negative amounts")

        # Check 3: Compare row counts with source
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) FROM transactions
                WHERE DATE(created_at) = :date
            """), {"date": partition_date})
            source_count = result.scalar()

        output_count = df_partition.count()
        if abs(output_count - source_count) > source_count * 0.01:  # Allow 1% difference
            raise ValueError(
                f"Row count mismatch: source={source_count}, output={output_count}"
            )

        logger.info("Validation passed")

# Usage
if __name__ == "__main__":
    pipeline = ETLPipeline(
        db_url="postgresql://user:pass@localhost/mldb",
        output_path="s3://bucket/data/transactions/"
    )

    # Run incremental ETL
    pipeline.run_incremental()

    # Validate
    pipeline.validate_output(datetime.now().date())
```

**Schedule with Airflow:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'etl_pipeline',
    default_args=default_args,
    description='ETL pipeline from PostgreSQL to Parquet',
    schedule_interval='0 2 * * *',  # Daily at 2am
    start_date=datetime(2024, 1, 1),
    catchup=False
)

def run_etl():
    pipeline = ETLPipeline(
        db_url="postgresql://user:pass@localhost/mldb",
        output_path="s3://bucket/data/transactions/"
    )
    pipeline.run_incremental()

def validate_etl():
    pipeline = ETLPipeline(
        db_url="postgresql://user:pass@localhost/mldb",
        output_path="s3://bucket/data/transactions/"
    )
    pipeline.validate_output(datetime.now().date())

etl_task = PythonOperator(
    task_id='run_etl',
    python_callable=run_etl,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_etl',
    python_callable=validate_etl,
    dag=dag
)

etl_task >> validate_task
```

**Key features:**
- Incremental loads (only process new data)
- Error handling and logging
- Monitoring table for tracking runs
- Data validation
- Spark for distributed processing
- Partitioned output for efficient queries

---

## Implementation & Coding (Q11-Q20)

### Q11: Write a function that converts a 5GB CSV file to Parquet in chunks without loading the entire file into memory. Include error handling and progress tracking.

**Answer:**

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

def csv_to_parquet_chunked(
    csv_path: str,
    parquet_path: str,
    chunksize: int = 100_000,
    compression: str = 'snappy'
) -> None:
    """
    Convert large CSV to Parquet in chunks to avoid memory issues.

    Args:
        csv_path: Input CSV file path
        parquet_path: Output Parquet file path
        chunksize: Number of rows per chunk
        compression: Compression algorithm (snappy, gzip, zstd)
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Get total rows for progress bar
    print("Counting rows...")
    total_rows = sum(1 for _ in open(csv_path)) - 1  # -1 for header

    # Initialize writer
    writer = None
    schema = None

    try:
        # Read CSV in chunks
        chunks = pd.read_csv(csv_path, chunksize=chunksize)

        with tqdm(total=total_rows, desc="Converting", unit=" rows") as pbar:
            for i, chunk in enumerate(chunks):
                # Clean data
                chunk = chunk.dropna(how='all')  # Remove empty rows

                # Convert to Arrow Table
                table = pa.Table.from_pandas(chunk, preserve_index=False)

                if writer is None:
                    # Create writer on first chunk
                    schema = table.schema
                    writer = pq.ParquetWriter(
                        parquet_path,
                        schema,
                        compression=compression
                    )

                # Write chunk
                writer.write_table(table)
                pbar.update(len(chunk))

        print(f"✅ Successfully converted {total_rows:,} rows")

        # Show file sizes
        csv_size = csv_path.stat().st_size / (1024**3)
        parquet_size = parquet_path.stat().st_size / (1024**3)
        ratio = csv_size / parquet_size

        print(f"CSV size: {csv_size:.2f} GB")
        print(f"Parquet size: {parquet_size:.2f} GB")
        print(f"Compression ratio: {ratio:.2f}x")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        if parquet_path.exists():
            parquet_path.unlink()  # Clean up partial file
        raise

    finally:
        if writer is not None:
            writer.close()

# Usage
csv_to_parquet_chunked(
    'large_file.csv',
    'large_file.parquet',
    chunksize=100_000,
    compression='snappy'
)
```

**Key features:**
- Chunked processing (doesn't load full file)
- Progress bar with tqdm
- Error handling and cleanup
- File size comparison
- Schema inference from first chunk

---

### Q12: Write SQL queries for: (1) Top 10 users by purchase amount with running total, (2) Users who made purchases in 3 consecutive months, (3) Month-over-month growth rate.

**Answer:**

**Q1: Top 10 users with running total**
```sql
WITH user_totals AS (
  SELECT
    user_id,
    SUM(amount) as total_amount
  FROM purchases
  GROUP BY user_id
  ORDER BY total_amount DESC
  LIMIT 10
)
SELECT
  user_id,
  total_amount,
  SUM(total_amount) OVER (ORDER BY total_amount DESC) as running_total,
  ROUND(
    100.0 * SUM(total_amount) OVER (ORDER BY total_amount DESC) /
    SUM(total_amount) OVER (),
    2
  ) as cumulative_pct
FROM user_totals;
```

**Q2: Users with 3 consecutive months of purchases**
```sql
WITH monthly_purchases AS (
  SELECT DISTINCT
    user_id,
    DATE_TRUNC('month', purchase_date) as month
  FROM purchases
),
consecutive_check AS (
  SELECT
    user_id,
    month,
    month - (ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY month) || ' months')::interval as grp
  FROM monthly_purchases
)
SELECT user_id
FROM consecutive_check
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

**Q3: Month-over-month growth rate**
```sql
WITH monthly_revenue AS (
  SELECT
    DATE_TRUNC('month', purchase_date) as month,
    SUM(amount) as revenue
  FROM purchases
  GROUP BY DATE_TRUNC('month', purchase_date)
)
SELECT
  month,
  revenue,
  LAG(revenue) OVER (ORDER BY month) as prev_month_revenue,
  ROUND(
    100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) /
    LAG(revenue) OVER (ORDER BY month),
    2
  ) as growth_rate_pct
FROM monthly_revenue
ORDER BY month;
```

---

### Q13: Use SQLAlchemy to create a model for storing ML experiment results with relationships. Include methods for querying best models and comparing experiments.

**Answer:**

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    runs = relationship('ExperimentRun', back_populates='experiment', cascade='all, delete-orphan')

    def get_best_run(self, metric='accuracy', ascending=False):
        """Get best run for this experiment by metric."""
        session = Session.object_session(self)
        query = session.query(ExperimentRun)\
            .filter(ExperimentRun.experiment_id == self.id)

        if ascending:
            return query.order_by(ExperimentRun.metrics[metric].astext.cast(Float)).first()
        else:
            return query.order_by(ExperimentRun.metrics[metric].astext.cast(Float).desc()).first()

class ExperimentRun(Base):
    __tablename__ = 'experiment_runs'

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    model_name = Column(String(100))
    hyperparameters = Column(JSON)
    metrics = Column(JSON)  # {accuracy: 0.95, f1: 0.93}
    training_time_sec = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    experiment = relationship('Experiment', back_populates='runs')

# Create tables
engine = create_engine('postgresql://user:pass@localhost/mldb')
Base.metadata.create_all(engine)

# Usage
with Session(engine) as session:
    # Create experiment
    exp = Experiment(
        name='Text Classification',
        description='Comparing BERT variants'
    )
    session.add(exp)
    session.flush()

    # Add runs
    run1 = ExperimentRun(
        experiment_id=exp.id,
        model_name='bert-base',
        hyperparameters={'lr': 2e-5, 'batch_size': 16},
        metrics={'accuracy': 0.92, 'f1': 0.91},
        training_time_sec=3600
    )
    run2 = ExperimentRun(
        experiment_id=exp.id,
        model_name='roberta-base',
        hyperparameters={'lr': 1e-5, 'batch_size': 16},
        metrics={'accuracy': 0.94, 'f1': 0.93},
        training_time_sec=4200
    )
    session.add_all([run1, run2])
    session.commit()

    # Query best model
    best = exp.get_best_run(metric='f1')
    print(f"Best model: {best.model_name} with F1={best.metrics['f1']}")

    # Compare all runs
    runs = session.query(ExperimentRun)\
        .filter(ExperimentRun.experiment_id == exp.id)\
        .order_by(ExperimentRun.metrics['accuracy'].astext.cast(Float).desc())\
        .all()

    for run in runs:
        print(f"{run.model_name}: acc={run.metrics['accuracy']}, f1={run.metrics['f1']}")
```

---

### Q14: Implement a function to safely execute parameterized SQL queries with proper SQL injection prevention. Show both vulnerable and secure examples.

**Answer:**

**Vulnerable code (SQL injection):**

```python
# ❌ NEVER DO THIS - Vulnerable to SQL injection
def get_user_vulnerable(username):
    import psycopg2
    conn = psycopg2.connect("postgresql://...")
    cur = conn.cursor()

    # String concatenation - DANGEROUS!
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cur.execute(query)

    return cur.fetchall()

# Attack: username = "admin' OR '1'='1"
# Resulting query: SELECT * FROM users WHERE username = 'admin' OR '1'='1'
# Returns ALL users!
```

**SQL injection attack examples:**

```python
# Attack 1: Bypass authentication
username = "admin' OR '1'='1' --"
# Query becomes: SELECT * FROM users WHERE username = 'admin' OR '1'='1' --'

# Attack 2: Drop table
username = "admin'; DROP TABLE users; --"
# Query becomes: SELECT * FROM users WHERE username = 'admin'; DROP TABLE users; --'

# Attack 3: Data exfiltration
username = "admin' UNION SELECT password FROM admin_users --"
# Query becomes: SELECT * FROM users WHERE username = 'admin' UNION SELECT password FROM admin_users --'
```

---

**Secure code (parameterized queries):**

```python
import psycopg2
from psycopg2 import sql
from typing import List, Dict, Any

def get_user_secure(username: str) -> List[Dict[str, Any]]:
    """
    Securely query user by username using parameterized query.

    Args:
        username: Username to search for

    Returns:
        List of user records
    """
    conn = psycopg2.connect("postgresql://...")
    cur = conn.cursor()

    # ✅ SECURE: Use parameterized query
    query = "SELECT * FROM users WHERE username = %s"
    cur.execute(query, (username,))

    return cur.fetchall()

# Attack attempt fails - username is treated as literal string
username = "admin' OR '1'='1"
# Query searches for username literally: "admin' OR '1'='1"
# Returns nothing (no user with that exact username)
```

---

**Secure implementation with SQLAlchemy:**

```python
from sqlalchemy import create_engine, text
from typing import List, Dict, Any, Optional

class SecureDatabase:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute parameterized query securely.

        Args:
            query: SQL query with named parameters
            params: Dictionary of parameter values

        Returns:
            List of result rows as dictionaries
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return [dict(row._mapping) for row in result]

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        query = "SELECT * FROM users WHERE username = :username"
        results = self.execute_query(query, {"username": username})
        return results[0] if results else None

    def get_users_by_criteria(
        self,
        min_age: Optional[int] = None,
        city: Optional[str] = None,
        email_domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get users matching criteria.
        Shows how to build dynamic queries securely.
        """
        conditions = []
        params = {}

        if min_age is not None:
            conditions.append("age >= :min_age")
            params["min_age"] = min_age

        if city is not None:
            conditions.append("city = :city")
            params["city"] = city

        if email_domain is not None:
            conditions.append("email LIKE :email_pattern")
            params["email_pattern"] = f"%@{email_domain}"

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM users WHERE {where_clause}"

        return self.execute_query(query, params)

# Usage
db = SecureDatabase("postgresql://user:pass@localhost/mydb")

# Safe from injection
user = db.get_user_by_username("admin' OR '1'='1")  # Returns None

# Dynamic query building (still safe)
users = db.get_users_by_criteria(
    min_age=18,
    city="NYC",
    email_domain="company.com"
)
```

---

**Secure dynamic table/column names:**

```python
from psycopg2 import sql

def get_column_values(table_name: str, column_name: str, limit: int = 10):
    """
    Safely query dynamic table and column names.

    IMPORTANT: Table/column names cannot be parameterized with %s.
    Use psycopg2.sql.Identifier instead.
    """
    conn = psycopg2.connect("postgresql://...")
    cur = conn.cursor()

    # ✅ SECURE: Use sql.Identifier for table/column names
    query = sql.SQL("SELECT {column} FROM {table} LIMIT %s").format(
        column=sql.Identifier(column_name),
        table=sql.Identifier(table_name)
    )

    cur.execute(query, (limit,))
    return cur.fetchall()

# Even malicious input is escaped properly
get_column_values("users; DROP TABLE users; --", "name")
# Looks for table literally named "users; DROP TABLE users; --"
```

---

**Input validation layer:**

```python
import re
from typing import Any

class InputValidator:
    """Validate and sanitize inputs before database queries."""

    @staticmethod
    def validate_username(username: str) -> str:
        """Validate username format."""
        if not username or len(username) > 50:
            raise ValueError("Invalid username length")

        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            raise ValueError("Username contains invalid characters")

        return username

    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")

        return email

    @staticmethod
    def validate_integer_range(
        value: Any,
        min_val: int,
        max_val: int,
        field_name: str = "value"
    ) -> int:
        """Validate integer is within range."""
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be an integer")

        if not (min_val <= int_value <= max_val):
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val}"
            )

        return int_value

# Usage with validation
def get_user_safe(username: str):
    # Validate input
    username = InputValidator.validate_username(username)

    # Execute parameterized query
    db = SecureDatabase("postgresql://...")
    return db.get_user_by_username(username)
```

---

**Complete secure example:**

```python
from sqlalchemy import create_engine, text
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SecureUserRepository:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def create_user(self, username: str, email: str, age: int) -> int:
        """Create user with validation and parameterized query."""
        # Validate inputs
        username = InputValidator.validate_username(username)
        email = InputValidator.validate_email(email)
        age = InputValidator.validate_integer_range(age, 1, 150, "age")

        try:
            with self.engine.begin() as conn:  # Transaction
                result = conn.execute(
                    text("""
                        INSERT INTO users (username, email, age)
                        VALUES (:username, :email, :age)
                        RETURNING id
                    """),
                    {"username": username, "email": email, "age": age}
                )
                user_id = result.scalar()
                logger.info(f"Created user {username} with ID {user_id}")
                return user_id

        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise

    def search_users(
        self,
        search_term: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search users by username or email.
        Uses parameterized LIKE query.
        """
        # Validate limit
        limit = InputValidator.validate_integer_range(limit, 1, 1000, "limit")

        # Sanitize search term (remove wildcards)
        search_term = search_term.replace("%", "").replace("_", "")

        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, username, email
                    FROM users
                    WHERE username ILIKE :pattern
                       OR email ILIKE :pattern
                    LIMIT :limit
                """),
                {
                    "pattern": f"%{search_term}%",
                    "limit": limit
                }
            )
            return [dict(row._mapping) for row in result]

# Usage
repo = SecureUserRepository("postgresql://user:pass@localhost/mydb")

# Safe from injection
repo.create_user(
    username="admin' OR '1'='1",  # Treated as literal string
    email="test@example.com",
    age=25
)

# Safe search
users = repo.search_users("admin'; DROP TABLE users; --")
# Searches for literal string, no SQL injection
```

**Key principles:**
1. ✅ Always use parameterized queries (%s, :name)
2. ✅ Never concatenate user input into SQL strings
3. ✅ Use sql.Identifier for dynamic table/column names
4. ✅ Validate and sanitize inputs
5. ✅ Use ORM when possible (SQLAlchemy handles escaping)
6. ✅ Limit database user permissions (principle of least privilege)

---

### Q15: Write a script to batch process 1M rows from PostgreSQL, apply transformations with pandas, and write results to Parquet. Include memory management and progress tracking.

**Answer:**

```python
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import Iterator, Callable
import logging
from datetime import datetime
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Process large datasets in batches to manage memory."""

    def __init__(
        self,
        db_url: str,
        batch_size: int = 10_000,
        output_path: str = "output.parquet"
    ):
        self.db_url = db_url
        self.batch_size = batch_size
        self.output_path = output_path
        self.conn = None

    def __enter__(self):
        self.conn = psycopg2.connect(self.db_url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def iter_batches(self, query: str) -> Iterator[pd.DataFrame]:
        """
        Iterate over query results in batches.

        Args:
            query: SQL query to execute

        Yields:
            DataFrames of batch_size rows
        """
        cursor_name = f"cursor_{datetime.now().timestamp()}"

        with self.conn.cursor(name=cursor_name) as cur:
            # Server-side cursor for memory efficiency
            cur.itersize = self.batch_size
            cur.execute(query)

            while True:
                rows = cur.fetchmany(self.batch_size)
                if not rows:
                    break

                # Convert to DataFrame
                columns = [desc[0] for desc in cur.description]
                df = pd.DataFrame(rows, columns=columns)

                yield df

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to a batch.

        Override this method for custom transformations.
        """
        # Example transformations
        df = df.copy()

        # Convert timestamps
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date
            df['hour'] = df['created_at'].dt.hour

        # Clean text columns
        if 'text' in df.columns:
            df['text'] = df['text'].str.strip()
            df['text_length'] = df['text'].str.len()

        # Numeric transformations
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])

        return df

    def process_and_write(
        self,
        query: str,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame] = None
    ):
        """
        Process batches and write to Parquet.

        Args:
            query: SQL query to fetch data
            transform_fn: Optional custom transformation function
        """
        transform_fn = transform_fn or self.transform_batch

        # Get total count for progress bar
        logger.info("Counting total rows...")
        with self.conn.cursor() as cur:
            count_query = f"SELECT COUNT(*) FROM ({query}) as t"
            cur.execute(count_query)
            total_rows = cur.fetchone()[0]

        logger.info(f"Processing {total_rows:,} rows in batches of {self.batch_size:,}")

        # Initialize Parquet writer
        writer = None
        schema = None
        total_processed = 0

        try:
            with tqdm(total=total_rows, desc="Processing", unit=" rows") as pbar:
                for batch_df in self.iter_batches(query):
                    # Transform batch
                    transformed_df = transform_fn(batch_df)

                    # Convert to Arrow Table
                    table = pa.Table.from_pandas(transformed_df, preserve_index=False)

                    # Initialize writer on first batch
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(
                            self.output_path,
                            schema,
                            compression='snappy'
                        )

                    # Write batch
                    writer.write_table(table)

                    # Update progress
                    batch_size = len(transformed_df)
                    total_processed += batch_size
                    pbar.update(batch_size)

                    # Memory management
                    del batch_df, transformed_df, table
                    gc.collect()

            logger.info(f"✅ Successfully processed {total_processed:,} rows")
            logger.info(f"Output written to: {self.output_path}")

        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise

        finally:
            if writer is not None:
                writer.close()

    def process_and_write_partitioned(
        self,
        query: str,
        partition_cols: list,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame] = None
    ):
        """
        Process batches and write to partitioned Parquet.

        Args:
            query: SQL query to fetch data
            partition_cols: Columns to partition by (e.g., ['year', 'month'])
            transform_fn: Optional custom transformation function
        """
        transform_fn = transform_fn or self.transform_batch

        logger.info(f"Processing with partitioning by {partition_cols}")

        accumulated_batches = []
        accumulated_rows = 0
        write_threshold = 100_000  # Write every 100k rows

        try:
            for batch_df in self.iter_batches(query):
                # Transform batch
                transformed_df = transform_fn(batch_df)
                accumulated_batches.append(transformed_df)
                accumulated_rows += len(transformed_df)

                # Write when accumulated enough rows
                if accumulated_rows >= write_threshold:
                    combined_df = pd.concat(accumulated_batches, ignore_index=True)

                    # Write with partitioning
                    combined_df.to_parquet(
                        self.output_path,
                        partition_cols=partition_cols,
                        compression='snappy',
                        engine='pyarrow',
                        mode='append'  # Append to existing partitions
                    )

                    logger.info(f"Wrote {accumulated_rows:,} rows")

                    # Clear accumulated data
                    accumulated_batches = []
                    accumulated_rows = 0
                    gc.collect()

            # Write remaining data
            if accumulated_batches:
                combined_df = pd.concat(accumulated_batches, ignore_index=True)
                combined_df.to_parquet(
                    self.output_path,
                    partition_cols=partition_cols,
                    compression='snappy',
                    engine='pyarrow',
                    mode='append'
                )

                logger.info(f"Wrote final {accumulated_rows:,} rows")

            logger.info("✅ Successfully completed partitioned write")

        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise

# Usage examples

def example_1_simple():
    """Simple batch processing."""
    with BatchProcessor(
        db_url="postgresql://user:pass@localhost/mydb",
        batch_size=10_000,
        output_path="output.parquet"
    ) as processor:
        query = "SELECT * FROM transactions WHERE created_at > '2024-01-01'"
        processor.process_and_write(query)

def example_2_custom_transform():
    """Custom transformation function."""
    def custom_transform(df: pd.DataFrame) -> pd.DataFrame:
        # Feature engineering
        df['amount_squared'] = df['amount'] ** 2
        df['is_weekend'] = pd.to_datetime(df['created_at']).dt.dayofweek >= 5

        # Filter invalid rows
        df = df[df['amount'] > 0]

        return df

    with BatchProcessor(
        db_url="postgresql://user:pass@localhost/mydb",
        batch_size=10_000,
        output_path="output.parquet"
    ) as processor:
        query = "SELECT * FROM transactions"
        processor.process_and_write(query, transform_fn=custom_transform)

def example_3_partitioned():
    """Partitioned output for efficient queries."""
    def add_partitions(df: pd.DataFrame) -> pd.DataFrame:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['year'] = df['created_at'].dt.year
        df['month'] = df['created_at'].dt.month
        return df

    with BatchProcessor(
        db_url="postgresql://user:pass@localhost/mydb",
        batch_size=10_000,
        output_path="output_partitioned/"
    ) as processor:
        query = "SELECT * FROM transactions"
        processor.process_and_write_partitioned(
            query,
            partition_cols=['year', 'month'],
            transform_fn=add_partitions
        )

def example_4_memory_monitoring():
    """Monitor memory usage during processing."""
    import psutil
    import os

    def monitor_memory(df: pd.DataFrame) -> pd.DataFrame:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 ** 2)
        logger.info(f"Memory usage: {memory_mb:.2f} MB")

        # Transform
        df = df.copy()
        df['processed'] = True

        return df

    with BatchProcessor(
        db_url="postgresql://user:pass@localhost/mydb",
        batch_size=5_000,  # Smaller batches for memory-constrained environments
        output_path="output.parquet"
    ) as processor:
        query = "SELECT * FROM large_table"
        processor.process_and_write(query, transform_fn=monitor_memory)

if __name__ == "__main__":
    # Run example
    example_1_simple()
```

**Key features:**
- Server-side cursors (don't load all data into memory)
- Batch processing with configurable size
- Progress tracking with tqdm
- Memory management (gc.collect())
- Parquet writing with streaming
- Partitioned output support
- Error handling and logging

**Memory usage comparison:**

```python
# Without batching (BAD for large datasets)
df = pd.read_sql("SELECT * FROM transactions", conn)  # Loads 1M rows → 8GB RAM
df_transformed = transform(df)
df_transformed.to_parquet("output.parquet")

# With batching (GOOD)
# Processes 10k rows at a time → 80MB RAM per batch
processor.process_and_write(query)
```

---

### Q16: Implement connection pooling with retry logic and circuit breaker pattern for database failures. Show how to handle transient failures gracefully.

**Answer:**

```python
from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, DBAPIError
import time
import logging
from typing import Callable, Any
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """
    Circuit breaker pattern for database connections.

    Prevents cascading failures by stopping requests to failing service.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker: HALF_OPEN (testing recovery)")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        return (
            self.last_failure_time is not None and
            datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)
        )

    def _on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker: CLOSED (recovered)")

    def _on_failure(self):
        """Handle failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        logger.warning(f"Circuit breaker: failure {self.failure_count}/{self.failure_threshold}")

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error("Circuit breaker: OPEN (too many failures)")

class ResilientDatabase:
    """
    Database client with connection pooling, retry logic, and circuit breaker.
    """

    def __init__(
        self,
        db_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.db_url = db_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create engine with connection pooling
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Test connections before use
            echo_pool=False
        )

        # Circuit breaker for database
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=(OperationalError, DBAPIError)
        )

        # Add event listeners
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring."""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug(f"New connection created: {id(dbapi_conn)}")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug(f"Connection checked out from pool: {id(dbapi_conn)}")

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            logger.debug(f"Connection returned to pool: {id(dbapi_conn)}")

    def execute_with_retry(
        self,
        query: str,
        params: dict = None
    ) -> Any:
        """
        Execute query with retry logic and circuit breaker.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Query result
        """
        def _execute():
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]

        # Execute with circuit breaker
        for attempt in range(self.max_retries):
            try:
                result = self.circuit_breaker.call(_execute)
                return result

            except (OperationalError, DBAPIError) as e:
                logger.warning(
                    f"Query failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error("Max retries exceeded")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    def execute_transaction(
        self,
        queries: list,
        params_list: list = None
    ) -> bool:
        """
        Execute multiple queries in a transaction with retry.

        Args:
            queries: List of SQL queries
            params_list: List of parameter dicts for each query

        Returns:
            True if successful
        """
        params_list = params_list or [{}] * len(queries)

        def _execute_transaction():
            with self.engine.begin() as conn:  # Automatic transaction
                for query, params in zip(queries, params_list):
                    conn.execute(text(query), params)

        for attempt in range(self.max_retries):
            try:
                self.circuit_breaker.call(_execute_transaction)
                logger.info("Transaction committed successfully")
                return True

            except (OperationalError, DBAPIError) as e:
                logger.warning(
                    f"Transaction failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    logger.error("Transaction failed after max retries")
                    raise

        return False

    def get_pool_status(self) -> dict:
        """Get current connection pool status."""
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.size() - pool.checkedout()
        }

    def health_check(self) -> bool:
        """Check database health."""
        try:
            result = self.execute_with_retry("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Usage examples

def example_1_basic_usage():
    """Basic usage with retry and circuit breaker."""
    db = ResilientDatabase(
        db_url="postgresql://user:pass@localhost/mydb",
        pool_size=10,
        max_retries=3,
        retry_delay=1.0
    )

    # Simple query with automatic retry
    try:
        results = db.execute_with_retry(
            "SELECT * FROM users WHERE age > :min_age",
            {"min_age": 18}
        )
        print(f"Found {len(results)} users")

    except Exception as e:
        print(f"Query failed: {e}")

def example_2_transaction():
    """Transaction with retry."""
    db = ResilientDatabase("postgresql://user:pass@localhost/mydb")

    queries = [
        "INSERT INTO accounts (user_id, balance) VALUES (:user_id, :balance)",
        "UPDATE users SET account_created = true WHERE id = :user_id"
    ]

    params_list = [
        {"user_id": 123, "balance": 1000.0},
        {"user_id": 123}
    ]

    try:
        db.execute_transaction(queries, params_list)
        print("Transaction completed")
    except Exception as e:
        print(f"Transaction failed: {e}")

def example_3_monitoring():
    """Monitor pool health."""
    db = ResilientDatabase("postgresql://user:pass@localhost/mydb")

    # Execute some queries
    for i in range(100):
        db.execute_with_retry("SELECT * FROM users LIMIT 10")

        if i % 10 == 0:
            status = db.get_pool_status()
            print(f"Pool status: {status}")

            health = db.health_check()
            print(f"Health: {'OK' if health else 'FAILING'}")

def example_4_simulate_failure():
    """Simulate database failure and recovery."""
    db = ResilientDatabase(
        db_url="postgresql://user:pass@localhost/mydb",
        max_retries=3,
        retry_delay=0.5
    )

    # Simulate temporary network issue
    for i in range(10):
        try:
            result = db.execute_with_retry("SELECT pg_sleep(0.1)")
            print(f"Query {i} succeeded")

        except Exception as e:
            print(f"Query {i} failed: {e}")

        time.sleep(1)

        # Check circuit breaker state
        print(f"Circuit state: {db.circuit_breaker.state}")

if __name__ == "__main__":
    example_1_basic_usage()
```

**Key features:**
- Connection pooling with SQLAlchemy
- Exponential backoff retry logic
- Circuit breaker pattern to prevent cascading failures
- Transaction support with rollback
- Pool monitoring and health checks
- Event listeners for debugging

**How circuit breaker works:**

```
Normal operation (CLOSED):
  Request → Database → Success

After 5 failures (OPEN):
  Request → Circuit Breaker → Reject (fail fast)

After 60s timeout (HALF_OPEN):
  Request → Database → Test recovery

If successful (CLOSED):
  Request → Database → Normal operation
```

This prevents overwhelming a failing database and allows it time to recover.

---

### Q17: Create a function to detect and handle Parquet file corruption. Include validation checks, partial file recovery, and error reporting.

**Answer:**

```python
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of Parquet file validation."""
    file_path: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    recovered_rows: Optional[int] = None

class ParquetValidator:
    """Validate and recover corrupted Parquet files."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.errors = []
        self.warnings = []

    def validate(self) -> ValidationResult:
        """
        Comprehensive validation of Parquet file.

        Returns:
            ValidationResult with detailed information
        """
        logger.info(f"Validating Parquet file: {self.file_path}")

        # Check 1: File exists
        if not self.file_path.exists():
            self.errors.append(f"File not found: {self.file_path}")
            return self._build_result(is_valid=False)

        # Check 2: File is not empty
        file_size = self.file_path.stat().st_size
        if file_size == 0:
            self.errors.append("File is empty (0 bytes)")
            return self._build_result(is_valid=False)

        # Check 3: Read metadata
        try:
            metadata = pq.read_metadata(self.file_path)
        except Exception as e:
            self.errors.append(f"Cannot read metadata: {e}")
            return self._build_result(is_valid=False)

        # Check 4: Validate metadata
        self._validate_metadata(metadata)

        # Check 5: Validate schema
        try:
            schema = pq.read_schema(self.file_path)
            self._validate_schema(schema)
        except Exception as e:
            self.errors.append(f"Invalid schema: {e}")

        # Check 6: Try reading data
        data_valid = self._validate_data()

        # Check 7: Validate row groups
        self._validate_row_groups(metadata)

        is_valid = len(self.errors) == 0

        return self._build_result(
            is_valid=is_valid,
            metadata={
                "file_size_mb": file_size / (1024 ** 2),
                "num_rows": metadata.num_rows,
                "num_row_groups": metadata.num_row_groups,
                "num_columns": metadata.num_columns,
                "created": metadata.created_by,
                "compression": metadata.row_group(0).column(0).compression
            }
        )

    def _validate_metadata(self, metadata: pq.FileMetaData):
        """Validate Parquet metadata."""
        # Check row count
        if metadata.num_rows == 0:
            self.warnings.append("File contains zero rows")

        # Check row groups
        if metadata.num_row_groups == 0:
            self.errors.append("File contains zero row groups")

        # Check columns
        if metadata.num_columns == 0:
            self.errors.append("File contains zero columns")

    def _validate_schema(self, schema: pa.Schema):
        """Validate Parquet schema."""
        # Check for duplicate column names
        column_names = [field.name for field in schema]
        if len(column_names) != len(set(column_names)):
            self.errors.append("Duplicate column names found")

        # Check for empty column names
        if any(not name for name in column_names):
            self.errors.append("Empty column names found")

    def _validate_data(self) -> bool:
        """Try reading actual data."""
        try:
            # Try reading first 1000 rows
            df = pq.read_table(
                self.file_path,
                columns=None,
                use_threads=True
            ).to_pandas(limit=1000)

            # Check for all-null columns
            null_cols = df.columns[df.isnull().all()].tolist()
            if null_cols:
                self.warnings.append(f"Columns with all null values: {null_cols}")

            return True

        except Exception as e:
            self.errors.append(f"Cannot read data: {e}")
            return False

    def _validate_row_groups(self, metadata: pq.FileMetaData):
        """Validate individual row groups."""
        for i in range(metadata.num_row_groups):
            try:
                row_group = metadata.row_group(i)

                # Check row group size
                if row_group.num_rows == 0:
                    self.warnings.append(f"Row group {i} is empty")

                # Check column chunks
                for j in range(row_group.num_columns):
                    column = row_group.column(j)

                    # Check for excessive compression ratio
                    if column.total_compressed_size > 0:
                        ratio = column.total_uncompressed_size / column.total_compressed_size
                        if ratio > 100:
                            self.warnings.append(
                                f"Row group {i}, column {j}: suspiciously high compression ratio ({ratio:.1f}x)"
                            )

            except Exception as e:
                self.errors.append(f"Error validating row group {i}: {e}")

    def _build_result(self, is_valid: bool, metadata: Dict[str, Any] = None) -> ValidationResult:
        """Build validation result."""
        return ValidationResult(
            file_path=str(self.file_path),
            is_valid=is_valid,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
            metadata=metadata or {}
        )

    def recover(self, output_path: str) -> ValidationResult:
        """
        Attempt to recover data from corrupted Parquet file.

        Args:
            output_path: Path to write recovered data

        Returns:
            ValidationResult with recovery information
        """
        logger.info(f"Attempting recovery of {self.file_path}")

        try:
            metadata = pq.read_metadata(self.file_path)
            num_row_groups = metadata.num_row_groups

            recovered_tables = []
            failed_row_groups = []

            # Try reading each row group individually
            for i in range(num_row_groups):
                try:
                    table = pq.read_table(
                        self.file_path,
                        use_threads=False,
                        filters=[('__row_group', '==', i)]  # Read specific row group
                    )
                    recovered_tables.append(table)
                    logger.info(f"✅ Recovered row group {i}/{num_row_groups}")

                except Exception as e:
                    logger.error(f"❌ Failed to read row group {i}: {e}")
                    failed_row_groups.append(i)

            if not recovered_tables:
                self.errors.append("No row groups could be recovered")
                return self._build_result(is_valid=False)

            # Concatenate recovered tables
            recovered_table = pa.concat_tables(recovered_tables)

            # Write recovered data
            pq.write_table(
                recovered_table,
                output_path,
                compression='snappy'
            )

            recovered_rows = len(recovered_table)
            original_rows = metadata.num_rows

            logger.info(
                f"✅ Recovered {recovered_rows}/{original_rows} rows "
                f"({100*recovered_rows/original_rows:.1f}%)"
            )

            if failed_row_groups:
                self.warnings.append(
                    f"Failed to recover row groups: {failed_row_groups}"
                )

            result = self._build_result(is_valid=True)
            result.recovered_rows = recovered_rows

            return result

        except Exception as e:
            self.errors.append(f"Recovery failed: {e}")
            return self._build_result(is_valid=False)

def validate_parquet_dataset(directory: str, fix_errors: bool = False) -> List[ValidationResult]:
    """
    Validate all Parquet files in a directory.

    Args:
        directory: Directory containing Parquet files
        fix_errors: Attempt to recover corrupted files

    Returns:
        List of validation results
    """
    dir_path = Path(directory)
    parquet_files = list(dir_path.rglob("*.parquet"))

    logger.info(f"Found {len(parquet_files)} Parquet files in {directory}")

    results = []

    for file_path in parquet_files:
        validator = ParquetValidator(str(file_path))
        result = validator.validate()

        if not result.is_valid and fix_errors:
            logger.warning(f"Attempting recovery of {file_path}")
            output_path = str(file_path).replace(".parquet", "_recovered.parquet")
            recovery_result = validator.recover(output_path)
            results.append(recovery_result)
        else:
            results.append(result)

    return results

def generate_validation_report(results: List[ValidationResult], output_file: str = None):
    """
    Generate human-readable validation report.

    Args:
        results: List of validation results
        output_file: Optional file to write report to
    """
    report = []
    report.append("=" * 80)
    report.append("Parquet Validation Report")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 80)
    report.append("")

    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = len(results) - valid_count

    report.append(f"Total files: {len(results)}")
    report.append(f"Valid: {valid_count}")
    report.append(f"Invalid: {invalid_count}")
    report.append("")

    for i, result in enumerate(results, 1):
        report.append(f"\n{i}. {result.file_path}")
        report.append(f"   Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")

        if result.metadata:
            report.append(f"   Rows: {result.metadata.get('num_rows', 'N/A'):,}")
            report.append(f"   Size: {result.metadata.get('file_size_mb', 0):.2f} MB")

        if result.recovered_rows is not None:
            report.append(f"   Recovered rows: {result.recovered_rows:,}")

        if result.errors:
            report.append("   Errors:")
            for error in result.errors:
                report.append(f"     - {error}")

        if result.warnings:
            report.append("   Warnings:")
            for warning in result.warnings:
                report.append(f"     - {warning}")

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Report written to {output_file}")
    else:
        print(report_text)

# Usage examples

def example_1_validate_single_file():
    """Validate a single Parquet file."""
    validator = ParquetValidator("data.parquet")
    result = validator.validate()

    if result.is_valid:
        print("✅ File is valid")
    else:
        print("❌ File is invalid:")
        for error in result.errors:
            print(f"  - {error}")

def example_2_recover_corrupted_file():
    """Recover data from corrupted file."""
    validator = ParquetValidator("corrupted.parquet")
    result = validator.recover("recovered.parquet")

    if result.recovered_rows:
        print(f"✅ Recovered {result.recovered_rows} rows")
    else:
        print("❌ Recovery failed")

def example_3_validate_dataset():
    """Validate entire dataset."""
    results = validate_parquet_dataset(
        "data/parquet/",
        fix_errors=True  # Attempt recovery
    )

    # Generate report
    generate_validation_report(results, "validation_report.txt")

if __name__ == "__main__":
    example_3_validate_dataset()
```

**Key features:**
- Comprehensive validation (metadata, schema, data)
- Row group-level recovery
- Detailed error reporting
- Dataset-wide validation
- Automatic recovery with fix_errors flag

---

### Q18: Write SQL queries demonstrating window functions for: (1) Running totals, (2) Ranking with gaps, (3) Lead/lag for time-series, (4) Moving averages. Include ML feature engineering use cases.

**Answer:**

```python
# Complete window functions guide with ML use cases
```

**Q1: Running totals (cumulative sums)**

```sql
-- Basic running total
SELECT
    user_id,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY transaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_total
FROM transactions
ORDER BY user_id, transaction_date;

-- Running total with reset each month
SELECT
    user_id,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY user_id, DATE_TRUNC('month', transaction_date)
        ORDER BY transaction_date
    ) as monthly_running_total
FROM transactions;

-- ML use case: Cumulative engagement score
SELECT
    user_id,
    event_timestamp,
    event_type,
    CASE
        WHEN event_type = 'login' THEN 1
        WHEN event_type = 'purchase' THEN 10
        WHEN event_type = 'review' THEN 5
        ELSE 0
    END as points,
    SUM(
        CASE
            WHEN event_type = 'login' THEN 1
            WHEN event_type = 'purchase' THEN 10
            WHEN event_type = 'review' THEN 5
            ELSE 0
        END
    ) OVER (
        PARTITION BY user_id
        ORDER BY event_timestamp
    ) as cumulative_engagement_score
FROM user_events;
```

---

**Q2: Ranking with gaps (RANK vs DENSE_RANK vs ROW_NUMBER)**

```sql
-- Different ranking methods
SELECT
    product_id,
    product_name,
    sales,
    -- ROW_NUMBER: 1, 2, 3, 4 (no gaps, no ties)
    ROW_NUMBER() OVER (ORDER BY sales DESC) as row_num,

    -- RANK: 1, 2, 2, 4 (gaps after ties)
    RANK() OVER (ORDER BY sales DESC) as rank,

    -- DENSE_RANK: 1, 2, 2, 3 (no gaps)
    DENSE_RANK() OVER (ORDER BY sales DESC) as dense_rank,

    -- Percentile rank
    PERCENT_RANK() OVER (ORDER BY sales DESC) as percentile
FROM products;

-- Ranking within groups
SELECT
    category,
    product_name,
    sales,
    RANK() OVER (
        PARTITION BY category
        ORDER BY sales DESC
    ) as category_rank
FROM products;

-- ML use case: Feature engineering for product recommendation
WITH ranked_products AS (
    SELECT
        user_id,
        product_id,
        purchase_count,
        -- Rank products by purchase frequency per user
        DENSE_RANK() OVER (
            PARTITION BY user_id
            ORDER BY purchase_count DESC
        ) as product_preference_rank,
        -- Percentile of product popularity
        PERCENT_RANK() OVER (
            PARTITION BY user_id
            ORDER BY purchase_count DESC
        ) as product_preference_percentile
    FROM user_purchases
)
SELECT *
FROM ranked_products
WHERE product_preference_rank <= 10;  -- Top 10 products per user
```

---

**Q3: Lead/Lag for time-series analysis**

```sql
-- LAG: Access previous row
SELECT
    user_id,
    purchase_date,
    amount,
    LAG(amount) OVER (
        PARTITION BY user_id
        ORDER BY purchase_date
    ) as previous_amount,
    LAG(purchase_date) OVER (
        PARTITION BY user_id
        ORDER BY purchase_date
    ) as previous_purchase_date,
    -- Days since last purchase
    purchase_date - LAG(purchase_date) OVER (
        PARTITION BY user_id
        ORDER BY purchase_date
    ) as days_since_last_purchase
FROM purchases;

-- LEAD: Access next row
SELECT
    user_id,
    event_date,
    event_type,
    LEAD(event_type) OVER (
        PARTITION BY user_id
        ORDER BY event_date
    ) as next_event,
    LEAD(event_date) OVER (
        PARTITION BY user_id
        ORDER BY event_date
    ) - event_date as days_until_next_event
FROM events;

-- ML use case: Churn prediction features
WITH user_activity AS (
    SELECT
        user_id,
        activity_date,
        LAG(activity_date, 1) OVER (
            PARTITION BY user_id ORDER BY activity_date
        ) as prev_activity_1,
        LAG(activity_date, 2) OVER (
            PARTITION BY user_id ORDER BY activity_date
        ) as prev_activity_2,
        LAG(activity_date, 3) OVER (
            PARTITION BY user_id ORDER BY activity_date
        ) as prev_activity_3
    FROM user_activities
)
SELECT
    user_id,
    activity_date,
    -- Average days between activities (churn signal)
    (
        (activity_date - prev_activity_1) +
        (prev_activity_1 - prev_activity_2) +
        (prev_activity_2 - prev_activity_3)
    ) / 3.0 as avg_days_between_activities,
    -- Increasing gap indicates potential churn
    (activity_date - prev_activity_1) >
    (prev_activity_1 - prev_activity_2) as activity_gap_increasing
FROM user_activity
WHERE prev_activity_3 IS NOT NULL;
```

---

**Q4: Moving averages and aggregations**

```sql
-- Simple moving average (SMA)
SELECT
    date,
    price,
    -- 7-day moving average
    AVG(price) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as sma_7,
    -- 30-day moving average
    AVG(price) OVER (
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as sma_30
FROM stock_prices;

-- Moving aggregations
SELECT
    user_id,
    transaction_date,
    amount,
    -- Moving average
    AVG(amount) OVER (
        PARTITION BY user_id
        ORDER BY transaction_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7_days,
    -- Moving sum
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY transaction_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_sum_7_days,
    -- Moving std dev
    STDDEV(amount) OVER (
        PARTITION BY user_id
        ORDER BY transaction_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_stddev_7_days,
    -- Moving count
    COUNT(*) OVER (
        PARTITION BY user_id
        ORDER BY transaction_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as transaction_count_7_days
FROM transactions;

-- RANGE vs ROWS (time-based windows)
SELECT
    user_id,
    transaction_timestamp,
    amount,
    -- Last 7 days (time-based)
    AVG(amount) OVER (
        PARTITION BY user_id
        ORDER BY transaction_timestamp
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) as avg_last_7_days,
    -- Last 7 transactions (row-based)
    AVG(amount) OVER (
        PARTITION BY user_id
        ORDER BY transaction_timestamp
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as avg_last_7_transactions
FROM transactions;
```

---

**Complete ML feature engineering example:**

```sql
-- Generate comprehensive time-series features for ML
WITH user_transactions AS (
    SELECT
        user_id,
        transaction_date,
        amount,
        category,
        -- Running totals
        SUM(amount) OVER (
            PARTITION BY user_id
            ORDER BY transaction_date
        ) as lifetime_value,
        -- Transaction counts
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY transaction_date
        ) as transaction_number,
        -- Days since last transaction
        transaction_date - LAG(transaction_date) OVER (
            PARTITION BY user_id
            ORDER BY transaction_date
        ) as days_since_last,
        -- Moving averages (last 30 days)
        AVG(amount) OVER (
            PARTITION BY user_id
            ORDER BY transaction_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
        ) as avg_amount_30d,
        -- Moving std dev (last 30 days)
        STDDEV(amount) OVER (
            PARTITION BY user_id
            ORDER BY transaction_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
        ) as stddev_amount_30d,
        -- Transaction frequency (last 90 days)
        COUNT(*) OVER (
            PARTITION BY user_id
            ORDER BY transaction_date
            RANGE BETWEEN INTERVAL '90 days' PRECEDING AND CURRENT ROW
        ) as transaction_count_90d,
        -- Category diversity (last 60 days)
        COUNT(DISTINCT category) OVER (
            PARTITION BY user_id
            ORDER BY transaction_date
            RANGE BETWEEN INTERVAL '60 days' PRECEDING AND CURRENT ROW
        ) as category_diversity_60d,
        -- Percentile rank
        PERCENT_RANK() OVER (
            PARTITION BY user_id
            ORDER BY amount
        ) as amount_percentile
    FROM transactions
),
features AS (
    SELECT
        user_id,
        transaction_date,
        amount,
        lifetime_value,
        transaction_number,
        days_since_last,
        avg_amount_30d,
        stddev_amount_30d,
        transaction_count_90d,
        category_diversity_60d,
        amount_percentile,
        -- Derived features
        amount / NULLIF(avg_amount_30d, 0) as amount_ratio_to_avg,
        CASE
            WHEN days_since_last IS NULL THEN 0
            WHEN days_since_last <= 7 THEN 1
            WHEN days_since_last <= 30 THEN 2
            WHEN days_since_last <= 90 THEN 3
            ELSE 4
        END as recency_bucket,
        CASE
            WHEN transaction_count_90d >= 10 THEN 'high_frequency'
            WHEN transaction_count_90d >= 5 THEN 'medium_frequency'
            ELSE 'low_frequency'
        END as frequency_segment
    FROM user_transactions
)
SELECT * FROM features;
```

**Usage in Python:**

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://...')

# Fetch features for ML model
features_df = pd.read_sql("""
    -- Your window function query here
""", engine)

# Use in ML model
from sklearn.ensemble import RandomForestClassifier

X = features_df[[
    'lifetime_value',
    'days_since_last',
    'avg_amount_30d',
    'transaction_count_90d',
    'category_diversity_60d'
]]

y = features_df['churned']  # Target variable

model = RandomForestClassifier()
model.fit(X, y)
```

**Key window function concepts:**
- `PARTITION BY`: Divides data into groups
- `ORDER BY`: Defines row order within partition
- `ROWS BETWEEN`: Row-based window frame
- `RANGE BETWEEN`: Value-based window frame
- `LAG/LEAD`: Access adjacent rows
- `RANK/ROW_NUMBER`: Assign rankings

---

### Q19: Implement a data versioning system for ML training datasets stored in Parquet. Track schema changes, data lineage, and enable rollback to previous versions.

**Answer:**

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import hashlib
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetVersion:
    """Represents a single version of a dataset."""

    def __init__(
        self,
        version: str,
        timestamp: datetime,
        file_path: str,
        schema_hash: str,
        num_rows: int,
        metadata: Dict[str, Any]
    ):
        self.version = version
        self.timestamp = timestamp
        self.file_path = file_path
        self.schema_hash = schema_hash
        self.num_rows = num_rows
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "schema_hash": self.schema_hash,
            "num_rows": self.num_rows,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            file_path=data["file_path"],
            schema_hash=data["schema_hash"],
            num_rows=data["num_rows"],
            metadata=data["metadata"]
        )

class DataVersionControl:
    """Version control system for ML datasets."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.versions_dir = self.base_path / "versions"
        self.metadata_file = self.base_path / "metadata.json"

        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self.versions = self._load_metadata()

    def _load_metadata(self) -> Dict[str, DatasetVersion]:
        """Load version metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    v["version"]: DatasetVersion.from_dict(v)
                    for v in data["versions"]
                }
        return {}

    def _save_metadata(self):
        """Save version metadata to disk."""
        data = {
            "versions": [v.to_dict() for v in self.versions.values()]
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_schema_hash(self, schema: pa.Schema) -> str:
        """Compute hash of schema for change detection."""
        schema_str = str(schema)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of data for integrity checking."""
        # Sample-based hash for large datasets
        sample = df.sample(min(1000, len(df)), random_state=42)
        data_str = sample.to_json()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def commit(
        self,
        df: pd.DataFrame,
        message: str,
        author: str,
        tags: List[str] = None
    ) -> str:
        """
        Commit a new version of the dataset.

        Args:
            df: DataFrame to version
            message: Commit message describing changes
            author: Who created this version
            tags: Optional tags (e.g., ['production', 'experiment_v2'])

        Returns:
            Version ID
        """
        # Generate version ID
        version_id = f"v{len(self.versions) + 1:04d}"
        timestamp = datetime.now()

        logger.info(f"Committing version {version_id}")

        # Compute schema hash
        schema = pa.Schema.from_pandas(df)
        schema_hash = self._compute_schema_hash(schema)

        # Check for schema changes
        if self.versions:
            last_version = self.get_latest_version()
            if last_version.schema_hash != schema_hash:
                logger.warning("⚠️  Schema changed from previous version")
                self._log_schema_changes(last_version.version, version_id, df)

        # Save data
        version_file = self.versions_dir / f"{version_id}.parquet"
        df.to_parquet(version_file, compression='snappy', index=False)

        # Create version object
        version = DatasetVersion(
            version=version_id,
            timestamp=timestamp,
            file_path=str(version_file),
            schema_hash=schema_hash,
            num_rows=len(df),
            metadata={
                "message": message,
                "author": author,
                "tags": tags or [],
                "data_hash": self._compute_data_hash(df),
                "file_size_mb": version_file.stat().st_size / (1024 ** 2)
            }
        )

        # Save version
        self.versions[version_id] = version
        self._save_metadata()

        logger.info(f"✅ Committed version {version_id}")
        logger.info(f"   Rows: {len(df):,}")
        logger.info(f"   Size: {version.metadata['file_size_mb']:.2f} MB")

        return version_id

    def checkout(self, version_id: str) -> pd.DataFrame:
        """
        Load a specific version of the dataset.

        Args:
            version_id: Version to load (e.g., 'v0001' or 'latest')

        Returns:
            DataFrame
        """
        if version_id == 'latest':
            version = self.get_latest_version()
        elif version_id in self.versions:
            version = self.versions[version_id]
        else:
            raise ValueError(f"Version {version_id} not found")

        logger.info(f"Checking out version {version.version}")
        df = pd.read_parquet(version.file_path)

        return df

    def get_latest_version(self) -> DatasetVersion:
        """Get the most recent version."""
        if not self.versions:
            raise ValueError("No versions available")

        return max(self.versions.values(), key=lambda v: v.timestamp)

    def list_versions(self) -> List[DatasetVersion]:
        """List all versions in chronological order."""
        return sorted(self.versions.values(), key=lambda v: v.timestamp)

    def diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions.

        Args:
            version1: First version ID
            version2: Second version ID

        Returns:
            Dictionary with differences
        """
        v1 = self.versions[version1]
        v2 = self.versions[version2]

        # Load both versions
        df1 = self.checkout(version1)
        df2 = self.checkout(version2)

        diff_result = {
            "version1": version1,
            "version2": version2,
            "row_count_diff": len(df2) - len(df1),
            "schema_changed": v1.schema_hash != v2.schema_hash,
            "time_diff": (v2.timestamp - v1.timestamp).total_seconds() / 3600,  # hours
        }

        # Schema differences
        if diff_result["schema_changed"]:
            cols1 = set(df1.columns)
            cols2 = set(df2.columns)

            diff_result["columns_added"] = list(cols2 - cols1)
            diff_result["columns_removed"] = list(cols1 - cols2)
            diff_result["columns_common"] = list(cols1 & cols2)

        # Data statistics diff
        if not diff_result["schema_changed"]:
            numeric_cols = df1.select_dtypes(include=['number']).columns
            stats_diff = {}

            for col in numeric_cols:
                stats_diff[col] = {
                    "mean_diff": df2[col].mean() - df1[col].mean(),
                    "std_diff": df2[col].std() - df1[col].std()
                }

            diff_result["statistics_diff"] = stats_diff

        return diff_result

    def _log_schema_changes(self, old_version: str, new_version: str, new_df: pd.DataFrame):
        """Log schema changes between versions."""
        old_df = self.checkout(old_version)

        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)

        added = new_cols - old_cols
        removed = old_cols - new_cols

        if added:
            logger.info(f"   Columns added: {added}")

        if removed:
            logger.info(f"   Columns removed: {removed}")

        # Check type changes
        for col in old_cols & new_cols:
            if old_df[col].dtype != new_df[col].dtype:
                logger.info(
                    f"   Column {col} type changed: {old_df[col].dtype} → {new_df[col].dtype}"
                )

    def create_branch(self, branch_name: str, from_version: str = 'latest'):
        """Create a branch from a specific version (for experimentation)."""
        # Implementation left as exercise
        pass

    def tag(self, version_id: str, tag: str):
        """Add a tag to a version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]
        if tag not in version.metadata["tags"]:
            version.metadata["tags"].append(tag)
            self._save_metadata()
            logger.info(f"Tagged {version_id} with '{tag}'")

    def get_by_tag(self, tag: str) -> List[DatasetVersion]:
        """Get all versions with a specific tag."""
        return [
            v for v in self.versions.values()
            if tag in v.metadata.get("tags", [])
        ]

    def export_lineage(self, output_file: str):
        """Export data lineage graph."""
        lineage = {
            "versions": [v.to_dict() for v in self.list_versions()]
        }

        with open(output_file, 'w') as f:
            json.dump(lineage, f, indent=2)

        logger.info(f"Lineage exported to {output_file}")

# Usage examples

def example_1_basic_versioning():
    """Basic version control usage."""
    dvc = DataVersionControl("data/ml_dataset")

    # Initial commit
    df_v1 = pd.DataFrame({
        'user_id': range(1000),
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'label': np.random.randint(0, 2, 1000)
    })

    v1 = dvc.commit(
        df_v1,
        message="Initial dataset",
        author="data-team",
        tags=["baseline"]
    )

    # Add new features
    df_v2 = df_v1.copy()
    df_v2['feature_3'] = np.random.randn(1000)
    df_v2['feature_4'] = np.random.randn(1000)

    v2 = dvc.commit(
        df_v2,
        message="Added feature_3 and feature_4",
        author="ml-engineer",
        tags=["experiment_1"]
    )

    # Load latest version
    df_latest = dvc.checkout('latest')
    print(f"Latest version: {df_latest.shape}")

    # Load specific version
    df_baseline = dvc.checkout(v1)
    print(f"Baseline version: {df_baseline.shape}")

def example_2_compare_versions():
    """Compare two versions."""
    dvc = DataVersionControl("data/ml_dataset")

    # Get diff
    diff = dvc.diff('v0001', 'v0002')

    print("Version comparison:")
    print(f"  Rows added: {diff['row_count_diff']}")
    print(f"  Schema changed: {diff['schema_changed']}")

    if diff['schema_changed']:
        print(f"  Columns added: {diff['columns_added']}")
        print(f"  Columns removed: {diff['columns_removed']}")

def example_3_production_workflow():
    """Production ML workflow with versioning."""
    dvc = DataVersionControl("data/production_dataset")

    # Daily ETL pipeline
    def daily_etl():
        # Extract and transform data
        df = extract_from_database()
        df = apply_transformations(df)

        # Commit new version
        version = dvc.commit(
            df,
            message=f"Daily ETL run - {datetime.now().date()}",
            author="etl-pipeline",
            tags=["daily"]
        )

        # Tag as production if passes validation
        if validate_data(df):
            dvc.tag(version, "production")
            logger.info(f"Version {version} promoted to production")

    # Training pipeline
    def train_model():
        # Always train on production-tagged data
        production_versions = dvc.get_by_tag("production")
        latest_prod = production_versions[-1]

        df = dvc.checkout(latest_prod.version)

        # Train model
        model = train_ml_model(df)

        return model, latest_prod.version

    # Reproducibility
    def reproduce_model(model_id: str, dataset_version: str):
        """Reproduce exact model training."""
        df = dvc.checkout(dataset_version)
        model = train_ml_model(df)
        return model

def example_4_list_versions():
    """List all versions with metadata."""
    dvc = DataVersionControl("data/ml_dataset")

    print("All versions:")
    print("-" * 80)

    for version in dvc.list_versions():
        print(f"\nVersion: {version.version}")
        print(f"  Date: {version.timestamp}")
        print(f"  Rows: {version.num_rows:,}")
        print(f"  Size: {version.metadata['file_size_mb']:.2f} MB")
        print(f"  Message: {version.metadata['message']}")
        print(f"  Author: {version.metadata['author']}")
        print(f"  Tags: {', '.join(version.metadata['tags'])}")

if __name__ == "__main__":
    example_1_basic_versioning()
```

**Key features:**
- Version tracking with metadata
- Schema change detection
- Data lineage
- Tagging system (production, experiment, etc.)
- Diff between versions
- Complete reproducibility
- Lightweight (stores actual data, not diffs)

This enables:
- Reproducible ML experiments
- Rollback to previous datasets
- Track data quality over time
- Debugging model performance degradation

---

### Q20: Build a monitoring dashboard query that tracks: (1) Table sizes over time, (2) Query performance metrics, (3) Index usage statistics, (4) Connection pool health. Use PostgreSQL system catalogs.

**Answer:**

```python
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DatabaseMonitor:
    """Monitor PostgreSQL database health and performance."""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def get_table_sizes(self) -> pd.DataFrame:
        """
        Get size of all tables in the database.

        Returns:
            DataFrame with table names and sizes
        """
        query = """
        SELECT
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
            pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes,
            pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
            pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as indexes_size,
            pg_stat_get_live_tuples(schemaname::regclass::oid) as row_count
        FROM pg_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df

    def get_query_performance(self) -> pd.DataFrame:
        """
        Get query performance statistics from pg_stat_statements.

        Requires pg_stat_statements extension to be enabled:
        CREATE EXTENSION pg_stat_statements;
        """
        query = """
        SELECT
            queryid,
            LEFT(query, 100) as query_preview,
            calls,
            total_exec_time / 1000 as total_time_sec,
            mean_exec_time / 1000 as mean_time_sec,
            min_exec_time / 1000 as min_time_sec,
            max_exec_time / 1000 as max_time_sec,
            stddev_exec_time / 1000 as stddev_time_sec,
            rows,
            100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) as cache_hit_ratio
        FROM pg_stat_statements
        ORDER BY total_exec_time DESC
        LIMIT 50;
        """

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            return df
        except Exception as e:
            print(f"pg_stat_statements not available: {e}")
            return pd.DataFrame()

    def get_index_usage(self) -> pd.DataFrame:
        """
        Get index usage statistics.

        Returns:
            DataFrame with index names and usage stats
        """
        query = """
        SELECT
            schemaname,
            tablename,
            indexname,
            idx_scan as index_scans,
            idx_tup_read as tuples_read,
            idx_tup_fetch as tuples_fetched,
            pg_size_pretty(pg_relation_size(indexrelid)) as index_size
        FROM pg_stat_user_indexes
        ORDER BY idx_scan ASC;  -- Show unused indexes first
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df

    def get_unused_indexes(self) -> pd.DataFrame:
        """
        Find indexes that are never used.

        These indexes waste space and slow down writes.
        """
        query = """
        SELECT
            schemaname,
            tablename,
            indexname,
            pg_size_pretty(pg_relation_size(indexrelid)) as index_size
        FROM pg_stat_user_indexes
        WHERE idx_scan = 0
          AND indexrelname NOT LIKE '%_pkey'  -- Exclude primary keys
        ORDER BY pg_relation_size(indexrelid) DESC;
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df

    def get_connection_stats(self) -> pd.DataFrame:
        """
        Get current connection statistics.

        Returns:
            DataFrame with connection states and counts
        """
        query = """
        SELECT
            state,
            application_name,
            COUNT(*) as connection_count,
            MAX(EXTRACT(EPOCH FROM (NOW() - state_change))) as max_idle_seconds
        FROM pg_stat_activity
        WHERE datname = current_database()
        GROUP BY state, application_name
        ORDER BY connection_count DESC;
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df

    def get_long_running_queries(self, min_duration_sec: int = 60) -> pd.DataFrame:
        """
        Find queries running longer than threshold.

        Args:
            min_duration_sec: Minimum query duration in seconds

        Returns:
            DataFrame with long-running queries
        """
        query = """
        SELECT
            pid,
            usename,
            application_name,
            client_addr,
            state,
            query_start,
            EXTRACT(EPOCH FROM (NOW() - query_start)) as duration_seconds,
            LEFT(query, 200) as query_preview
        FROM pg_stat_activity
        WHERE datname = current_database()
          AND state = 'active'
          AND query_start < NOW() - INTERVAL ':duration seconds'
        ORDER BY query_start;
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params={"duration": min_duration_sec}
            )

        return df

    def get_cache_hit_ratio(self) -> float:
        """
        Get overall cache hit ratio.

        A ratio below 90% indicates possible memory issues.
        """
        query = """
        SELECT
            SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit + heap_blks_read), 0) * 100 as cache_hit_ratio
        FROM pg_statio_user_tables;
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            ratio = result.scalar()

        return ratio if ratio is not None else 0.0

    def get_bloat_estimate(self) -> pd.DataFrame:
        """
        Estimate table and index bloat.

        Bloat occurs when UPDATE/DELETE leaves dead tuples.
        Run VACUUM to reclaim space.
        """
        query = """
        WITH constants AS (
            SELECT current_setting('block_size')::numeric AS bs, 23 AS hdr, 4 AS ma
        ),
        bloat_info AS (
            SELECT
                schemaname, tablename,
                (datawidth+(hdr+ma-(CASE WHEN hdr%ma=0 THEN ma ELSE hdr%ma END)))::numeric AS datahdr,
                (maxfracsum*(nullhdr+ma-(CASE WHEN nullhdr%ma=0 THEN ma ELSE nullhdr%ma END))) AS nullhdr2
            FROM (
                SELECT
                    schemaname, tablename, hdr, ma, bs,
                    SUM((1-null_frac)*avg_width) AS datawidth,
                    MAX(null_frac) AS maxfracsum,
                    hdr+(
                        SELECT 1+COUNT(*)/8
                        FROM pg_stats s2
                        WHERE null_frac<>0 AND s2.schemaname = s.schemaname AND s2.tablename = s.tablename
                    ) AS nullhdr
                FROM pg_stats s, constants
                GROUP BY 1,2,3,4,5
            ) AS foo
        )
        SELECT
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
            ROUND(CASE WHEN otta=0 OR sml.relpages=0 OR sml.relpages=otta THEN 0.0
                ELSE sml.relpages/otta::numeric END,1) AS bloat_ratio
        FROM bloat_info
        JOIN pg_class cc ON cc.relname = bloat_info.tablename
        JOIN pg_namespace nn ON cc.relnamespace = nn.oid AND nn.nspname = bloat_info.schemaname AND nn.nspname <> 'information_schema'
        JOIN (
            SELECT
                (datawidth+(hdr+ma-(CASE WHEN hdr%ma=0 THEN ma ELSE hdr%ma END)))::numeric AS otta,
                c.relname, n.nspname, c.reltuples, c.relpages
            FROM bloat_info
            JOIN pg_class c ON c.relname = bloat_info.tablename
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind='r'
        ) AS sml ON sml.nspname = bloat_info.schemaname AND sml.relname = bloat_info.tablename
        WHERE sml.relpages > 0
        ORDER BY bloat_ratio DESC;
        """

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            return df
        except Exception as e:
            print(f"Could not estimate bloat: {e}")
            return pd.DataFrame()

    def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive database health report.

        Returns:
            Dictionary with all metrics
        """
        print("Generating database health report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "table_sizes": self.get_table_sizes().to_dict('records'),
            "index_usage": self.get_index_usage().to_dict('records'),
            "unused_indexes": self.get_unused_indexes().to_dict('records'),
            "connection_stats": self.get_connection_stats().to_dict('records'),
            "long_running_queries": self.get_long_running_queries(min_duration_sec=30).to_dict('records'),
            "cache_hit_ratio": self.get_cache_hit_ratio(),
            "bloat_estimate": self.get_bloat_estimate().to_dict('records'),
            "query_performance": self.get_query_performance().to_dict('records')
        }

        return report

    def create_dashboard(self, output_html: str = "db_dashboard.html"):
        """
        Create interactive HTML dashboard with Plotly.

        Args:
            output_html: Output file path for HTML dashboard
        """
        # Get data
        table_sizes = self.get_table_sizes()
        connection_stats = self.get_connection_stats()
        index_usage = self.get_index_usage()
        cache_hit_ratio = self.get_cache_hit_ratio()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Table Sizes', 'Connection States',
                            'Index Usage', 'Cache Hit Ratio'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]]
        )

        # Plot 1: Table sizes
        fig.add_trace(
            go.Bar(
                x=table_sizes['tablename'].head(10),
                y=table_sizes['total_size_bytes'].head(10) / (1024**3),
                name='Table Size (GB)'
            ),
            row=1, col=1
        )

        # Plot 2: Connection states
        fig.add_trace(
            go.Pie(
                labels=connection_stats['state'],
                values=connection_stats['connection_count'],
                name='Connections'
            ),
            row=1, col=2
        )

        # Plot 3: Index usage
        fig.add_trace(
            go.Scatter(
                x=index_usage['index_scans'],
                y=index_usage['tuples_fetched'],
                mode='markers',
                name='Index Usage',
                text=index_usage['indexname']
            ),
            row=2, col=1
        )

        # Plot 4: Cache hit ratio
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cache_hit_ratio,
                title={'text': "Cache Hit %"},
                gauge={'axis': {'range': [None, 100]},
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75,
                           'value': 90}}
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Database Health Dashboard",
            showlegend=False,
            height=800
        )

        # Save
        fig.write_html(output_html)
        print(f"Dashboard saved to {output_html}")

# Usage

if __name__ == "__main__":
    monitor = DatabaseMonitor("postgresql://user:pass@localhost/mldb")

    # Generate report
    report = monitor.generate_health_report()

    print("\n=== Database Health Report ===\n")
    print(f"Cache Hit Ratio: {report['cache_hit_ratio']:.2f}%")
    print(f"Total Tables: {len(report['table_sizes'])}")
    print(f"Unused Indexes: {len(report['unused_indexes'])}")
    print(f"Long Running Queries: {len(report['long_running_queries'])}")

    # Create dashboard
    monitor.create_dashboard("dashboard.html")
```

**Key monitoring queries:**
1. Table sizes - Track growth over time
2. Query performance - Find slow queries
3. Index usage - Identify unused indexes
4. Connection pool - Monitor active/idle connections
5. Cache hit ratio - Memory utilization
6. Bloat estimate - Need for VACUUM

This provides comprehensive database health monitoring for ML systems.

---

## Debugging & Troubleshooting (Q21-Q25)

### Q21: Your Parquet file reads are slow despite having only 10 columns out of 100. What could be wrong? How do you diagnose and fix it?

**Answer:**

**Possible issues:**

**1. Not using column pruning:**
```python
# Bad: reads all columns internally
df = pd.read_parquet('data.parquet')
df = df[['col1', 'col2']]

# Good: only reads specified columns
df = pd.read_parquet('data.parquet', columns=['col1', 'col2'])
```

**2. Wrong compression algorithm:**
```python
# Check compression
import pyarrow.parquet as pq

metadata = pq.read_metadata('data.parquet')
print(f"Compression: {metadata.row_group(0).column(0).compression}")

# If it's GZIP (slow), convert to Snappy
df = pd.read_parquet('data.parquet')
df.to_parquet('data_snappy.parquet', compression='snappy')
```

**3. Large row groups:**
```python
# Check row group size
metadata = pq.read_metadata('data.parquet')
print(f"Row groups: {metadata.num_row_groups}")
print(f"Rows per group: {metadata.row_group(0).num_rows}")

# Rewrite with smaller row groups (faster parallel reads)
pq.write_table(
    pa.Table.from_pandas(df),
    'data_optimized.parquet',
    row_group_size=100_000  # Adjust based on data
)
```

**4. No partitioning for filtered queries:**
```python
# If you frequently filter by date, partition by it
df.to_parquet(
    'data_partitioned/',
    partition_cols=['year', 'month'],
    compression='snappy'
)

# Now filtered queries only read relevant partitions
df = pd.read_parquet(
    'data_partitioned/',
    filters=[('year', '==', 2024), ('month', '==', 1)]
)
```

**Diagnostic script:**
```python
import pyarrow.parquet as pq
import time

def diagnose_parquet(file_path):
    metadata = pq.read_metadata(file_path)

    print(f"File: {file_path}")
    print(f"Rows: {metadata.num_rows:,}")
    print(f"Row groups: {metadata.num_row_groups}")
    print(f"Columns: {metadata.num_columns}")
    print(f"File size: {os.path.getsize(file_path) / (1024**2):.2f} MB")
    print(f"Compression: {metadata.row_group(0).column(0).compression}")

    # Test read speeds
    start = time.time()
    df_all = pd.read_parquet(file_path)
    time_all = time.time() - start

    start = time.time()
    df_subset = pd.read_parquet(file_path, columns=['col1', 'col2'])
    time_subset = time.time() - start

    print(f"Read all columns: {time_all:.2f}s")
    print(f"Read 2 columns: {time_subset:.2f}s")
    print(f"Speedup: {time_all/time_subset:.2f}x")

diagnose_parquet('data.parquet')
```

---

### Q22: A SQL query with JOINs is taking 30 seconds. Walk through your debugging process with EXPLAIN ANALYZE.

**Answer:**

**Step 1: Get baseline with EXPLAIN ANALYZE**
```sql
EXPLAIN ANALYZE
SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01'
GROUP BY u.id, u.name
ORDER BY total_spent DESC
LIMIT 10;
```

**Output example:**
```
Limit  (cost=150000..150010 rows=10) (actual time=29843..29845 rows=10)
  ->  Sort  (cost=150000..152000 rows=100000) (actual time=29843..29844)
        Sort Key: (sum(o.amount)) DESC
        ->  Hash Left Join  (cost=50000..120000 rows=100000) (actual time=5000..28000)
              Hash Cond: (u.id = o.user_id)
              ->  Seq Scan on users u  (cost=0..10000 rows=100000) (actual time=0..100)
              ->  Hash  (cost=30000..30000 rows=500000) (actual time=4900..4900)
                    ->  Seq Scan on orders o  (cost=0..30000 rows=500000) (actual time=0..4000)
                          Filter: (created_at > '2024-01-01')
                          Rows Removed by Filter: 2000000
Planning Time: 2.3 ms
Execution Time: 29845.8 ms
```

**Problems identified:**
1. ❌ Seq Scan on orders (scanning 2.5M rows, filtering 2M)
2. ❌ No index on created_at
3. ❌ Expensive sort on aggregated data

**Step 2: Add indexes**
```sql
-- Index on filter column
CREATE INDEX idx_orders_created ON orders(created_at);

-- Composite index for JOIN + filter
CREATE INDEX idx_orders_user_created ON orders(user_id, created_at);
```

**Step 3: Rewrite query**
```sql
-- Optimize: filter before JOIN
EXPLAIN ANALYZE
SELECT u.name, stats.order_count, stats.total_spent
FROM users u
LEFT JOIN (
  SELECT user_id,
         COUNT(*) as order_count,
         SUM(amount) as total_spent
  FROM orders
  WHERE created_at > '2024-01-01'
  GROUP BY user_id
) stats ON u.id = stats.user_id
ORDER BY stats.total_spent DESC NULLS LAST
LIMIT 10;
```

**New output:**
```
Limit  (cost=5000..5010 rows=10) (actual time=125..127 rows=10)
  ->  Sort  (cost=5000..5500 rows=50000) (actual time=125..126)
        Sort Key: (stats.total_spent) DESC NULLS LAST
        ->  Hash Left Join  (cost=3000..4500 rows=100000) (actual time=50..120)
              Hash Cond: (u.id = stats.user_id)
              ->  Seq Scan on users u  (cost=0..10000 rows=100000) (actual time=0..10)
              ->  Hash  (cost=2500..2500 rows=50000) (actual time=48..48)
                    ->  Subquery Scan on stats  (cost=2000..2500 rows=50000) (actual time=10..45)
                          ->  HashAggregate  (cost=2000..2250 rows=50000) (actual time=10..40)
                                ->  Index Scan on orders  (cost=0..1500 rows=500000) (actual time=0..8)
                                      Index Cond: (created_at > '2024-01-01')
Planning Time: 1.2 ms
Execution Time: 127.3 ms
```

✅ **Result: 230x faster (29.8s → 0.127s)**

**Key optimizations:**
1. Index on filter column (orders.created_at)
2. Aggregation before JOIN (reduces rows)
3. Index scan instead of seq scan

---

### Q23: Your application suddenly starts getting "too many connections" errors from PostgreSQL. How do you diagnose the root cause and fix it?

**Answer:**

**Step 1: Check current connections**

```sql
-- Count connections by state
SELECT
    state,
    application_name,
    COUNT(*) as count,
    MAX(EXTRACT(EPOCH FROM (NOW() - state_change))) as max_idle_seconds
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY state, application_name
ORDER BY count DESC;
```

**Output example:**
```
      state       | application_name | count | max_idle_seconds
------------------+------------------+-------+------------------
 idle             | myapp            |   45  |      1234
 active           | myapp            |   12  |        10
 idle in transaction | myapp        |   38  |      5678  ← PROBLEM!
```

**Problem identified:** "idle in transaction" connections are holding connections without releasing them.

---

**Step 2: Find connections by user/host**

```sql
SELECT
    usename,
    client_addr,
    COUNT(*) as connection_count,
    SUM(CASE WHEN state = 'idle in transaction' THEN 1 ELSE 0 END) as idle_in_transaction,
    SUM(CASE WHEN state = 'active' THEN 1 ELSE 0 END) as active,
    SUM(CASE WHEN state = 'idle' THEN 1 ELSE 0 END) as idle
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY usename, client_addr
ORDER BY connection_count DESC;
```

---

**Step 3: Find long-running idle transactions (connection leaks)**

```sql
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    state_change,
    EXTRACT(EPOCH FROM (NOW() - state_change)) as idle_seconds,
    LEFT(query, 100) as last_query
FROM pg_stat_activity
WHERE datname = current_database()
  AND state = 'idle in transaction'
  AND state_change < NOW() - INTERVAL '5 minutes'
ORDER BY state_change;
```

---

**Step 4: Check max_connections setting**

```sql
SHOW max_connections;
-- Result: 100 (default)

-- Check current usage
SELECT
    COUNT(*) as current_connections,
    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections,
    ROUND(100.0 * COUNT(*) / (SELECT setting::int FROM pg_settings WHERE name = 'max_connections'), 2) as usage_pct
FROM pg_stat_activity;
```

---

**Common causes and fixes:**

**1. Connection leaks in application code**

```python
# ❌ BAD: Connection never returned
import psycopg2

def bad_query():
    conn = psycopg2.connect("postgresql://...")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    results = cur.fetchall()
    return results
    # Connection never closed! Leak!

# ✅ GOOD: Use context manager
def good_query():
    with psycopg2.connect("postgresql://...") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users")
            results = cur.fetchall()
    return results
    # Connection automatically closed
```

**2. Not using connection pooling**

```python
# ❌ BAD: New connection every time
from sqlalchemy import create_engine

engine = create_engine("postgresql://...", pool_size=1, max_overflow=0)  # No pooling!

# ✅ GOOD: Proper pooling
engine = create_engine(
    "postgresql://...",
    pool_size=20,        # Core pool size
    max_overflow=30,     # Additional connections
    pool_timeout=30,     # Wait for connection
    pool_recycle=3600,   # Recycle after 1 hour
    pool_pre_ping=True   # Test before use
)
```

**3. Transactions not committed/rolled back**

```python
# ❌ BAD: Transaction left open
conn = psycopg2.connect("postgresql://...")
cur = conn.cursor()
cur.execute("BEGIN")
cur.execute("UPDATE users SET active = true WHERE id = 1")
# Forgot to commit or rollback!
# Connection stuck in "idle in transaction"

# ✅ GOOD: Always commit or rollback
conn = psycopg2.connect("postgresql://...")
try:
    cur = conn.cursor()
    cur.execute("BEGIN")
    cur.execute("UPDATE users SET active = true WHERE id = 1")
    conn.commit()  # Explicit commit
except Exception as e:
    conn.rollback()  # Rollback on error
    raise
finally:
    conn.close()
```

**4. Long-running queries blocking connections**

```sql
-- Find and kill blocking queries
SELECT
    pid,
    usename,
    EXTRACT(EPOCH FROM (NOW() - query_start)) as duration_seconds,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND query_start < NOW() - INTERVAL '10 minutes'
ORDER BY query_start;

-- Kill specific query
SELECT pg_terminate_backend(12345);  -- Replace with actual PID
```

---

**Quick fixes:**

**1. Kill idle in transaction connections**

```sql
-- Kill all idle in transaction older than 5 minutes
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = current_database()
  AND state = 'idle in transaction'
  AND state_change < NOW() - INTERVAL '5 minutes';
```

**2. Set idle_in_transaction_session_timeout**

```sql
-- Automatically kill idle transactions after 30 seconds
ALTER DATABASE mydb SET idle_in_transaction_session_timeout = '30s';

-- Or in postgresql.conf
idle_in_transaction_session_timeout = 30000  # milliseconds
```

**3. Increase max_connections (temporary fix)**

```sql
-- Increase max connections (requires restart)
ALTER SYSTEM SET max_connections = 200;
-- Then restart PostgreSQL
```

⚠️ **Warning:** Increasing max_connections uses more memory. Fix the root cause instead!

---

**Permanent solution:**

```python
from sqlalchemy import create_engine, event, exc
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Create engine with proper pooling
engine = create_engine(
    "postgresql://user:pass@localhost/mydb",
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True
)

# Monitor pool usage
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    logger.debug(f"Connection created: {id(dbapi_conn)}")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    pool = engine.pool
    logger.debug(
        f"Pool status: size={pool.size()}, "
        f"checked_out={pool.checkedout()}, "
        f"overflow={pool.overflow()}"
    )

    # Alert if pool is exhausted
    if pool.checkedout() > pool.size() * 0.9:
        logger.warning("Connection pool nearly exhausted!")

# Always use context managers
@contextmanager
def get_connection():
    """Safe connection context manager."""
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with get_connection() as conn:
    result = conn.execute("SELECT * FROM users")
    # Connection automatically returned to pool
```

---

**Monitoring script:**

```python
import psycopg2
import time

def monitor_connections(db_url, alert_threshold=80):
    """Monitor connection usage and alert if high."""

    while True:
        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            # Get connection stats
            cur.execute("""
                SELECT
                    COUNT(*) as current,
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max,
                    SUM(CASE WHEN state = 'idle in transaction' THEN 1 ELSE 0 END) as idle_in_trans
                FROM pg_stat_activity
            """)

            current, max_conn, idle_in_trans = cur.fetchone()
            usage_pct = 100.0 * current / max_conn

            print(f"{time.ctime()}: {current}/{max_conn} ({usage_pct:.1f}%) | Idle in trans: {idle_in_trans}")

            if usage_pct > alert_threshold:
                print(f"⚠️  WARNING: Connection usage at {usage_pct:.1f}%")

            if idle_in_trans > 10:
                print(f"⚠️  WARNING: {idle_in_trans} idle in transaction connections")

            cur.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error: {e}")

        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    monitor_connections("postgresql://user:pass@localhost/mydb")
```

**Summary of fixes:**
1. ✅ Use connection pooling (SQLAlchemy, psycopg2.pool)
2. ✅ Always close connections (context managers)
3. ✅ Commit or rollback transactions explicitly
4. ✅ Set `idle_in_transaction_session_timeout`
5. ✅ Monitor pool usage
6. ✅ Find and fix connection leaks in code

---

### Q24: A pandas read_sql query returns data but columns are in wrong types (numbers as strings, dates as objects). How do you debug and fix this?

**Answer:**

**Common problem:**

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://...")

# Read data
df = pd.read_sql("SELECT * FROM users", engine)

# Check types
print(df.dtypes)
```

**Output (WRONG types):**
```
user_id      object  ← Should be int
age          object  ← Should be int
created_at   object  ← Should be datetime
is_active    object  ← Should be bool
amount       object  ← Should be float
```

---

**Diagnosis:**

**Check 1: Inspect actual data**

```python
print(df.head())
print(df['user_id'].iloc[0], type(df['user_id'].iloc[0]))
```

**Common issues:**
- Null values mixed with numbers → pandas defaults to object
- Leading zeros in numeric strings
- Mixed types in column
- Database type not supported by pandas

---

**Check 2: Inspect database schema**

```sql
-- Check column types in PostgreSQL
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'users'
ORDER BY ordinal_position;
```

---

**Solutions:**

**Solution 1: Explicit dtype specification**

```python
import pandas as pd
import numpy as np

# Specify dtypes when reading
df = pd.read_sql(
    "SELECT * FROM users",
    engine,
    dtype={
        'user_id': 'Int64',      # Nullable integer
        'age': 'Int64',
        'amount': 'float64',
        'is_active': 'boolean'
    }
)

# Parse dates explicitly
df = pd.read_sql(
    "SELECT * FROM users",
    engine,
    parse_dates=['created_at', 'updated_at']
)
```

---

**Solution 2: Post-processing type conversion**

```python
def fix_types(df):
    """Fix common type issues."""

    # Convert numeric columns
    numeric_cols = ['user_id', 'age', 'amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert datetime columns
    date_cols = ['created_at', 'updated_at', 'last_login']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert boolean columns
    bool_cols = ['is_active', 'is_verified']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('boolean')

    return df

# Usage
df = pd.read_sql("SELECT * FROM users", engine)
df = fix_types(df)
```

---

**Solution 3: SQL-level casting**

```python
# Cast types in SQL query
query = """
SELECT
    user_id::INTEGER,
    age::INTEGER,
    amount::NUMERIC(10,2) as amount,
    created_at::TIMESTAMP,
    is_active::BOOLEAN
FROM users
"""

df = pd.read_sql(query, engine)
```

---

**Solution 4: Handle NULL values properly**

```python
# Use nullable integer types (pandas 1.0+)
df = pd.read_sql(
    "SELECT * FROM users",
    engine,
    dtype={
        'user_id': 'Int64',     # Capital I = nullable integer
        'age': 'Int64',
        'amount': 'Float64'     # Capital F = nullable float
    }
)

# Check for nulls
print(df.isnull().sum())
```

---

**Solution 5: Create type mapping from schema**

```python
from sqlalchemy import inspect

def get_dtype_mapping(engine, table_name):
    """
    Automatically generate dtype mapping from database schema.
    """
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)

    dtype_map = {}

    for col in columns:
        col_name = col['name']
        col_type = str(col['type'])

        # Map SQL types to pandas types
        if 'INTEGER' in col_type or 'SERIAL' in col_type:
            dtype_map[col_name] = 'Int64'
        elif 'NUMERIC' in col_type or 'DECIMAL' in col_type or 'FLOAT' in col_type:
            dtype_map[col_name] = 'float64'
        elif 'BOOLEAN' in col_type:
            dtype_map[col_name] = 'boolean'
        elif 'VARCHAR' in col_type or 'TEXT' in col_type:
            dtype_map[col_name] = 'string'
        # Note: datetime types handled by parse_dates

    return dtype_map

# Usage
dtype_map = get_dtype_mapping(engine, 'users')
print(f"Generated dtype mapping: {dtype_map}")

df = pd.read_sql(
    "SELECT * FROM users",
    engine,
    dtype=dtype_map,
    parse_dates=['created_at', 'updated_at']
)
```

---

**Solution 6: Reusable data loader**

```python
from typing import Optional, List, Dict
import pandas as pd

class TypeSafeDataLoader:
    """Load data from SQL with proper type handling."""

    def __init__(self, engine):
        self.engine = engine

    def load_table(
        self,
        table_name: str,
        dtype_overrides: Optional[Dict[str, str]] = None,
        date_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load table with automatic type detection.

        Args:
            table_name: Name of table to load
            dtype_overrides: Optional dtype overrides
            date_columns: Optional list of date columns

        Returns:
            DataFrame with proper types
        """
        # Get schema-based types
        dtype_map = get_dtype_mapping(self.engine, table_name)

        # Apply overrides
        if dtype_overrides:
            dtype_map.update(dtype_overrides)

        # Detect date columns from schema if not provided
        if date_columns is None:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            date_columns = [
                col['name'] for col in columns
                if 'TIMESTAMP' in str(col['type']) or 'DATE' in str(col['type'])
            ]

        # Load data
        df = pd.read_sql_table(
            table_name,
            self.engine,
            dtype=dtype_map,
            parse_dates=date_columns
        )

        return df

# Usage
loader = TypeSafeDataLoader(engine)
df = loader.load_table('users')

print(df.dtypes)  # All types correct!
```

---

**Debugging checklist:**

```python
def diagnose_types(df):
    """Comprehensive type diagnosis."""

    print("=== DataFrame Type Diagnosis ===\n")

    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"  Dtype: {df[col].dtype}")
        print(f"  Null count: {df[col].isnull().sum()}")

        # Sample values
        sample = df[col].dropna().head(3).tolist()
        print(f"  Sample values: {sample}")

        # Check if object column contains numbers
        if df[col].dtype == 'object':
            try:
                numeric_test = pd.to_numeric(df[col], errors='coerce')
                if numeric_test.notna().sum() > len(df) * 0.5:
                    print(f"  ⚠️  WARNING: Object column contains mostly numbers")
            except:
                pass

            # Check if contains dates
            try:
                date_test = pd.to_datetime(df[col], errors='coerce')
                if date_test.notna().sum() > len(df) * 0.5:
                    print(f"  ⚠️  WARNING: Object column contains mostly dates")
            except:
                pass

# Usage
df = pd.read_sql("SELECT * FROM users", engine)
diagnose_types(df)
```

---

**Best practices:**

```python
# ✅ GOOD: Complete type-safe loading
def load_data_safe(engine, query):
    """Load data with comprehensive type handling."""

    # Step 1: Load with initial types
    df = pd.read_sql(
        query,
        engine,
        dtype={
            'user_id': 'Int64',
            'age': 'Int64',
            'amount': 'Float64',
            'is_active': 'boolean'
        },
        parse_dates=['created_at', 'updated_at']
    )

    # Step 2: Validate types
    assert df['user_id'].dtype == 'Int64', "user_id should be Int64"
    assert df['amount'].dtype == 'float64', "amount should be float64"
    assert pd.api.types.is_datetime64_any_dtype(df['created_at']), "created_at should be datetime"

    # Step 3: Handle edge cases
    # Remove leading/trailing whitespace from strings
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        df[col] = df[col].str.strip()

    return df

# Usage
df = load_data_safe(engine, "SELECT * FROM users")
```

**Summary:**
1. Use `dtype` parameter in `read_sql`
2. Use `parse_dates` for datetime columns
3. Use nullable types (Int64, Float64, boolean)
4. Generate dtype mapping from schema
5. Validate types after loading
6. Create reusable type-safe loader

---

### Q25: Your ETL pipeline writes Parquet files that work locally but fail when read in Spark on the cluster with "schema mismatch" errors. How do you debug and fix this?

**Answer:**

**Error example:**
```
pyspark.sql.utils.AnalysisException: Schema mismatch detected:
  Field 'amount' has different types: expected DecimalType(10,2), found DoubleType
```

---

**Diagnosis:**

**Step 1: Compare schemas**

```python
import pyarrow.parquet as pq
from pyspark.sql import SparkSession

# Read schema with PyArrow (local)
parquet_schema = pq.read_schema('output.parquet')
print("PyArrow schema:")
print(parquet_schema)

# Read schema with Spark (cluster)
spark = SparkSession.builder.getOrCreate()
spark_df = spark.read.parquet('output.parquet')
print("\nSpark schema:")
spark_df.printSchema()
```

**Output:**
```
PyArrow schema:
user_id: int64
amount: double
created_at: timestamp[us]

Spark schema:
root
 |-- user_id: long (nullable = true)
 |-- amount: decimal(10,2) (nullable = true)
 |-- created_at: timestamp (nullable = true)
```

Problem: `amount` type mismatch (double vs decimal)

---

**Common causes:**

**1. Pandas writes DoubleType, Spark expects DecimalType**

```python
# Problem: pandas uses float64 (becomes DoubleType in Spark)
import pandas as pd

df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'amount': [10.50, 20.75, 30.00]  # float64
})

df.to_parquet('output.parquet')

# Spark expects decimal for monetary values
# Solution: Use PyArrow with explicit schema
import pyarrow as pa

schema = pa.schema([
    ('user_id', pa.int64()),
    ('amount', pa.decimal128(10, 2)),  # Decimal(10,2)
    ('created_at', pa.timestamp('us'))
])

table = pa.Table.from_pandas(df, schema=schema)
pq.write_table(table, 'output.parquet')
```

---

**2. Timestamp precision mismatch**

```python
# Problem: Different timestamp units
# Pandas: timestamp[ns] (nanoseconds)
# Spark: timestamp[us] (microseconds)

# Solution: Standardize to microseconds
schema = pa.schema([
    ('created_at', pa.timestamp('us'))  # Microseconds
])

# Or convert in pandas
df['created_at'] = pd.to_datetime(df['created_at']).astype('datetime64[us]')
```

---

**3. Nullable vs non-nullable**

```python
# Problem: Schema nullability mismatch

# PyArrow allows nulls by default
schema = pa.schema([
    ('user_id', pa.int64(), False),  # NOT NULL
    ('amount', pa.decimal128(10, 2), True)  # NULLABLE
])
```

---

**4. Partitioned Parquet with inconsistent schemas**

```python
# Problem: Different partitions have different schemas
# Partition 1: amount as float
# Partition 2: amount as decimal

# Diagnosis: Check all partition schemas
from pathlib import Path

for parquet_file in Path('output_partitioned/').rglob('*.parquet'):
    schema = pq.read_schema(parquet_file)
    print(f"{parquet_file}: {schema}")
```

---

**Solutions:**

**Solution 1: Write with explicit PyArrow schema**

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def write_parquet_with_schema(df: pd.DataFrame, output_path: str):
    """
    Write Parquet with Spark-compatible schema.
    """

    # Define schema explicitly
    schema = pa.schema([
        ('user_id', pa.int64()),
        ('user_name', pa.string()),
        ('age', pa.int32()),
        ('amount', pa.decimal128(10, 2)),  # Decimal for money
        ('created_at', pa.timestamp('us')),  # Microseconds
        ('is_active', pa.bool_())
    ])

    # Convert DataFrame to PyArrow Table with schema
    table = pa.Table.from_pandas(df, schema=schema)

    # Write Parquet
    pq.write_table(
        table,
        output_path,
        compression='snappy',
        use_dictionary=True,
        write_statistics=True
    )

    print(f"✅ Written {len(df)} rows to {output_path}")
    print(f"Schema: {table.schema}")

# Usage
df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'user_name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'amount': [10.50, 20.75, 30.00],
    'created_at': pd.date_range('2024-01-01', periods=3),
    'is_active': [True, False, True]
})

write_parquet_with_schema(df, 'output.parquet')
```

---

**Solution 2: Generate schema from Spark DDL**

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType, DecimalType, TimestampType, BooleanType
import pyarrow as pa

# Define Spark schema
spark_schema = StructType([
    StructField("user_id", LongType(), False),
    StructField("user_name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("amount", DecimalType(10, 2), True),
    StructField("created_at", TimestampType(), True),
    StructField("is_active", BooleanType(), True)
])

def spark_to_pyarrow_schema(spark_schema: StructType) -> pa.Schema:
    """Convert Spark schema to PyArrow schema."""

    type_mapping = {
        'long': pa.int64(),
        'integer': pa.int32(),
        'string': pa.string(),
        'boolean': pa.bool_(),
        'timestamp': pa.timestamp('us'),
        'decimal': lambda precision, scale: pa.decimal128(precision, scale)
    }

    fields = []
    for field in spark_schema.fields:
        field_type = field.dataType.typeName()

        if field_type == 'decimal':
            pa_type = pa.decimal128(field.dataType.precision, field.dataType.scale)
        else:
            pa_type = type_mapping.get(field_type, pa.string())

        fields.append(pa.field(field.name, pa_type, nullable=field.nullable))

    return pa.schema(fields)

# Convert
pyarrow_schema = spark_to_pyarrow_schema(spark_schema)
print(pyarrow_schema)

# Write with compatible schema
table = pa.Table.from_pandas(df, schema=pyarrow_schema)
pq.write_table(table, 'output.parquet')
```

---

**Solution 3: Schema evolution handling**

```python
def merge_schemas(*schemas):
    """
    Merge multiple Parquet schemas safely.

    Handles schema evolution across partitions.
    """
    import pyarrow as pa

    # Start with first schema
    merged_fields = {field.name: field for field in schemas[0]}

    # Merge subsequent schemas
    for schema in schemas[1:]:
        for field in schema:
            if field.name not in merged_fields:
                # New field - add it
                merged_fields[field.name] = field
            else:
                # Field exists - check compatibility
                existing = merged_fields[field.name]

                # If types differ, promote to compatible type
                if existing.type != field.type:
                    # Promote to most general type
                    if pa.types.is_integer(existing.type) and pa.types.is_integer(field.type):
                        # Use larger integer type
                        merged_fields[field.name] = pa.field(
                            field.name,
                            pa.int64(),
                            nullable=True
                        )
                    elif pa.types.is_floating(existing.type) or pa.types.is_floating(field.type):
                        # Promote to float
                        merged_fields[field.name] = pa.field(
                            field.name,
                            pa.float64(),
                            nullable=True
                        )

    return pa.schema(list(merged_fields.values()))

# Usage: Read all partition schemas and merge
schemas = []
for file in Path('partitioned/').rglob('*.parquet'):
    schemas.append(pq.read_schema(file))

unified_schema = merge_schemas(*schemas)
print(f"Unified schema: {unified_schema}")

# Rewrite all partitions with unified schema
# (left as exercise)
```

---

**Solution 4: Validation before writing**

```python
def validate_and_write_parquet(
    df: pd.DataFrame,
    output_path: str,
    expected_schema: pa.Schema
):
    """
    Validate DataFrame matches expected schema before writing.
    """

    # Convert to PyArrow Table
    table = pa.Table.from_pandas(df)

    # Check schema compatibility
    for i, expected_field in enumerate(expected_schema):
        if i >= len(table.schema):
            raise ValueError(f"Missing column: {expected_field.name}")

        actual_field = table.schema[i]

        # Check name
        if actual_field.name != expected_field.name:
            raise ValueError(
                f"Column name mismatch at position {i}: "
                f"expected '{expected_field.name}', got '{actual_field.name}'"
            )

        # Check type (allow compatible types)
        if not pa.types.is_compatible(actual_field.type, expected_field.type):
            raise ValueError(
                f"Column '{actual_field.name}' type mismatch: "
                f"expected {expected_field.type}, got {actual_field.type}"
            )

    # Cast to expected schema
    table = table.cast(expected_schema)

    # Write
    pq.write_table(table, output_path, compression='snappy')
    print(f"✅ Validated and written {len(table)} rows")

# Define expected schema
expected_schema = pa.schema([
    ('user_id', pa.int64()),
    ('amount', pa.decimal128(10, 2))
])

# Write with validation
validate_and_write_parquet(df, 'output.parquet', expected_schema)
```

---

**Solution 5: Read and normalize existing files**

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Read with schema merging
df = spark.read \
    .option("mergeSchema", "true") \
    .parquet("partitioned/*")

# Or specify expected schema
from pyspark.sql.types import StructType, StructField, LongType, DecimalType

expected_schema = StructType([
    StructField("user_id", LongType(), False),
    StructField("amount", DecimalType(10, 2), True)
])

df = spark.read \
    .schema(expected_schema) \
    .parquet("partitioned/*")

# Rewrite with consistent schema
df.write \
    .mode("overwrite") \
    .parquet("partitioned_fixed/")
```

---

**Best practices:**

```python
# ✅ Complete ETL with schema management

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

class SchemaAwareParquetWriter:
    """Write Parquet files with schema consistency."""

    def __init__(self, schema: pa.Schema):
        self.schema = schema

    def write(self, df: pd.DataFrame, output_path: str):
        """Write DataFrame with schema enforcement."""

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)

        # Cast to target schema (with validation)
        try:
            table = table.cast(self.schema, safe=True)
        except pa.ArrowInvalid as e:
            raise ValueError(f"Schema casting failed: {e}")

        # Write
        pq.write_table(
            table,
            output_path,
            compression='snappy',
            version='2.6',  # Use Parquet 2.6 for best compatibility
            use_dictionary=True,
            write_statistics=True
        )

# Define schema once
SCHEMA = pa.schema([
    ('user_id', pa.int64()),
    ('amount', pa.decimal128(10, 2)),
    ('created_at', pa.timestamp('us'))
])

# Use everywhere
writer = SchemaAwareParquetWriter(SCHEMA)

# ETL step 1
df1 = extract_data_1()
writer.write(df1, 'partition1.parquet')

# ETL step 2
df2 = extract_data_2()
writer.write(df2, 'partition2.parquet')

# All partitions have consistent schema!
```

**Summary:**
1. Use explicit PyArrow schemas
2. Match Spark's decimal types for money
3. Use microsecond timestamps
4. Validate schemas before writing
5. Handle schema evolution carefully
6. Test with Spark before production

---

## Trade-offs & Decisions (Q26-Q30)

### Q26: Your team debates: store ML training data in Parquet files vs PostgreSQL. You have 50GB of tabular data, updated weekly, queried for training daily. What do you recommend and why?

**Answer:**

**Option A: Parquet Files**

**Pros:**
- ✅ Faster read for ML (columnar format)
- ✅ Better compression (2-5x)
- ✅ No database overhead
- ✅ Easy to version (Git LFS, DVC)
- ✅ Portable (S3, local, anywhere)

**Cons:**
- ❌ No ACID transactions
- ❌ Harder to update (immutable)
- ❌ No concurrent writes
- ❌ No relational queries

**Option B: PostgreSQL**

**Pros:**
- ✅ ACID transactions
- ✅ Easy updates/deletes
- ✅ Concurrent access
- ✅ Relational queries (JOINs)
- ✅ Access control

**Cons:**
- ❌ Slower for ML training
- ❌ Row-based storage (reads all columns)
- ❌ Connection overhead
- ❌ Scaling limitations

**Recommendation: Hybrid Approach**

```python
# ETL Pipeline
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime

# 1. Store raw/live data in PostgreSQL
engine = create_engine('postgresql://...')

# 2. Weekly: Export to Parquet for training
def export_training_data():
    query = """
    SELECT features.*, labels.label
    FROM features
    JOIN labels ON features.id = labels.id
    WHERE updated_at >= current_date - interval '1 week'
    """

    df = pd.read_sql(query, engine)

    # Write to Parquet with partitioning
    df['export_date'] = datetime.now().date()
    df.to_parquet(
        f's3://ml-data/training_data/',
        partition_cols=['export_date'],
        compression='snappy'
    )

    print(f"Exported {len(df)} rows to Parquet")

# 3. Training: Read from Parquet (fast)
def load_training_data():
    df = pd.read_parquet('s3://ml-data/training_data/',
                          filters=[('export_date', '==', '2024-02-01')])
    return df

# 4. Inference: Query PostgreSQL directly (real-time)
def get_features_for_inference(entity_ids):
    query = """
    SELECT * FROM features
    WHERE id IN :ids
    """
    return pd.read_sql(query, engine, params={'ids': tuple(entity_ids)})
```

**Architecture:**
```
PostgreSQL (source of truth)
    ↓ Weekly ETL
Parquet (training)  →  ML Training (fast)
    ↑
PostgreSQL (serving) →  Real-time inference
```

**Why hybrid:**
- PostgreSQL: operational data, updates, relationships
- Parquet: training data snapshots, fast reads, versioning
- Best of both worlds

---

### Q27: Compare Snappy vs Gzip vs Zstd compression for Parquet. When would you use each? Show benchmarks.

**Answer:**

```python
import pandas as pd
import numpy as np
import time
import os

# Generate test data (1M rows, 20 columns)
np.random.seed(42)
df = pd.DataFrame({
    f'col_{i}': np.random.randn(1_000_000)
    for i in range(20)
})
df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
df['id'] = range(len(df))

print(f"Data shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / (1024**2):.2f} MB")

# Benchmark compressions
results = []

for compression in ['snappy', 'gzip', 'zstd', None]:
    filename = f'test_{compression}.parquet'

    # Write
    start = time.time()
    df.to_parquet(filename, compression=compression, index=False)
    write_time = time.time() - start

    # Read all
    start = time.time()
    df_read = pd.read_parquet(filename)
    read_all_time = time.time() - start

    # Read subset of columns
    start = time.time()
    df_subset = pd.read_parquet(filename, columns=['col_0', 'col_1', 'category'])
    read_subset_time = time.time() - start

    # File size
    file_size_mb = os.path.getsize(filename) / (1024**2)

    results.append({
        'compression': compression or 'none',
        'write_time': write_time,
        'read_all_time': read_all_time,
        'read_subset_time': read_subset_time,
        'file_size_mb': file_size_mb,
        'compression_ratio': df.memory_usage().sum() / (1024**2) / file_size_mb
    })

    os.remove(filename)

# Display results
results_df = pd.DataFrame(results)
print("\nBenchmark Results:")
print(results_df.to_string(index=False))
```

**Typical output:**
```
Benchmark Results:
 compression  write_time  read_all_time  read_subset_time  file_size_mb  compression_ratio
      snappy        0.45           0.25              0.08         55.20               2.76
        gzip        3.20           1.80              0.65         28.40               5.37
        zstd        0.75           0.35              0.12         32.10               4.75
        none        0.25           0.20              0.06        152.50               1.00
```

**When to use each:**

| Compression | Use Case | Why |
|-------------|----------|-----|
| **Snappy** | Hot data, frequent reads | Fastest decompression, good enough compression |
| **Gzip** | Cold storage, archival | Best compression, slow reads acceptable |
| **Zstd** | Production default | Best balance of speed and compression |
| **None** | Temporary, local dev | Fastest write/read, but huge files |

**Recommendations:**
- **Development/testing:** None or Snappy (fast iteration)
- **Production serving:** Snappy (milliseconds matter)
- **Data warehouse:** Zstd (storage cost matters)
- **Long-term archival:** Gzip (maximum compression)
- **Streaming/real-time:** Snappy (low latency)

---

### Q28: When would you choose DuckDB over pandas for data processing? Give specific scenarios with code examples.

**Answer:**

**Use DuckDB when:**
1. Data doesn't fit in memory
2. Need SQL for complex queries
3. Need to query multiple files
4. Want better performance on aggregations

**Scenario 1: Larger-than-memory data**

```python
import duckdb
import pandas as pd

# pandas: Fails with OOM on 20GB CSV
try:
    df = pd.read_csv('huge_file.csv')  # Crash!
except MemoryError:
    print("pandas: Out of memory")

# DuckDB: Streams data, no memory issues
con = duckdb.connect()
result = con.execute("""
    SELECT category, AVG(price), COUNT(*)
    FROM read_csv_auto('huge_file.csv')
    GROUP BY category
""").df()
print(result)
```

**Scenario 2: Query multiple Parquet files**

```python
# pandas: Load all, then filter
import glob

dfs = []
for file in glob.glob('data/*.parquet'):
    df = pd.read_parquet(file)
    dfs.append(df[df['year'] == 2024])
result = pd.concat(dfs)

# DuckDB: Filter at source (faster)
result = con.execute("""
    SELECT *
    FROM read_parquet('data/*.parquet')
    WHERE year = 2024
""").df()
```

**Scenario 3: Complex SQL queries**

```python
# pandas: Multiple operations, intermediate DataFrames
df = pd.read_parquet('sales.parquet')
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
monthly = df.groupby(['region', 'month'])['amount'].agg(['sum', 'count'])
monthly = monthly.reset_index()
monthly['prev_month'] = monthly.groupby('region')['sum'].shift(1)
monthly['growth'] = (monthly['sum'] - monthly['prev_month']) / monthly['prev_month']
result = monthly[monthly['growth'] > 0.1]

# DuckDB: Single query (faster, more readable)
result = con.execute("""
    WITH monthly_sales AS (
      SELECT
        region,
        DATE_TRUNC('month', date) as month,
        SUM(amount) as total,
        COUNT(*) as orders
      FROM read_parquet('sales.parquet')
      GROUP BY region, month
    )
    SELECT *,
           LAG(total) OVER (PARTITION BY region ORDER BY month) as prev_month,
           (total - LAG(total) OVER (PARTITION BY region ORDER BY month)) /
           LAG(total) OVER (PARTITION BY region ORDER BY month) as growth
    FROM monthly_sales
    WHERE growth > 0.1
""").df()
```

**Scenario 4: Performance comparison**

```python
import time

# Create test data
df = pd.DataFrame({
    'id': range(10_000_000),
    'value': np.random.randn(10_000_000),
    'category': np.random.choice(list('ABCDE'), 10_000_000)
})
df.to_parquet('test.parquet')

# pandas aggregation
start = time.time()
result_pd = df.groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max', 'count']
})
pandas_time = time.time() - start

# DuckDB aggregation
start = time.time()
result_dk = con.execute("""
    SELECT category,
           AVG(value) as mean,
           STDDEV(value) as std,
           MIN(value) as min,
           MAX(value) as max,
           COUNT(*) as count
    FROM read_parquet('test.parquet')
    GROUP BY category
""").df()
duckdb_time = time.time() - start

print(f"pandas: {pandas_time:.2f}s")
print(f"DuckDB: {duckdb_time:.2f}s")
print(f"Speedup: {pandas_time/duckdb_time:.2f}x")
```

**Typical result:** DuckDB is 3-10x faster for aggregations

**Recommendation:**
- **Use pandas:** Small data (<1GB), need DataFrame operations, prototyping
- **Use DuckDB:** Large data, SQL queries, multiple files, aggregations
- **Best practice:** Start with pandas, switch to DuckDB when you hit limits

---

### Q29: You're designing a data lake for ML. Should you store data in: (1) Single large Parquet files, (2) Many small Parquet files, (3) Partitioned by date, (4) Partitioned by entity ID? Analyze trade-offs for different access patterns.

**Answer:**

Let's analyze each approach with real-world access patterns.

---

**Option 1: Single large Parquet files**

```python
# Structure:
# data/
#   └── transactions.parquet (10GB)

df = pd.read_parquet('data/transactions.parquet')
```

**Pros:**
- ✅ Simple structure
- ✅ No small file problem
- ✅ Good compression

**Cons:**
- ❌ Must read entire file (even for filtered queries)
- ❌ No parallelism in reads
- ❌ Updates require rewriting entire file
- ❌ Memory issues for large files

**Use case:** Small datasets (<1GB), infrequent access, read-only

---

**Option 2: Many small Parquet files**

```python
# Structure:
# data/
#   ├── transactions_000001.parquet (10MB)
#   ├── transactions_000002.parquet (10MB)
#   └── ... (1000 files)

df = pd.read_parquet('data/')  # Reads all files
```

**Pros:**
- ✅ Parallel reads
- ✅ Easy to append new data

**Cons:**
- ❌ "Small file problem" (metadata overhead)
- ❌ Slow to list files
- ❌ Poor compression (small files compress worse)
- ❌ Filtering still reads all files

**Performance:**

```python
import time
import pandas as pd

# Benchmark: Single large file vs many small files
# Same total data (1GB)

# Single file: 1GB
start = time.time()
df = pd.read_parquet('single/data.parquet', columns=['user_id', 'amount'])
print(f"Single file: {time.time() - start:.2f}s")

# Many small files: 100 x 10MB
start = time.time()
df = pd.read_parquet('many/', columns=['user_id', 'amount'])
print(f"Many files: {time.time() - start:.2f}s")
```

**Output:**
```
Single file: 2.1s
Many files: 5.8s  ← Slower due to file listing overhead
```

**Use case:** Streaming data ingestion, but not recommended for analytics

---

**Option 3: Partitioned by date (time-based)**

```python
# Structure:
# data/
#   ├── year=2024/
#   │   ├── month=01/
#   │   │   ├── day=01/
#   │   │   │   └── data.parquet
#   │   │   └── day=02/
#   │   │       └── data.parquet
#   │   └── month=02/
#   └── year=2023/

# Write with partitioning
df.to_parquet(
    'data/',
    partition_cols=['year', 'month', 'day'],
    compression='snappy'
)

# Read specific partition (fast!)
df = pd.read_parquet(
    'data/',
    filters=[('year', '==', 2024), ('month', '==', 1)]
)
```

**Pros:**
- ✅ **Fast time-range queries** (only reads relevant partitions)
- ✅ Parallel reads
- ✅ Easy to delete old data
- ✅ Incremental updates (append new partitions)

**Cons:**
- ❌ Poor for non-time queries (scans all partitions)
- ❌ Many partitions = many directories

**Performance:**

```python
# Query last 7 days of data

# Without partitioning
start = time.time()
df = pd.read_parquet('data/transactions.parquet')  # 10GB
df = df[df['date'] >= '2024-01-27']  # Filter in memory
print(f"No partitioning: {time.time() - start:.2f}s")

# With date partitioning
start = time.time()
df = pd.read_parquet(
    'data_partitioned/',
    filters=[('date', '>=', '2024-01-27')]  # Only reads 7 days
)
print(f"With partitioning: {time.time() - start:.2f}s")
```

**Output:**
```
No partitioning: 45.3s  (reads 10GB)
With partitioning: 1.2s  (reads 200MB)  ← 37x faster!
```

**Use case:** Time-series data, logs, events - anything with time-based queries

---

**Option 4: Partitioned by entity ID (hash-based)**

```python
# Structure:
# data/
#   ├── user_id_bucket=0/
#   │   └── data.parquet
#   ├── user_id_bucket=1/
#   │   └── data.parquet
#   └── ... (100 buckets)

# Create hash bucket
df['user_id_bucket'] = df['user_id'] % 100

# Write with partitioning
df.to_parquet(
    'data/',
    partition_cols=['user_id_bucket'],
    compression='snappy'
)

# Read specific user's data (fast!)
user_id = 12345
bucket = user_id % 100
df = pd.read_parquet(
    'data/',
    filters=[('user_id_bucket', '==', bucket)]
)
df = df[df['user_id'] == user_id]
```

**Pros:**
- ✅ **Fast entity-specific queries** (only reads 1 partition)
- ✅ Balanced partition sizes
- ✅ Good for user-level analytics

**Cons:**
- ❌ Poor for time-range queries (scans all buckets)
- ❌ Requires knowing bucket function

**Performance:**

```python
# Query single user's data

# Without partitioning
start = time.time()
df = pd.read_parquet('data/transactions.parquet')  # 10GB
df = df[df['user_id'] == 12345]  # Filter in memory
print(f"No partitioning: {time.time() - start:.2f}s")

# With entity ID partitioning (100 buckets)
start = time.time()
bucket = 12345 % 100
df = pd.read_parquet(
    'data_partitioned/',
    filters=[('user_id_bucket', '==', bucket)]  # Only reads 100MB
)
df = df[df['user_id'] == 12345]
print(f"With partitioning: {time.time() - start:.2f}s")
```

**Output:**
```
No partitioning: 42.1s  (reads 10GB)
With partitioning: 0.8s  (reads 100MB)  ← 52x faster!
```

**Use case:** User-level features, personalization, per-entity analytics

---

**Hybrid: Multi-level partitioning**

```python
# Structure:
# data/
#   ├── year=2024/
#   │   ├── month=01/
#   │   │   ├── user_bucket=0/
#   │   │   │   └── data.parquet
#   │   │   └── user_bucket=1/
#   │   │       └── data.parquet
#   │   └── month=02/
#   └── year=2023/

# Write with multi-level partitioning
df.to_parquet(
    'data/',
    partition_cols=['year', 'month', 'user_bucket'],
    compression='snappy'
)

# Query: User 12345, last 30 days
df = pd.read_parquet(
    'data/',
    filters=[
        ('year', '==', 2024),
        ('month', 'in', [1, 2]),
        ('user_bucket', '==', 45)
    ]
)
df = df[df['user_id'] == 12345]
```

**Pros:**
- ✅ Fast for both time-range AND entity queries
- ✅ Maximum flexibility

**Cons:**
- ❌ Complex directory structure
- ❌ Many small files if too many partitions

---

**Decision matrix:**

| Access Pattern | Best Partitioning Strategy | Speedup |
|----------------|----------------------------|---------|
| Time-range queries (last N days) | Date partitioning | 30-100x |
| Entity-specific queries (user_id) | Entity ID partitioning | 50-100x |
| Both time + entity | Multi-level (date + entity) | 100-1000x |
| Full table scans | Single large file | 1x |
| Random sampling | No partitioning | 1x |

---

**File size guidelines:**

```python
# Optimal file size: 128MB - 1GB

# Too small (<10MB)
# - Metadata overhead
# - Slow to list
# - Poor compression

# Too large (>5GB)
# - Can't fit in memory
# - No parallelism
# - Long read times

# Optimal target: ~256MB per file
```

**How to achieve optimal file size:**

```python
def write_optimized_parquet(df, output_path, target_size_mb=256):
    """
    Write Parquet with optimal file sizes.

    Args:
        df: DataFrame to write
        output_path: Output directory
        target_size_mb: Target file size in MB
    """
    # Estimate memory size
    memory_size_mb = df.memory_usage(deep=True).sum() / (1024**2)

    # Calculate number of files needed
    num_files = max(1, int(memory_size_mb / target_size_mb))

    # Add file partition column
    df['file_partition'] = pd.cut(
        range(len(df)),
        bins=num_files,
        labels=False
    )

    # Write with file partitioning
    df.to_parquet(
        output_path,
        partition_cols=['file_partition'],
        compression='snappy'
    )

    # Remove partition column metadata
    # (so users don't see it)

    print(f"Written {num_files} files (~{target_size_mb}MB each)")

# Usage
df = pd.DataFrame(...)  # 5GB of data
write_optimized_parquet(df, 'data/', target_size_mb=256)
# Creates ~20 files of 256MB each
```

---

**Real-world recommendation:**

**For ML data lake:**

```python
# Recommended structure:
# data/
#   └── {dataset_name}/
#       └── year={year}/
#           └── month={month}/
#               ├── part-0000.parquet (128-512MB)
#               ├── part-0001.parquet
#               └── ...

# Example: ML training data
# data/
#   └── user_features/
#       └── year=2024/
#           └── month=01/
#               ├── part-0000.parquet (256MB)
#               ├── part-0001.parquet (256MB)
#               └── ... (40 files total = 10GB)

# Write
df.to_parquet(
    'data/user_features/',
    partition_cols=['year', 'month'],
    compression='snappy',
    row_group_size=100_000  # Balance between read speed and file size
)

# Read: Last month of data for training
train_df = pd.read_parquet(
    'data/user_features/',
    filters=[('year', '==', 2024), ('month', '==', 1)]
)
```

**Why this works:**
- Date partitioning for incremental updates
- 256MB files for optimal I/O
- Snappy compression (fast, good compression)
- Row groups ~100k rows (good for column pruning)

---

**Partition explosion warning:**

```python
# ❌ BAD: Too many partitions
df.to_parquet(
    'data/',
    partition_cols=['year', 'month', 'day', 'hour', 'user_id']
)
# Creates 365 * 24 * 1M users = billions of tiny files!

# ✅ GOOD: Reasonable partitions
df['user_bucket'] = df['user_id'] % 100  # Only 100 buckets
df.to_parquet(
    'data/',
    partition_cols=['year', 'month', 'user_bucket']
)
# Creates 12 * 100 = 1,200 directories (manageable)
```

**Rule of thumb:** Keep total partitions under 10,000

---

**Summary:**

| Strategy | Best For | File Size | Partitions |
|----------|----------|-----------|------------|
| Single file | Small data (<1GB) | 1GB | 1 |
| Many small files | Streaming ingestion | <10MB | 1000s |
| Date partitioning | Time-series analytics | 256MB | 100-1000 |
| Entity partitioning | Per-user analytics | 256MB | 100-1000 |
| Multi-level | Both time + entity | 256MB | 1000-10000 |

**My recommendation for ML data lake:**
- **Primary:** Date partitioning (year/month)
- **File size:** 128-512MB per file
- **Compression:** Snappy
- **Add entity bucket if needed** (but keep partitions < 10k)

---

### Q30: Design a data pipeline for storing 100M ticket embeddings with metadata. Compare: (1) PostgreSQL with pgvector, (2) Separate vector DB (Qdrant) + PostgreSQL, (3) Parquet + FAISS. Which would you choose and why?

**Answer:**

**Option 1: PostgreSQL with pgvector**

```sql
CREATE EXTENSION vector;

CREATE TABLE tickets (
    id SERIAL PRIMARY KEY,
    title TEXT,
    description TEXT,
    category VARCHAR(50),
    created_at TIMESTAMP,
    embedding vector(768)  -- sentence-transformer size
);

CREATE INDEX tickets_embedding_idx ON tickets
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 1000);

-- Query
SELECT id, title, 1 - (embedding <=> query_embedding) as similarity
FROM tickets
WHERE category = 'billing'
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

**Pros:**
- ✅ Single database (simple)
- ✅ ACID transactions
- ✅ Can filter by metadata + vector search

**Cons:**
- ❌ Slower than specialized vector DBs
- ❌ Limited to ~10M vectors (performance degrades)
- ❌ Less sophisticated indexing (no HNSW)

**Option 2: Qdrant + PostgreSQL**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import psycopg2

# Qdrant: vectors
qdrant = QdrantClient("localhost", port=6333)
qdrant.create_collection(
    collection_name="tickets",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# PostgreSQL: metadata
conn = psycopg2.connect("postgresql://...")
cur = conn.cursor()
cur.execute("""
    CREATE TABLE tickets (
        id SERIAL PRIMARY KEY,
        title TEXT,
        category VARCHAR(50),
        created_at TIMESTAMP
    )
""")

# Insert
qdrant.upsert(
    collection_name="tickets",
    points=[{"id": 1, "vector": embedding, "payload": {"ticket_id": 1}}]
)

# Query: vector search in Qdrant, then enrich from PostgreSQL
results = qdrant.search(
    collection_name="tickets",
    query_vector=query_embedding,
    limit=100,
    query_filter={"must": [{"key": "category", "match": {"value": "billing"}}]}
)

ticket_ids = [hit.payload["ticket_id"] for hit in results]
cur.execute("SELECT * FROM tickets WHERE id IN %s", (tuple(ticket_ids),))
metadata = cur.fetchall()
```

**Pros:**
- ✅ Fast vector search (100M+ vectors)
- ✅ Advanced indexing (HNSW)
- ✅ Scalable
- ✅ Metadata filters in vector search

**Cons:**
- ❌ Two systems to maintain
- ❌ Eventual consistency
- ❌ Complex deployment

**Option 3: Parquet + FAISS**

```python
import pandas as pd
import faiss
import numpy as np

# Store metadata in Parquet (partitioned)
df = pd.DataFrame({
    'id': ticket_ids,
    'title': titles,
    'category': categories
})
df.to_parquet('tickets/', partition_cols=['category'])

# Store vectors in FAISS
embeddings = np.array(embeddings).astype('float32')
index = faiss.IndexHNSWFlat(768, 32)  # HNSW for fast search
index.add(embeddings)
faiss.write_index(index, 'tickets.index')

# Query
D, I = index.search(query_embedding, k=100)
matching_ids = I[0]
results = pd.read_parquet('tickets/', filters=[('id', 'in', matching_ids)])
```

**Pros:**
- ✅ Extremely fast (in-memory FAISS)
- ✅ No database overhead
- ✅ Easy to version control
- ✅ Cheapest (no DB hosting)

**Cons:**
- ❌ No ACID/concurrency
- ❌ Updates require rebuild
- ❌ Limited filtering (must post-process)

**Recommendation Matrix:**

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| <10M vectors, frequent updates | PostgreSQL + pgvector | Simple, transactional |
| 10M-1B vectors, production | Qdrant + PostgreSQL | Scalable, fast, filtering |
| Static data, batch processing | Parquet + FAISS | Fastest, cheapest |
| Real-time updates critical | Qdrant + PostgreSQL | Low-latency updates |
| Budget/simplicity critical | Parquet + FAISS | No hosting costs |

**For 100M tickets, I recommend:**

**Option 2 (Qdrant + PostgreSQL)** because:
1. Scales to 100M+ vectors efficiently
2. Fast search with HNSW (<50ms)
3. Metadata filtering in vector search
4. Real-time updates
5. Production-grade reliability

**Implementation architecture:**
```
API Request
    ↓
Qdrant (vector search with filters) → Top 100 results
    ↓
PostgreSQL (enrich metadata) → Full ticket details
    ↓
Response
```

This gives you sub-100ms queries on 100M vectors with full metadata support.
