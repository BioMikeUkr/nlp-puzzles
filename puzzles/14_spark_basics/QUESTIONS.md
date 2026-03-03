# Module 12: Spark Basics — Interview Questions

## Architecture & Design (Q1–Q10)

### Q1: What's the difference between transformations and actions in Spark?

**Transformations** are lazy — they build a DAG (directed acyclic graph) but don't execute: `select`, `filter`, `groupBy`, `join`, `withColumn`, `union`.

**Actions** trigger execution: `collect()`, `count()`, `show()`, `write`, `take()`, `first()`.

Spark chains transformations into a logical plan and optimizes the entire pipeline before executing. This means `.filter().select()` and `.select().filter()` may produce the same physical plan — the Catalyst optimizer reorders operations for efficiency.

Why it matters: if you call `df.filter(...)` nothing happens until you call `.count()` or `.show()`. This enables Spark to optimize across the entire chain.

---

### Q2: Explain narrow vs wide transformations.

**Narrow** — each input partition produces exactly one output partition, no data movement:
- `select`, `filter`, `withColumn`, `map`, `flatMap`, `union`

**Wide** — data must be shuffled across partitions:
- `groupBy`, `join`, `orderBy`, `distinct`, `repartition`, `reduceByKey`

Wide transformations create **stage boundaries**. Within a stage, Spark pipelines narrow transformations together efficiently. At stage boundaries, data is written to disk (shuffle write) and read by the next stage (shuffle read).

Key insight: minimize the number of wide transformations in your pipeline. Two `groupBy` calls = two shuffles = two stage boundaries.

---

### Q3: What is shuffle and why is it the most expensive operation?

Shuffle redistributes data across executors so that rows with the same key end up on the same partition. It happens during `groupBy`, `join`, `orderBy`, `distinct`.

Why it's expensive:
1. **Serialization** — data must be serialized to bytes
2. **Disk I/O** — shuffle data is written to local disk (shuffle write)
3. **Network I/O** — data is sent across the network to other executors
4. **Deserialization** — receiving executor deserializes the data
5. **Disk I/O again** — receiving side reads from local shuffle files

Mitigation strategies:
- **Broadcast joins** for small tables (avoids shuffle entirely)
- **Pre-partition data** by the join/group key
- **Reduce data before shuffle** (filter early, select only needed columns)
- Tune `spark.sql.shuffle.partitions` (default 200 is often wrong)

---

### Q4: How does the Catalyst optimizer work?

Catalyst is Spark SQL's query optimizer. It transforms your logical plan through 4 stages:

1. **Analysis** — resolves column/table references, checks types
2. **Logical Optimization** — applies rule-based optimizations:
   - **Predicate pushdown** — push filters closer to the data source
   - **Projection pruning** — only read needed columns
   - **Constant folding** — pre-compute constant expressions
   - **Filter reordering** — most selective filters first
3. **Physical Planning** — generates physical plans, picks the best:
   - Choose between BroadcastHashJoin vs SortMergeJoin
   - Decide on partition counts
4. **Code Generation (Tungsten)** — generates optimized Java bytecode, avoiding virtual function calls

`df.explain(True)` shows all 4 plan stages. This is your primary debugging tool for performance.

---

### Q5: What are the different join strategies in Spark?

**Broadcast Hash Join (BHJ)**:
- Small table (< 10MB default) is broadcast to all executors
- No shuffle required — fastest join type
- Force with `broadcast(small_df)`
- Controlled by `spark.sql.autoBroadcastJoinThreshold`

**Sort Merge Join (SMJ)**:
- Both sides shuffled by join key, then sorted and merged
- Default for large-large joins
- Efficient for sorted/pre-partitioned data

**Shuffle Hash Join (SHJ)**:
- Smaller side shuffled and built into hash table
- Used when one side is much smaller but too big to broadcast
- Enable with `spark.sql.join.preferSortMergeJoin=false`

**Cartesian Product / BroadcastNestedLoopJoin**:
- Fallback for non-equi joins or cross joins
- O(n*m) — avoid if possible

Check your join strategy: `df.explain()` — look for `BroadcastHashJoin`, `SortMergeJoin`, etc.

---

### Q6: Explain Spark's execution model: Driver, Executors, Jobs, Stages, Tasks.

**Driver** — the coordinator process:
- Creates SparkContext/SparkSession
- Builds the DAG of transformations
- Schedules tasks across executors
- Collects results from actions

**Executor** — worker process on each node:
- Executes tasks assigned by the driver
- Stores cached data in memory/disk
- Reports status back to the driver

**Execution hierarchy**:
- **Application** → has one Driver
- **Job** → triggered by each action (count, collect, write)
- **Stage** → separated by shuffle boundaries (wide transformations)
- **Task** — one task per partition per stage, smallest unit of work

Example: `df.filter(...).groupBy(...).count()` creates:
- 1 Job (triggered by `count()`)
- 2 Stages (separated by the shuffle from `groupBy`)
- N tasks per stage (where N = number of partitions)

---

### Q7: `repartition` vs `coalesce` — when to use each?

**`repartition(n)`**:
- Full shuffle — redistributes data evenly across n partitions
- Can increase or decrease partition count
- Use when you need even distribution (before a `groupBy` or `join`)
- `repartition("col")` — partition by column value (good before write with `partitionBy`)

**`coalesce(n)`**:
- No shuffle — just merges adjacent partitions
- Can only decrease partition count
- Faster than repartition but may create uneven partitions
- Use to reduce partitions before writing (e.g., 200 → 4 output files)

Rule of thumb: use `coalesce` to decrease, `repartition` to increase or when you need even distribution.

```python
# After a filter that removed 90% of data — reduce partitions
df.filter(F.col("status") == "active").coalesce(10)

# Before writing — ensure even output files
df.repartition(4).write.parquet("output/")

# Partition by column for efficient reads
df.repartition("country").write.partitionBy("country").parquet("output/")
```

---

### Q8: What is AQE (Adaptive Query Execution)?

AQE (Spark 3.0+) optimizes the query plan at runtime, after seeing actual data statistics from completed stages. Key features:

1. **Coalescing shuffle partitions** — merges small partitions after shuffle (fixes the "too many small partitions" problem)
2. **Switching join strategies** — switches to broadcast join if one side turns out to be small
3. **Skew join optimization** — splits skewed partitions into smaller ones

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")  # default true in Spark 3.2+
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

AQE largely solves the problem of choosing the right `shuffle.partitions` value.

---

### Q9: How do you handle data skew in Spark?

Data skew = one partition has significantly more data than others. Symptoms: one task takes hours while others finish in seconds.

**Detection**:
- Spark UI → Stages → check task duration distribution
- Check key cardinality: `df.groupBy("key").count().orderBy(F.desc("count")).show()`

**Solutions**:

1. **Salting** — add random prefix to skewed key, aggregate twice:
```python
df = df.withColumn("salted_key", F.concat(F.col("key"), F.lit("_"), (F.rand() * 10).cast("int")))
step1 = df.groupBy("salted_key").agg(F.sum("value").alias("partial_sum"))
step2 = step1.withColumn("key", F.split("salted_key", "_")[0])
result = step2.groupBy("key").agg(F.sum("partial_sum").alias("total"))
```

2. **Broadcast join** — if skew is in the join, broadcast the smaller side
3. **AQE skew join** — enable `spark.sql.adaptive.skewJoin.enabled`
4. **Filter out nulls** — null keys often cause skew; handle separately
5. **Repartition by different key** — if possible

---

### Q10: DataFrame API vs Spark SQL — when to use which?

**Performance**: Identical. Both go through Catalyst optimizer → same physical plan.

**DataFrame API advantages**:
- IDE autocomplete and type checking
- Programmatic query building (loops, conditionals)
- Easier to compose and chain
- Better for library/framework code

**Spark SQL advantages**:
- Familiar for SQL users
- Better for complex CTEs, subqueries, window functions
- Easier to read for ad-hoc analysis
- Can be stored as strings (config-driven queries)

**Real-world pattern**: use DataFrame API for ETL pipelines and library code, SQL for ad-hoc analysis and complex analytical queries. Many teams mix both.

---

## Implementation & Coding (Q11–Q20)

### Q11: How do window functions work? When to use them instead of groupBy?

`groupBy` collapses rows — you lose individual row data. Window functions compute aggregates while keeping all rows.

Use window functions when you need:
- **Ranking** within groups (top-N per group)
- **Running totals** / moving averages
- **Comparing row to group aggregate** (salary vs dept average)
- **lag/lead** — access previous/next row values

```python
from pyspark.sql import Window

# Rank employees within department
w = Window.partitionBy("dept").orderBy(F.desc("salary"))
df.withColumn("rank", F.rank().over(w))

# Running total
w = Window.partitionBy("customer_id").orderBy("date")
df.withColumn("running_total", F.sum("amount").over(w))

# Compare to group average
w_full = Window.partitionBy("dept")
df.withColumn("diff_from_avg", F.col("salary") - F.avg("salary").over(w_full))
```

Key: `partitionBy` defines the group, `orderBy` defines the order within the group. Without `orderBy`, window aggregates span the entire partition.

---

### Q12: Explain the difference between `rank`, `dense_rank`, and `row_number`.

Given department with salaries: 100k, 90k, 90k, 80k:

| Function | 100k | 90k | 90k | 80k |
|----------|------|-----|-----|-----|
| `row_number` | 1 | 2 | 3 | 4 |
| `rank` | 1 | 2 | 2 | **4** |
| `dense_rank` | 1 | 2 | 2 | **3** |

- **`row_number`** — always unique, ties broken arbitrarily (non-deterministic)
- **`rank`** — same value = same rank, gap after tie (skips 3)
- **`dense_rank`** — same value = same rank, no gaps

Use `row_number` for top-N (when you want exactly N rows). Use `rank`/`dense_rank` when ties matter.

---

### Q13: Why is Parquet better than CSV for Spark workloads?

| Feature | CSV | Parquet |
|---------|-----|---------|
| Format | Row-based text | Columnar binary |
| Schema | No (must infer) | Embedded in metadata |
| Compression | None by default | Snappy/GZIP (2-10x smaller) |
| Column pruning | Must read entire row | Reads only needed columns |
| Predicate pushdown | No | Yes (filter at read time) |
| Speed | Slow | 10-100x faster for analytics |

```python
# Parquet automatically uses predicate pushdown
df = spark.read.parquet("data.parquet")
df.filter(F.col("year") == 2024)  # only reads 2024 data (if partitioned)
```

Always use Parquet for intermediate and output data. CSV only for initial ingestion from external systems.

---

### Q14: How do partitioned writes work? What is partition pruning?

**Partitioned writes** create a directory structure based on column values:
```python
df.write.partitionBy("year", "month").parquet("output/")
# output/year=2024/month=01/part-001.parquet
# output/year=2024/month=02/part-001.parquet
```

**Partition pruning** — when you filter on the partition column, Spark only reads relevant directories:
```python
df = spark.read.parquet("output/")
df.filter(F.col("year") == 2024)  # only reads year=2024 directories
```

Best practices:
- Partition by low-cardinality columns (date, country, status — not user_id)
- Too many partitions = too many small files (degrades performance)
- Common pattern: partition by date for time-series data

---

### Q15: How do you write a UDF? When should you avoid UDFs?

```python
from pyspark.sql.types import StringType
import pyspark.sql.functions as F

@F.udf(returnType=StringType())
def classify(value):
    if value > 100: return "high"
    return "low"

df.withColumn("class", classify(F.col("value")))
```

**Avoid UDFs when**:
- A built-in function exists (`F.when`, `F.regexp_extract`, `F.concat`, etc.)
- You can express the logic with column expressions

**Why UDFs are slow**:
- Data serialized: JVM → Python → JVM (for each row!)
- Prevents Catalyst optimization (UDF is a black box)
- 2-100x slower than built-in functions

**If you must use UDFs**, prefer Pandas UDFs (vectorized):
```python
import pandas as pd

@F.pandas_udf("double")
def normalize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()
```

Pandas UDFs use Arrow for batch transfer — 10-100x faster than regular UDFs.

---

### Q16: How does `cache()` work? When is it useful?

`cache()` = `persist(StorageLevel.MEMORY_AND_DISK)`. It's **lazy** — nothing happens until an action triggers it.

```python
df = spark.read.parquet("huge_table.parquet")
df = df.filter(F.col("active") == True).cache()
df.count()  # triggers caching

# Now these reuse the cached data (no re-read/re-filter):
df.groupBy("dept").count().show()
df.filter(F.col("salary") > 100000).show()

df.unpersist()  # release when done
```

**When to cache**:
- DataFrame is reused multiple times in different actions
- After an expensive computation (complex joins, heavy filtering)
- In iterative algorithms (ML training loops)

**When NOT to cache**:
- DataFrame used only once
- Data doesn't fit in memory (will spill to disk anyway)
- Caching would evict more important data

Storage levels: `MEMORY_ONLY`, `MEMORY_AND_DISK`, `DISK_ONLY`, `MEMORY_ONLY_SER` (serialized, less memory, more CPU).

---

### Q17: How do you read the output of `explain()`?

```python
df.filter(F.col("salary") > 80000) \
  .groupBy("dept") \
  .agg(F.avg("salary").alias("avg_sal")) \
  .explain(True)
```

Read bottom-to-top (execution order):
1. **FileScan** — reads data from disk (check `PushedFilters` for predicate pushdown)
2. **Filter** — applies filter conditions
3. **Exchange** — shuffle (hashpartitioning) — this is a stage boundary
4. **HashAggregate** — partial aggregation (before shuffle) + final aggregation (after)

Red flags in explain output:
- `CartesianProduct` — cross join, usually a bug
- `BroadcastNestedLoopJoin` — inefficient join strategy
- No `PushedFilters` — predicate pushdown not working
- Huge `Exchange` — too much data being shuffled

---

### Q18: What are the different ways to handle schema evolution in Spark?

**Schema on read** — Parquet stores the schema, Spark reads it:
```python
df = spark.read.parquet("data/")  # schema from metadata
```

**Merge schemas** — when different parquet files have different columns:
```python
df = spark.read.option("mergeSchema", "true").parquet("data/")
```

**Schema enforcement** — reject files that don't match:
```python
schema = StructType([...])
df = spark.read.schema(schema).parquet("data/")  # error if mismatch
```

**Adding columns** — safe with Parquet (new column defaults to null):
```python
df_v1 = spark.read.parquet("v1/")  # columns: [id, name]
df_v2 = spark.read.parquet("v2/")  # columns: [id, name, email]
df_merged = df_v1.unionByName(df_v2, allowMissingColumns=True)
```

---

### Q19: How do you write efficient Spark SQL queries?

1. **Filter early** — push filters before joins/aggregations
2. **Select only needed columns** — projection pruning
3. **Use broadcast joins** for small tables
4. **Avoid UDFs** — use built-in functions
5. **Avoid `collect()` in the middle** of pipelines
6. **Use Parquet** with partitioning
7. **Set `spark.sql.shuffle.partitions`** appropriately
8. **Cache wisely** — reused DataFrames only

Anti-patterns:
```python
# BAD: collect then re-create DataFrame
ids = df.select("id").collect()  # pulls to driver
filtered = df.filter(F.col("id").isin([r.id for r in ids]))

# GOOD: use join or subquery
filtered = df.join(other_df, on="id")

# BAD: Python loop over rows
for row in df.collect():  # kills parallelism
    process(row)

# GOOD: use DataFrame operations
df.withColumn("result", process_udf(F.col("input")))
```

---

### Q20: Explain `createOrReplaceTempView` vs `createGlobalTempView`.

**`createOrReplaceTempView("name")`**:
- Session-scoped — only visible in the current SparkSession
- Replaced if already exists
- Most common usage

**`createGlobalTempView("name")`**:
- Application-scoped — visible across all SparkSessions
- Access with `global_temp.name` prefix
- Useful for shared data in multi-session apps

```python
df.createOrReplaceTempView("my_table")
spark.sql("SELECT * FROM my_table")  # works

df.createGlobalTempView("shared_table")
spark.sql("SELECT * FROM global_temp.shared_table")  # note prefix
```

---

## Debugging & Troubleshooting (Q21–Q25)

### Q21: A Spark job is running slowly — how do you diagnose it?

Step-by-step:
1. **Spark UI (port 4040)** — check the Jobs/Stages tabs:
   - Which stage is slow? Look at task duration distribution
   - Is one task much slower than others? → Data skew
   - Are tasks spilling to disk? → Not enough memory

2. **`df.explain(True)`** — check the plan:
   - `SortMergeJoin` where `BroadcastHashJoin` would work
   - Missing predicate pushdown
   - `CartesianProduct` (cross join by accident)

3. **Check shuffle metrics**:
   - Shuffle read/write size — if huge, filter data earlier
   - Number of shuffle partitions — 200 default might be wrong

4. **Check for common problems**:
   - `collect()` on large data → OOM on driver
   - Python UDFs → switch to built-in or Pandas UDFs
   - Too many small files → coalesce before write
   - Skewed keys → salting or broadcast join

---

### Q22: What causes OOM (Out of Memory) errors and how to fix them?

**Driver OOM**:
- `collect()` / `toPandas()` on large DataFrame → use `.limit()` first
- Too many tasks tracked by driver → reduce parallelism
- Fix: increase `spark.driver.memory`

**Executor OOM**:
- Data skew — one partition too large → salt or repartition
- Aggregation with `collect_list`/`collect_set` on high-cardinality keys
- Not enough memory per task → increase `spark.executor.memory` or reduce `spark.executor.cores`
- Shuffle spill → increase `spark.memory.fraction`

**Common pattern**: GC overhead limit exceeded → too many objects, try serialized caching (`MEMORY_ONLY_SER`).

---

### Q23: Spark job produces wrong results — how to debug?

1. **Check for null handling** — nulls propagate silently:
```python
# NULL + 5 = NULL, NULL == NULL is NULL (not true!)
df.filter(F.col("x") == F.col("y"))  # won't match NULL == NULL
df.filter(F.col("x").eqNullSafe(F.col("y")))  # handles nulls
```

2. **Check join type** — inner vs left changes results:
```python
# Inner drops unmatched rows silently
df1.join(df2, on="key")  # rows without match disappear
```

3. **Check for duplicate keys in joins** — causes row multiplication:
```python
# If df2 has duplicate keys, join produces duplicates
df1.join(df2, on="key")  # count increases!
```

4. **Check orderBy stability** — `row_number` without unique ordering is non-deterministic

5. **Print intermediate results**: `df.show()`, `df.count()`, `df.printSchema()` at each step

---

### Q24: What is speculative execution?

When enabled, Spark launches duplicate copies of slow-running tasks on different executors. Whichever finishes first wins, the other is killed.

```python
spark.conf.set("spark.speculation", "true")
spark.conf.set("spark.speculation.multiplier", "1.5")  # 1.5x slower than median
```

Useful for: heterogeneous clusters, unreliable nodes, GC pauses.

Dangerous for: non-idempotent operations (writing to external systems), as both copies might complete.

---

### Q25: How do you handle corrupt/malformed data in Spark?

**CSV with corrupt rows**:
```python
df = spark.read.csv("data.csv",
    header=True,
    mode="PERMISSIVE",           # default: puts corrupt rows in _corrupt_record
    columnNameOfCorruptRecord="_corrupt_record"
)
bad_rows = df.filter(F.col("_corrupt_record").isNotNull())
```

Modes: `PERMISSIVE` (keep with null), `DROPMALFORMED` (skip), `FAILFAST` (throw error).

**JSON with schema mismatch**:
```python
df = spark.read.schema(expected_schema).json("data.json")
# Mismatched fields become null
```

---

## Trade-offs & Decisions (Q26–Q30)

### Q26: When should you use Spark vs Pandas?

| Factor | Pandas | Spark |
|--------|--------|-------|
| Data size | < 10GB (fits in RAM) | > 10GB or distributed |
| Latency | Milliseconds | Seconds (JVM startup, scheduling) |
| API | Richer, more mature | Subset of Pandas-like API |
| Debugging | Easy (single process) | Hard (distributed, lazy eval) |
| Ecosystem | NumPy, scikit-learn | Spark MLlib, Delta Lake |

Rule: if your data fits in memory on one machine, use Pandas. Spark overhead isn't worth it for small data.

Hybrid: use Spark for distributed processing, then `.toPandas()` for final analysis/visualization on aggregated results.

---

### Q27: Spark Streaming vs Batch — what are the trade-offs?

**Batch**: process all data at once, scheduled (hourly, daily). Simple, predictable resource usage, easy debugging.

**Structured Streaming**: micro-batch or continuous processing. Same DataFrame API, but reads from a stream.

```python
# Batch
df = spark.read.parquet("data/")

# Streaming
df = spark.readStream.format("kafka").option("subscribe", "topic").load()
df.writeStream.format("parquet").start("output/")
```

Trade-offs:
- Streaming adds complexity (checkpointing, late data, exactly-once)
- Batch is simpler to debug and retry
- Use streaming only when latency requirement < batch interval

---

### Q28: How do you choose the right number of partitions?

**Too few** — underutilization, large partitions, OOM risk
**Too many** — overhead per task (~100ms), small file problem, scheduler overhead

Guidelines:
- **Input partitions**: 1 partition per 128MB of data (HDFS block size)
- **Shuffle partitions**: 2-3x number of cores in the cluster
- **Output partitions**: aim for 100MB-1GB per file

```python
# Check current
df.rdd.getNumPartitions()

# Tune shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "100")

# Or use AQE (auto-tunes)
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

---

### Q29: `persist()` with different storage levels — how to choose?

| Level | Memory | Disk | Serialized | When |
|-------|--------|------|------------|------|
| `MEMORY_ONLY` | Yes | No | No | Data fits, need fast access |
| `MEMORY_AND_DISK` | Yes | Spillover | No | Data might not fit in memory |
| `DISK_ONLY` | No | Yes | - | Recomputation more expensive than disk I/O |
| `MEMORY_ONLY_SER` | Yes | No | Yes | Memory tight, CPU available |
| `MEMORY_AND_DISK_SER` | Yes | Spillover | Yes | Memory tight, need persistence |

`cache()` = `persist(MEMORY_AND_DISK)` — good default.

Use `_SER` variants when memory is tight — serialized data takes ~2-5x less space but requires CPU for deserialization.

---

### Q30: Delta Lake vs plain Parquet — when to use Delta?

**Parquet** — simple columnar files. No transactions, no updates, no schema enforcement.

**Delta Lake** — adds ACID transactions on top of Parquet:
- **ACID transactions** — concurrent reads/writes are safe
- **Time travel** — query previous versions (`VERSION AS OF 3`)
- **MERGE** — upserts (update or insert)
- **Schema enforcement/evolution** — rejects incompatible writes
- **Z-ordering** — multi-dimensional clustering for faster queries
- **VACUUM** — clean up old files

Use Delta Lake when:
- Multiple writers/pipelines touch the same table
- You need UPSERT (merge) capability
- You need data versioning or rollback
- You're building a lakehouse architecture

Use plain Parquet when:
- Simple, one-time writes
- Read-only analytics
- Don't want the Delta dependency
