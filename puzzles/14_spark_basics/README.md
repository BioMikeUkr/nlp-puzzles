# Module 12: Spark Basics

## Overview

Apache Spark is a distributed computing framework for processing large datasets across clusters. This module covers **PySpark** — the Python API for Spark — including DataFrames, transformations, aggregations, joins, window functions, Spark SQL, UDFs, and I/O formats.

### Learning Objectives
- Create SparkSession and DataFrames from various sources
- Apply select, filter, groupBy, agg, join, window functions
- Write SQL queries via Spark SQL
- Use UDFs (regular and Pandas UDF)
- Read/write Parquet, CSV, JSON
- Understand lazy evaluation, execution plans, caching, and partitioning

---

## 1. SparkSession & DataFrames

### SparkSession
Entry point to all Spark functionality. In local mode, use `local[*]` to use all CPU cores:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("MyApp") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
```

Key configs:
- `spark.driver.memory` — memory for the driver process
- `spark.executor.memory` — memory per executor (cluster mode)
- `spark.sql.shuffle.partitions` — number of partitions after shuffle (default 200)

### Creating DataFrames

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# From list of tuples
data = [("Alice", 30, 95000.0), ("Bob", 25, 88000.0)]
df = spark.createDataFrame(data, ["name", "age", "salary"])

# With explicit schema
schema = StructType([
    StructField("name", StringType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("salary", DoubleType(), nullable=True),
])
df = spark.createDataFrame(data, schema)

# From Pandas
import pandas as pd
pdf = pd.DataFrame({"name": ["Alice"], "age": [30]})
sdf = spark.createDataFrame(pdf)

# From CSV
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# From Parquet
df = spark.read.parquet("data.parquet")
```

### Schema Inspection
```python
df.printSchema()      # tree view
df.dtypes             # list of (name, type)
df.columns            # list of column names
df.count()            # number of rows
df.describe().show()  # basic stats
```

---

## 2. Basic Operations

### select, filter, withColumn
```python
from pyspark.sql import functions as F

# Select columns
df.select("name", "salary").show()

# Filter rows
df.filter(F.col("salary") > 90000).show()
df.filter((F.col("dept") == "Eng") & (F.col("salary") > 90000))

# Add/modify column
df.withColumn("salary_k", F.col("salary") / 1000)
df.withColumn("bonus", F.lit(5000))  # literal value
```

### when / otherwise (conditional logic)
```python
df.withColumn(
    "level",
    F.when(F.col("salary") >= 100000, "Senior")
     .when(F.col("salary") >= 75000, "Mid")
     .otherwise("Junior")
)
```

### orderBy, distinct, drop, limit
```python
df.orderBy(F.col("salary").desc())
df.select("dept").distinct()
df.drop("temp_col")
df.limit(10)
```

### Renaming and Casting
```python
df.withColumnRenamed("old_name", "new_name")
df.withColumn("age", F.col("age").cast("integer"))
```

---

## 3. String and Date Functions

### String functions
```python
F.upper(F.col("name"))              # ALICE
F.lower(F.col("name"))              # alice
F.trim(F.col("text"))               # remove whitespace
F.length(F.col("name"))             # character count
F.substring(F.col("name"), 1, 3)    # first 3 chars
F.concat_ws(" ", "first", "last")   # join with separator
F.regexp_extract(F.col("text"), r"(\d+)", 1)  # regex group
F.regexp_replace(F.col("text"), r"\d+", "NUM")
```

### Date functions
```python
F.year(F.col("date"))               # extract year
F.month(F.col("date"))              # extract month
F.datediff(F.col("end"), F.col("start"))  # days between
F.date_add(F.col("date"), 30)       # add 30 days
F.current_date()                     # today
F.to_date(F.col("str"), "yyyy-MM-dd")  # parse string
```

---

## 4. Handling Nulls

```python
df.dropna()                              # drop rows with any null
df.dropna(subset=["name", "email"])      # drop if name or email is null
df.fillna({"name": "Unknown", "age": 0}) # fill nulls
df.filter(F.col("name").isNull())        # find null rows
df.filter(F.col("name").isNotNull())     # find non-null rows
```

`coalesce` picks the first non-null value:
```python
df.withColumn("val", F.coalesce(F.col("primary"), F.col("secondary"), F.lit(0)))
```

---

## 5. Aggregations

### groupBy + agg
```python
df.groupBy("department").agg(
    F.count("*").alias("count"),
    F.round(F.avg("salary"), 2).alias("avg_salary"),
    F.min("salary").alias("min_salary"),
    F.max("salary").alias("max_salary"),
    F.sum("salary").alias("total_salary"),
    F.collect_list("name").alias("names"),
)
```

### Multiple groupBy keys
```python
df.groupBy("dept", "year").agg(F.count("*").alias("cnt"))
```

### Pivot tables
```python
df.groupBy("customer_id").pivot("product").agg(F.sum("amount")).fillna(0)
```

---

## 6. Joins

### Join types
```python
# Inner (default) — only matching rows
df1.join(df2, on="key", how="inner")

# Left — all from left, matching from right
df1.join(df2, on="key", how="left")

# Right, Full outer
df1.join(df2, on="key", how="right")
df1.join(df2, on="key", how="full")

# Cross join — cartesian product (dangerous!)
df1.crossJoin(df2)
```

### Join on different column names
```python
df1.join(df2, df1["user_id"] == df2["id"], how="inner")
```

### Anti join (rows in left NOT in right)
```python
df1.join(df2, on="key", how="left_anti")
```

### Broadcast join (force small table broadcast)
```python
from pyspark.sql.functions import broadcast
big_df.join(broadcast(small_df), on="key")
```

---

## 7. Window Functions

Window functions compute values across a "window" of rows without collapsing them (unlike `groupBy`).

### Define a window
```python
from pyspark.sql import Window

w = Window.partitionBy("department").orderBy(F.col("salary").desc())
```

### Ranking functions
```python
df.withColumn("rank", F.rank().over(w))           # gaps after ties
df.withColumn("dense_rank", F.dense_rank().over(w)) # no gaps
df.withColumn("row_num", F.row_number().over(w))   # unique sequential
```

### Aggregate over window
```python
w = Window.partitionBy("customer_id").orderBy("order_date")
df.withColumn("running_total", F.sum("amount").over(w))
df.withColumn("avg_so_far", F.avg("amount").over(w))
```

### lag / lead
```python
df.withColumn("prev_amount", F.lag("amount", 1).over(w))
df.withColumn("next_amount", F.lead("amount", 1).over(w))
```

### Top-N per group
```python
w = Window.partitionBy("dept").orderBy(F.col("salary").desc())
df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= 3)
```

### Unbounded windows
```python
# Full partition window (no ordering — aggregate over entire group)
w_full = Window.partitionBy("dept")
df.withColumn("dept_avg", F.avg("salary").over(w_full))
df.withColumn("pct_of_dept", F.col("salary") / F.sum("salary").over(w_full))
```

---

## 8. Union and Deduplication

```python
# Union — same schema required, keeps duplicates
df1.union(df2)

# Union by column name (not position)
df1.unionByName(df2)

# Deduplicate
df.dropDuplicates()                     # all columns
df.dropDuplicates(["email"])            # by specific columns
```

---

## 9. Spark SQL

Register DataFrames as temporary views and query with SQL:

```python
df.createOrReplaceTempView("employees")

result = spark.sql("""
    SELECT department, COUNT(*) as cnt, ROUND(AVG(salary), 0) as avg_sal
    FROM employees
    GROUP BY department
    ORDER BY avg_sal DESC
""")
result.show()
```

### CTEs (Common Table Expressions)
```python
spark.sql("""
    WITH dept_avg AS (
        SELECT department, AVG(salary) as avg_sal
        FROM employees
        GROUP BY department
    )
    SELECT e.name, e.salary, d.avg_sal
    FROM employees e
    JOIN dept_avg d ON e.department = d.department
    WHERE e.salary > d.avg_sal
""")
```

DataFrame API and SQL produce identical execution plans — both go through the Catalyst optimizer.

---

## 10. User Defined Functions (UDFs)

### Python UDF
```python
from pyspark.sql.types import StringType

@F.udf(returnType=StringType())
def grade(salary):
    if salary >= 100000: return "A"
    elif salary >= 80000: return "B"
    return "C"

df.withColumn("grade", grade(F.col("salary")))
```

**Warning**: Python UDFs serialize data JVM → Python → JVM. This is 2-100x slower than built-in functions. Avoid UDFs when a built-in alternative exists.

### Pandas UDF (Vectorized)
Much faster — operates on Arrow batches:

```python
import pandas as pd

@F.pandas_udf(DoubleType())
def normalize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()
```

### Register UDF for SQL
```python
spark.udf.register("my_grade", lambda s: "A" if s >= 100000 else "B", StringType())
spark.sql("SELECT name, my_grade(salary) as grade FROM employees")
```

---

## 11. Reading & Writing Data

### CSV
```python
df = spark.read.csv("data.csv", header=True, inferSchema=True)
df.write.mode("overwrite").csv("output.csv", header=True)
```

### Parquet (preferred for analytics)
```python
df = spark.read.parquet("data.parquet")
df.write.mode("overwrite").parquet("output.parquet")
```

Parquet advantages over CSV:
- **Columnar** — reads only needed columns (projection pruning)
- **Compressed** — snappy/gzip, 2-10x smaller
- **Schema in metadata** — no `inferSchema` needed
- **Predicate pushdown** — filters pushed to read stage

### JSON
```python
df = spark.read.json("data.json")
df.write.mode("overwrite").json("output.json")
```

### Partitioned writes
```python
df.write.mode("overwrite").partitionBy("department").parquet("output/")
# Creates: output/department=Engineering/part-001.parquet
#          output/department=Marketing/part-001.parquet
```

Reading partitioned data enables **partition pruning** — Spark only reads relevant directories.

### Write modes
| Mode | Behavior |
|------|----------|
| `overwrite` | Replace existing data |
| `append` | Add to existing data |
| `ignore` | Skip if exists |
| `error` (default) | Throw if exists |

---

## 12. Caching & Performance

### cache / persist
```python
df.cache()       # = persist(StorageLevel.MEMORY_AND_DISK)
df.count()       # triggers caching (lazy!)
df.unpersist()   # release cached data

df.is_cached     # check if cached
```

Cache when a DataFrame is reused multiple times. Don't cache one-time transformations.

### explain — viewing the query plan
```python
df.explain()       # physical plan only
df.explain(True)   # parsed → analyzed → optimized → physical
```

Look for:
- `BroadcastHashJoin` (good for small tables) vs `SortMergeJoin`
- `FileScan` with `PushedFilters` (predicate pushdown working)
- `Exchange` nodes (shuffle happening)

### Catalyst Optimizer
Spark's query optimizer. Stages:
1. **Analysis** — resolve column names, tables
2. **Logical Optimization** — predicate pushdown, projection pruning, constant folding
3. **Physical Planning** — choose join strategy, broadcast vs sort-merge
4. **Code Generation** — Tungsten generates optimized JVM bytecode

---

## 13. Partitioning & Shuffle

### Transformations: Narrow vs Wide
**Narrow** — no data movement: `select`, `filter`, `withColumn`, `map`
**Wide** — require shuffle: `groupBy`, `join`, `orderBy`, `distinct`, `repartition`

### repartition vs coalesce
```python
df.repartition(10)             # full shuffle, even distribution
df.repartition("department")   # partition by column value
df.coalesce(2)                 # no shuffle, just merge partitions (only decrease)
```

Use `coalesce` to reduce partitions (cheaper). Use `repartition` to increase or rebalance.

### Checking partitions
```python
df.rdd.getNumPartitions()
```

---

## 14. Error Handling

```python
from pyspark.sql.utils import AnalysisException

try:
    df.select("nonexistent_column")
except AnalysisException as e:
    print(f"Column not found: {e}")

# Safe column check
if "salary" in df.columns:
    df.select("salary")
```

---

## 15. Collecting Results

```python
# collect() — all data to driver (careful with large data!)
rows = df.collect()
for row in rows:
    print(row["name"], row["salary"])

# toPandas() — convert to Pandas DataFrame
pdf = df.toPandas()

# take(n) — first n rows
first_5 = df.take(5)

# first() — single row
row = df.first()
```

**Warning**: `collect()` and `toPandas()` bring ALL data to the driver. For large datasets, this causes OOM. Always `.limit()` or `.filter()` first.

---

## Module Structure

```
12_spark_basics/
├── README.md              # This file — theory
├── QUESTIONS.md           # 30 interview questions
├── requirements.txt
├── fixtures/input/        # employees.csv, orders.csv, customers.csv
├── learning/
│   ├── 01_pyspark_fundamentals.ipynb
│   ├── 02_transformations_and_aggregations.ipynb
│   └── 03_spark_sql_and_io.ipynb
├── tasks/
│   ├── task_01_dataframe_basics.ipynb
│   ├── task_02_aggregations_joins.ipynb
│   └── task_03_spark_sql_window.ipynb
└── solutions/
    ├── task_01_dataframe_basics_solution.ipynb
    ├── task_02_aggregations_joins_solution.ipynb
    └── task_03_spark_sql_window_solution.ipynb
```
