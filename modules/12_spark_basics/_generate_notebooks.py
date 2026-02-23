#!/usr/bin/env python3
"""Generate all notebooks for Module 12 — Spark Basics."""

import nbformat as nbf
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

JAVA_SETUP = """\
import os, shutil, subprocess, sys

def _find_java():
    \"\"\"Check if java is available on PATH or in JAVA_HOME.\"\"\"
    if os.environ.get("JAVA_HOME"):
        java_bin = os.path.join(os.environ["JAVA_HOME"], "bin", "java")
        if os.path.isfile(java_bin):
            return True
    return shutil.which("java") is not None

def _find_installed_jdk():
    \"\"\"Look for a previously installed JDK in ~/.jdk.\"\"\"
    jdk_dir = os.path.expanduser("~/.jdk")
    if os.path.exists(jdk_dir):
        for d in sorted(os.listdir(jdk_dir)):
            candidate = os.path.join(jdk_dir, d)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "bin", "java")):
                return candidate
    return None

# Auto-install Java if not available (required by PySpark)
if not _find_java():
    prev = _find_installed_jdk()
    if prev:
        os.environ["JAVA_HOME"] = prev
        print(f"Using JAVA_HOME={prev}")
    else:
        print("Java not found. Installing JDK 17 via install-jdk...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "install-jdk"])
        import jdk
        path = jdk.install("17")
        os.environ["JAVA_HOME"] = path
        print(f"JAVA_HOME set to {path}")
else:
    print(f"Java found. JAVA_HOME={os.environ.get('JAVA_HOME', '(system)')}")
"""


def nb():
    """Create a new notebook."""
    return nbf.v4.new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    })


def md(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def save(notebook, path):
    full = os.path.join(MODULE_DIR, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        nbf.write(notebook, f)
    print(f"  Written: {path}")


# ==============================================================================
# LEARNING 01 — PySpark Fundamentals
# ==============================================================================
def learning_01():
    n = nb()
    n.cells = [
        md("# 01 — PySpark Fundamentals\n\nCore concepts: SparkSession, DataFrame creation, basic operations."),

        md("## 0. Java Setup (PySpark requires Java)"),
        code(JAVA_SETUP),

        md("## 1. Creating a SparkSession"),
        code("""\
from pyspark.sql import SparkSession

spark = SparkSession.builder \\
    .master("local[*]") \\
    .appName("Module12-Fundamentals") \\
    .config("spark.driver.memory", "2g") \\
    .config("spark.sql.shuffle.partitions", "4") \\
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"Spark version: {spark.version}")
print(f"App name: {spark.sparkContext.appName}")"""),

        md("## 2. Creating DataFrames"),
        md("### From Python lists"),
        code("""\
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# From list of tuples (schema inferred)
data = [("Alice", 30, 95000.0), ("Bob", 25, 88000.0), ("Charlie", 35, 72000.0)]
df = spark.createDataFrame(data, ["name", "age", "salary"])
df.show()
df.printSchema()"""),

        md("### With explicit schema"),
        code("""\
schema = StructType([
    StructField("name", StringType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("salary", DoubleType(), nullable=True),
])

df2 = spark.createDataFrame(data, schema)
df2.printSchema()"""),

        md("### From Pandas DataFrame"),
        code("""\
import pandas as pd

pdf = pd.DataFrame({"product": ["Laptop", "Phone", "Tablet"], "price": [1200, 800, 500]})
sdf = spark.createDataFrame(pdf)
sdf.show()"""),

        md("### From CSV file"),
        code("""\
import os

FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))

emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
emp_df.show(5)
emp_df.printSchema()
print(f"Rows: {emp_df.count()}, Columns: {len(emp_df.columns)}")"""),

        md("## 3. Basic DataFrame Operations"),
        md("### select, filter, withColumn"),
        code("""\
from pyspark.sql import functions as F

# select columns
emp_df.select("name", "department", "salary").show(5)

# filter rows
eng = emp_df.filter(F.col("department") == "Engineering")
eng.show()

# add/modify column
emp_df.withColumn("salary_k", F.round(F.col("salary") / 1000, 1)).select("name", "salary", "salary_k").show(5)"""),

        md("### orderBy, distinct, drop"),
        code("""\
# sort
emp_df.orderBy(F.col("salary").desc()).select("name", "salary").show(5)

# distinct departments
emp_df.select("department").distinct().show()

# drop column
emp_df.drop("hire_date").columns"""),

        md("### Column expressions and when/otherwise"),
        code("""\
emp_df.withColumn(
    "salary_band",
    F.when(F.col("salary") >= 95000, "Senior")
     .when(F.col("salary") >= 75000, "Mid")
     .otherwise("Junior")
).select("name", "salary", "salary_band").show()"""),

        md("## 4. String and Date Functions"),
        code("""\
emp_df.select(
    "name",
    F.upper(F.col("name")).alias("name_upper"),
    F.length(F.col("name")).alias("name_len"),
    F.col("hire_date"),
    F.year(F.col("hire_date")).alias("hire_year"),
    F.datediff(F.current_date(), F.col("hire_date")).alias("days_employed"),
).show(5)"""),

        md("## 5. Handling Nulls"),
        code("""\
data_nulls = [(1, "Alice", 100), (2, None, 200), (3, "Charlie", None)]
df_nulls = spark.createDataFrame(data_nulls, ["id", "name", "value"])

# Drop rows with any null
df_nulls.dropna().show()

# Fill nulls
df_nulls.fillna({"name": "Unknown", "value": 0}).show()

# Check for null
df_nulls.filter(F.col("name").isNull()).show()"""),

        md("## 6. Collecting Results"),
        code("""\
# collect() — returns list of Row objects
rows = emp_df.select("name", "salary").limit(3).collect()
for row in rows:
    print(f"{row['name']}: ${row['salary']:,}")

# toPandas() — converts to pandas DataFrame
pdf = emp_df.select("name", "department", "salary").limit(5).toPandas()
print(type(pdf))
pdf"""),

        md("## Cleanup"),
        code("spark.stop()\nprint('SparkSession stopped.')"),
    ]
    save(n, "learning/01_pyspark_fundamentals.ipynb")


# ==============================================================================
# LEARNING 02 — Transformations and Aggregations
# ==============================================================================
def learning_02():
    n = nb()
    n.cells = [
        md("# 02 — Transformations and Aggregations\n\ngroupBy, agg, joins, union, window functions."),

        md("## Setup"),
        code(JAVA_SETUP),
        code("""\
import os
from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder \\
    .master("local[*]") \\
    .appName("Module12-Aggregations") \\
    .config("spark.sql.shuffle.partitions", "4") \\
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))

emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
orders_df = spark.read.csv(os.path.join(FIXTURES, "orders.csv"), header=True, inferSchema=True)
customers_df = spark.read.csv(os.path.join(FIXTURES, "customers.csv"), header=True, inferSchema=True)
print("Data loaded.")"""),

        md("## 1. groupBy + Aggregations"),
        code("""\
# Basic aggregation
emp_df.groupBy("department").agg(
    F.count("*").alias("count"),
    F.round(F.avg("salary"), 0).alias("avg_salary"),
    F.min("salary").alias("min_salary"),
    F.max("salary").alias("max_salary"),
    F.sum("salary").alias("total_salary"),
).orderBy("department").show()"""),

        md("### Multiple groupBy keys"),
        code("""\
orders_df.groupBy("customer_id", "product").agg(
    F.count("*").alias("num_orders"),
    F.round(F.sum("amount"), 2).alias("total_spent"),
).orderBy("customer_id", "product").show()"""),

        md("## 2. Joins"),
        md("### Inner join"),
        code("""\
joined = orders_df.join(customers_df, on="customer_id", how="inner")
joined.select("order_id", "name", "product", "amount", "city").show(5)"""),

        md("### Left join"),
        code("""\
# All customers, even those without orders
all_customers = customers_df.join(orders_df, on="customer_id", how="left")
all_customers.select("customer_id", "name", "order_id", "product").show()"""),

        md("### Join with different column names"),
        code("""\
# Rename for clarity
emp_renamed = emp_df.withColumnRenamed("name", "emp_name").withColumnRenamed("id", "emp_id")

# Join on different column names
result = orders_df.join(
    emp_renamed,
    orders_df["customer_id"] == emp_renamed["emp_id"],
    how="inner"
)
result.select("order_id", "emp_name", "product", "amount").show(5)"""),

        md("## 3. Union and Deduplication"),
        code("""\
df1 = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "val"])
df2 = spark.createDataFrame([(2, "B"), (3, "C")], ["id", "val"])

# union (keeps dupes)
df1.union(df2).show()

# union + dedupe
df1.union(df2).dropDuplicates().show()

# unionByName (matches columns by name, not position)
df3 = spark.createDataFrame([("X", 4), ("Y", 5)], ["val", "id"])
df1.unionByName(df3).show()"""),

        md("## 4. Window Functions"),
        md("### rank, row_number, dense_rank"),
        code("""\
w = Window.partitionBy("department").orderBy(F.col("salary").desc())

ranked = emp_df.withColumn("rank", F.rank().over(w)) \\
               .withColumn("dense_rank", F.dense_rank().over(w)) \\
               .withColumn("row_num", F.row_number().over(w))

ranked.select("name", "department", "salary", "rank", "dense_rank", "row_num").show()"""),

        md("### Running totals and lag/lead"),
        code("""\
w_order = Window.partitionBy("customer_id").orderBy("order_date")

orders_df.withColumn("running_total", F.sum("amount").over(w_order)) \\
         .withColumn("prev_amount", F.lag("amount", 1).over(w_order)) \\
         .withColumn("next_amount", F.lead("amount", 1).over(w_order)) \\
         .select("customer_id", "order_date", "amount", "running_total", "prev_amount", "next_amount") \\
         .filter(F.col("customer_id") == 1) \\
         .show()"""),

        md("### Top-N per group"),
        code("""\
# Top-2 highest paid per department
w = Window.partitionBy("department").orderBy(F.col("salary").desc())
emp_df.withColumn("rn", F.row_number().over(w)) \\
      .filter(F.col("rn") <= 2) \\
      .select("department", "name", "salary", "rn") \\
      .orderBy("department", "rn") \\
      .show()"""),

        md("## 5. Pivot and Unpivot"),
        code("""\
# Pivot: product sales by customer
orders_df.groupBy("customer_id").pivot("product").agg(
    F.round(F.sum("amount"), 2)
).fillna(0).orderBy("customer_id").show()"""),

        md("## Cleanup"),
        code("spark.stop()\nprint('Done.')"),
    ]
    save(n, "learning/02_transformations_and_aggregations.ipynb")


# ==============================================================================
# LEARNING 03 — Spark SQL and I/O
# ==============================================================================
def learning_03():
    n = nb()
    n.cells = [
        md("# 03 — Spark SQL, UDFs, and I/O\n\nSQL queries, UDFs, reading/writing Parquet/JSON, caching, explain."),

        md("## Setup"),
        code(JAVA_SETUP),
        code("""\
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, DoubleType

spark = SparkSession.builder \\
    .master("local[*]") \\
    .appName("Module12-SQL-IO") \\
    .config("spark.sql.shuffle.partitions", "4") \\
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))

emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
orders_df = spark.read.csv(os.path.join(FIXTURES, "orders.csv"), header=True, inferSchema=True)
customers_df = spark.read.csv(os.path.join(FIXTURES, "customers.csv"), header=True, inferSchema=True)
print("Data loaded.")"""),

        md("## 1. Spark SQL"),
        md("### Register temp views and run SQL"),
        code("""\
emp_df.createOrReplaceTempView("employees")
orders_df.createOrReplaceTempView("orders")
customers_df.createOrReplaceTempView("customers")

result = spark.sql(\"\"\"
    SELECT department, COUNT(*) as cnt, ROUND(AVG(salary), 0) as avg_sal
    FROM employees
    GROUP BY department
    ORDER BY avg_sal DESC
\"\"\")
result.show()"""),

        md("### Subqueries and CTEs"),
        code("""\
spark.sql(\"\"\"
    WITH dept_stats AS (
        SELECT department,
               AVG(salary) as avg_salary
        FROM employees
        GROUP BY department
    )
    SELECT e.name, e.department, e.salary,
           ROUND(d.avg_salary, 0) as dept_avg,
           ROUND(e.salary - d.avg_salary, 0) as diff_from_avg
    FROM employees e
    JOIN dept_stats d ON e.department = d.department
    ORDER BY diff_from_avg DESC
\"\"\").show()"""),

        md("### Joins in SQL"),
        code("""\
spark.sql(\"\"\"
    SELECT c.name as customer, o.product, o.amount, o.order_date
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.amount > 500
    ORDER BY o.amount DESC
\"\"\").show()"""),

        md("## 2. User Defined Functions (UDFs)"),
        md("### Python UDF"),
        code("""\
@F.udf(returnType=StringType())
def salary_grade(salary):
    if salary is None:
        return "Unknown"
    if salary >= 100000:
        return "A"
    elif salary >= 80000:
        return "B"
    elif salary >= 70000:
        return "C"
    return "D"

emp_df.withColumn("grade", salary_grade(F.col("salary"))) \\
      .select("name", "salary", "grade").show()"""),

        md("### Pandas UDF (Vectorized — much faster)"),
        code("""\
import pandas as pd

@F.pandas_udf(DoubleType())
def normalize_salary(salary: pd.Series) -> pd.Series:
    return (salary - salary.mean()) / salary.std()

emp_df.withColumn("salary_normalized", normalize_salary(F.col("salary"))) \\
      .select("name", "salary", F.round("salary_normalized", 2).alias("normalized")) \\
      .show()"""),

        md("### Register UDF for SQL"),
        code("""\
spark.udf.register("salary_grade_sql", lambda s: "A" if s and s >= 100000 else "B" if s and s >= 80000 else "C", StringType())

spark.sql(\"\"\"
    SELECT name, salary, salary_grade_sql(salary) as grade
    FROM employees
    ORDER BY salary DESC
    LIMIT 5
\"\"\").show()"""),

        md("## 3. Reading and Writing Data"),
        md("### Parquet (columnar, compressed)"),
        code("""\
import tempfile

tmpdir = tempfile.mkdtemp()

# Write
parquet_path = os.path.join(tmpdir, "employees.parquet")
emp_df.write.mode("overwrite").parquet(parquet_path)
print(f"Written to {parquet_path}")

# Read
df_parquet = spark.read.parquet(parquet_path)
df_parquet.show(3)
print(f"Count: {df_parquet.count()}")"""),

        md("### JSON"),
        code("""\
json_path = os.path.join(tmpdir, "orders.json")
orders_df.write.mode("overwrite").json(json_path)

df_json = spark.read.json(json_path)
df_json.show(3)"""),

        md("### Partitioned writes"),
        code("""\
part_path = os.path.join(tmpdir, "emp_by_dept")
emp_df.write.mode("overwrite").partitionBy("department").parquet(part_path)

# Read a single partition
eng_df = spark.read.parquet(os.path.join(part_path, "department=Engineering"))
print(f"Engineering employees: {eng_df.count()}")
eng_df.show()"""),

        md("## 4. Caching and Performance"),
        code("""\
# Cache a DataFrame in memory
emp_df.cache()
emp_df.count()  # triggers caching

# Check if cached
print(f"Is cached: {emp_df.is_cached}")

# Unpersist
emp_df.unpersist()
print(f"After unpersist: {emp_df.is_cached}")"""),

        md("### Explain — viewing the query plan"),
        code("""\
# See execution plan
result = emp_df.filter(F.col("salary") > 80000).groupBy("department").agg(F.avg("salary"))
result.explain(True)"""),

        md("### Broadcast join hint"),
        code("""\
from pyspark.sql.functions import broadcast

# Force broadcast of small table
result = orders_df.join(broadcast(customers_df), on="customer_id")
result.explain()"""),

        md("## 5. Error Handling Patterns"),
        code("""\
from pyspark.sql.utils import AnalysisException

# Handle missing columns gracefully
try:
    emp_df.select("nonexistent_column")
except AnalysisException as e:
    print(f"Caught AnalysisException: {e}")

# Safe column check
if "salary" in emp_df.columns:
    print("salary column exists")
else:
    print("salary column missing")"""),

        md("## Cleanup"),
        code("""\
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
spark.stop()
print("Done.")"""),
    ]
    save(n, "learning/03_spark_sql_and_io.ipynb")


# ==============================================================================
# TASK 01 — DataFrame Basics (+ Solution)
# ==============================================================================
def task_01():
    setup = """\
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

spark = SparkSession.builder \\
    .master("local[*]") \\
    .appName("Task01") \\
    .config("spark.sql.shuffle.partitions", "4") \\
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))

print("Setup complete.")"""

    cleanup = 'spark.stop()\nprint("All tasks done!")'

    # --- TASK ---
    t = nb()
    t.cells = [
        md("# Task 01 — DataFrame Basics\n\nCreate DataFrames, select, filter, add columns, handle nulls."),
        md("## Setup"),
        code(JAVA_SETUP),
        code(setup),

        md("""\
## Task 1.1: Create a DataFrame from a list

Create a DataFrame `products_df` from the following data with columns: `name` (string), `category` (string), `price` (double).

Data:
- ("Laptop", "Electronics", 1200.0)
- ("Desk", "Furniture", 350.0)
- ("Phone", "Electronics", 800.0)
- ("Chair", "Furniture", 250.0)
- ("Tablet", "Electronics", 500.0)"""),

        code("""\
# YOUR CODE HERE
products_df = ...

# TEST — Do not modify
assert products_df.count() == 5, f"Expected 5 rows, got {products_df.count()}"
assert set(products_df.columns) == {"name", "category", "price"}
assert products_df.schema["price"].dataType == DoubleType()
rows = {r["name"]: r["price"] for r in products_df.collect()}
assert rows["Laptop"] == 1200.0
print("Task 1.1 passed!")"""),

        md("""\
## Task 1.2: Read CSV and basic stats

Read `employees.csv` into `emp_df`. Then create `eng_df` containing only Engineering department employees with salary > 95000, sorted by salary descending."""),

        code("""\
# YOUR CODE HERE
emp_df = ...
eng_df = ...

# TEST — Do not modify
assert emp_df.count() == 15, f"Expected 15, got {emp_df.count()}"
assert eng_df.count() == 3, f"Expected 3 high-paid engineers, got {eng_df.count()}"
salaries = [r["salary"] for r in eng_df.collect()]
assert salaries == sorted(salaries, reverse=True), "Should be sorted desc"
assert all(s > 95000 for s in salaries)
print("Task 1.2 passed!")"""),

        md("""\
## Task 1.3: Add computed columns

Starting from `emp_df` (read employees.csv again if needed), create `emp_enhanced` with two new columns:
- `salary_band`: "Senior" if salary >= 95000, "Mid" if >= 75000, else "Junior"
- `tenure_years`: number of full years from `hire_date` to 2024-01-01 (use `datediff` and divide by 365, cast to int)"""),

        code("""\
# YOUR CODE HERE
emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
emp_enhanced = ...

# TEST — Do not modify
assert "salary_band" in emp_enhanced.columns
assert "tenure_years" in emp_enhanced.columns
bands = {r["name"]: r["salary_band"] for r in emp_enhanced.collect()}
assert bands["Noah"] == "Senior"   # 105000
assert bands["Charlie"] == "Junior"  # 72000
assert bands["Diana"] == "Mid"    # 78000
years = {r["name"]: r["tenure_years"] for r in emp_enhanced.collect()}
assert years["Noah"] >= 6, f"Noah should have 6+ years, got {years['Noah']}"
assert years["Ivy"] <= 2, f"Ivy should have <=2 years, got {years['Ivy']}"
print("Task 1.3 passed!")"""),

        md("""\
## Task 1.4: Handle nulls

Create a DataFrame `df_nulls` from this data: `[(1, "Alice", 100.0), (2, None, 200.0), (3, "Charlie", None), (4, None, None)]` with columns `id`, `name`, `value`.

Then create:
- `df_filled`: nulls filled — name with "Unknown", value with 0.0
- `df_complete`: only rows where ALL columns are non-null"""),

        code("""\
# YOUR CODE HERE
df_nulls = ...
df_filled = ...
df_complete = ...

# TEST — Do not modify
assert df_nulls.count() == 4
assert df_filled.filter(F.col("name").isNull()).count() == 0
assert df_filled.filter(F.col("value").isNull()).count() == 0
filled = {r["id"]: (r["name"], r["value"]) for r in df_filled.collect()}
assert filled[2] == ("Unknown", 200.0)
assert filled[4] == ("Unknown", 0.0)
assert df_complete.count() == 1  # only Alice has all values
print("Task 1.4 passed!")"""),

        md("""\
## Task 1.5: String and date functions

From `emp_df`, create `emp_strings` with columns:
- `name_upper`: name in upper case
- `dept_lower`: department in lower case
- `hire_year`: year extracted from hire_date
- `name_length`: length of the name"""),

        code("""\
# YOUR CODE HERE
emp_strings = ...

# TEST — Do not modify
row = emp_strings.filter(F.col("name") == "Alice").collect()[0]
assert row["name_upper"] == "ALICE"
assert row["dept_lower"] == "engineering"
assert row["hire_year"] == 2020
assert row["name_length"] == 5
print("Task 1.5 passed!")"""),

        md("## Cleanup"),
        code(cleanup),
    ]
    save(t, "tasks/task_01_dataframe_basics.ipynb")

    # --- SOLUTION ---
    s = nb()
    s.cells = [
        md("# Solution — Task 01: DataFrame Basics"),
        md("## Setup"),
        code(JAVA_SETUP),
        code(setup),

        md("## Solution 1.1: Create a DataFrame from a list"),
        code("""\
data = [
    ("Laptop", "Electronics", 1200.0),
    ("Desk", "Furniture", 350.0),
    ("Phone", "Electronics", 800.0),
    ("Chair", "Furniture", 250.0),
    ("Tablet", "Electronics", 500.0),
]
schema = StructType([
    StructField("name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", DoubleType(), True),
])
products_df = spark.createDataFrame(data, schema)
products_df.show()

# TEST — Do not modify
assert products_df.count() == 5, f"Expected 5 rows, got {products_df.count()}"
assert set(products_df.columns) == {"name", "category", "price"}
assert products_df.schema["price"].dataType == DoubleType()
rows = {r["name"]: r["price"] for r in products_df.collect()}
assert rows["Laptop"] == 1200.0
print("Task 1.1 passed!")"""),

        md("## Solution 1.2: Read CSV and basic stats"),
        code("""\
emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
eng_df = emp_df.filter(
    (F.col("department") == "Engineering") & (F.col("salary") > 95000)
).orderBy(F.col("salary").desc())
eng_df.show()

# TEST — Do not modify
assert emp_df.count() == 15, f"Expected 15, got {emp_df.count()}"
assert eng_df.count() == 3, f"Expected 3 high-paid engineers, got {eng_df.count()}"
salaries = [r["salary"] for r in eng_df.collect()]
assert salaries == sorted(salaries, reverse=True), "Should be sorted desc"
assert all(s > 95000 for s in salaries)
print("Task 1.2 passed!")"""),

        md("## Solution 1.3: Add computed columns"),
        code("""\
emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
emp_enhanced = emp_df.withColumn(
    "salary_band",
    F.when(F.col("salary") >= 95000, "Senior")
     .when(F.col("salary") >= 75000, "Mid")
     .otherwise("Junior")
).withColumn(
    "tenure_years",
    (F.datediff(F.lit("2024-01-01").cast("date"), F.col("hire_date")) / 365).cast("int")
)
emp_enhanced.select("name", "salary", "salary_band", "hire_date", "tenure_years").show()

# TEST — Do not modify
assert "salary_band" in emp_enhanced.columns
assert "tenure_years" in emp_enhanced.columns
bands = {r["name"]: r["salary_band"] for r in emp_enhanced.collect()}
assert bands["Noah"] == "Senior"   # 105000
assert bands["Charlie"] == "Junior"  # 72000
assert bands["Diana"] == "Mid"    # 78000
years = {r["name"]: r["tenure_years"] for r in emp_enhanced.collect()}
assert years["Noah"] >= 6, f"Noah should have 6+ years, got {years['Noah']}"
assert years["Ivy"] <= 2, f"Ivy should have <=2 years, got {years['Ivy']}"
print("Task 1.3 passed!")"""),

        md("## Solution 1.4: Handle nulls"),
        code("""\
df_nulls = spark.createDataFrame(
    [(1, "Alice", 100.0), (2, None, 200.0), (3, "Charlie", None), (4, None, None)],
    ["id", "name", "value"]
)
df_filled = df_nulls.fillna({"name": "Unknown", "value": 0.0})
df_complete = df_nulls.dropna()
df_filled.show()
df_complete.show()

# TEST — Do not modify
assert df_nulls.count() == 4
assert df_filled.filter(F.col("name").isNull()).count() == 0
assert df_filled.filter(F.col("value").isNull()).count() == 0
filled = {r["id"]: (r["name"], r["value"]) for r in df_filled.collect()}
assert filled[2] == ("Unknown", 200.0)
assert filled[4] == ("Unknown", 0.0)
assert df_complete.count() == 1  # only Alice has all values
print("Task 1.4 passed!")"""),

        md("## Solution 1.5: String and date functions"),
        code("""\
emp_strings = emp_df.select(
    "name",
    F.upper(F.col("name")).alias("name_upper"),
    F.lower(F.col("department")).alias("dept_lower"),
    F.year(F.col("hire_date")).alias("hire_year"),
    F.length(F.col("name")).alias("name_length"),
)
emp_strings.show()

# TEST — Do not modify
row = emp_strings.filter(F.col("name") == "Alice").collect()[0]
assert row["name_upper"] == "ALICE"
assert row["dept_lower"] == "engineering"
assert row["hire_year"] == 2020
assert row["name_length"] == 5
print("Task 1.5 passed!")"""),

        md("## Cleanup"),
        code(cleanup),
    ]
    save(s, "solutions/task_01_dataframe_basics_solution.ipynb")


# ==============================================================================
# TASK 02 — Aggregations and Joins (+ Solution)
# ==============================================================================
def task_02():
    setup = """\
import os
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder \\
    .master("local[*]") \\
    .appName("Task02") \\
    .config("spark.sql.shuffle.partitions", "4") \\
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))

emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
orders_df = spark.read.csv(os.path.join(FIXTURES, "orders.csv"), header=True, inferSchema=True)
customers_df = spark.read.csv(os.path.join(FIXTURES, "customers.csv"), header=True, inferSchema=True)
print("Setup complete.")"""

    cleanup = 'spark.stop()\nprint("All tasks done!")'

    # --- TASK ---
    t = nb()
    t.cells = [
        md("# Task 02 — Aggregations and Joins"),
        md("## Setup"),
        code(JAVA_SETUP),
        code(setup),

        md("""\
## Task 2.1: Department statistics

Create `dept_stats` with columns: `department`, `num_employees`, `avg_salary` (rounded to 0), `max_salary`.
Sort by `avg_salary` descending."""),

        code("""\
# YOUR CODE HERE
dept_stats = ...

# TEST — Do not modify
rows = {r["department"]: r for r in dept_stats.collect()}
assert len(rows) == 3
assert rows["Engineering"]["num_employees"] == 6
assert rows["Engineering"]["max_salary"] == 105000
collected = dept_stats.collect()
avg_salaries = [r["avg_salary"] for r in collected]
assert avg_salaries == sorted(avg_salaries, reverse=True), "Should be sorted by avg_salary desc"
print("Task 2.1 passed!")"""),

        md("""\
## Task 2.2: Join orders with customers

Create `order_details` by joining `orders_df` with `customers_df` on `customer_id` (inner join).
Select columns: `order_id`, `name` (customer name), `product`, `amount`, `city`.
Sort by `order_id`."""),

        code("""\
# YOUR CODE HERE
order_details = ...

# TEST — Do not modify
assert order_details.count() == 15
assert set(order_details.columns) == {"order_id", "name", "product", "amount", "city"}
first = order_details.collect()[0]
assert first["order_id"] == 101
assert first["name"] == "Alice"
assert first["city"] == "New York"
print("Task 2.2 passed!")"""),

        md("""\
## Task 2.3: Customer spending summary

Create `customer_spending` with: `customer_id`, `name`, `city`, `total_orders` (count), `total_spent` (sum of amount).
Sort by `total_spent` descending."""),

        code("""\
# YOUR CODE HERE
customer_spending = ...

# TEST — Do not modify
assert customer_spending.count() == 5
rows = {r["customer_id"]: r for r in customer_spending.collect()}
assert rows[1]["total_orders"] == 4
assert rows[1]["total_spent"] == 2750.0
spending_vals = [r["total_spent"] for r in customer_spending.collect()]
assert spending_vals == sorted(spending_vals, reverse=True)
print("Task 2.3 passed!")"""),

        md("""\
## Task 2.4: Pivot — spending by product per customer

Create `pivot_df` showing each customer's total spending per product.
Rows = customer_id, Columns = product names, Values = sum of amount.
Fill nulls with 0. Sort by customer_id."""),

        code("""\
# YOUR CODE HERE
pivot_df = ...

# TEST — Do not modify
rows = {r["customer_id"]: r for r in pivot_df.collect()}
assert rows[1]["Laptop"] == 1200.0
assert rows[1]["Headphones"] == 150.0
assert rows[4]["Laptop"] == 0  # or 0.0
assert pivot_df.count() == 5
print("Task 2.4 passed!")"""),

        md("""\
## Task 2.5: Product revenue ranking

Create `product_ranking` with: `product`, `total_revenue` (sum of amount), `order_count`.
Add a column `revenue_rank` using `row_number()` window function ordered by `total_revenue` descending."""),

        code("""\
from pyspark.sql import Window

# YOUR CODE HERE
product_ranking = ...

# TEST — Do not modify
rows = {r["product"]: r for r in product_ranking.collect()}
assert rows["Laptop"]["revenue_rank"] == 1  # highest revenue
assert rows["Laptop"]["total_revenue"] == 4950.0
assert rows["Laptop"]["order_count"] == 4
print("Task 2.5 passed!")"""),

        md("## Cleanup"),
        code(cleanup),
    ]
    save(t, "tasks/task_02_aggregations_joins.ipynb")

    # --- SOLUTION ---
    s = nb()
    s.cells = [
        md("# Solution — Task 02: Aggregations and Joins"),
        md("## Setup"),
        code(JAVA_SETUP),
        code(setup),

        md("## Solution 2.1: Department statistics"),
        code("""\
dept_stats = emp_df.groupBy("department").agg(
    F.count("*").alias("num_employees"),
    F.round(F.avg("salary"), 0).alias("avg_salary"),
    F.max("salary").alias("max_salary"),
).orderBy(F.col("avg_salary").desc())
dept_stats.show()

# TEST — Do not modify
rows = {r["department"]: r for r in dept_stats.collect()}
assert len(rows) == 3
assert rows["Engineering"]["num_employees"] == 6
assert rows["Engineering"]["max_salary"] == 105000
collected = dept_stats.collect()
avg_salaries = [r["avg_salary"] for r in collected]
assert avg_salaries == sorted(avg_salaries, reverse=True), "Should be sorted by avg_salary desc"
print("Task 2.1 passed!")"""),

        md("## Solution 2.2: Join orders with customers"),
        code("""\
order_details = orders_df.join(customers_df, on="customer_id", how="inner") \\
    .select("order_id", "name", "product", "amount", "city") \\
    .orderBy("order_id")
order_details.show()

# TEST — Do not modify
assert order_details.count() == 15
assert set(order_details.columns) == {"order_id", "name", "product", "amount", "city"}
first = order_details.collect()[0]
assert first["order_id"] == 101
assert first["name"] == "Alice"
assert first["city"] == "New York"
print("Task 2.2 passed!")"""),

        md("## Solution 2.3: Customer spending summary"),
        code("""\
customer_spending = orders_df.join(customers_df, on="customer_id") \\
    .groupBy("customer_id", "name", "city") \\
    .agg(
        F.count("*").alias("total_orders"),
        F.sum("amount").alias("total_spent"),
    ) \\
    .orderBy(F.col("total_spent").desc())
customer_spending.show()

# TEST — Do not modify
assert customer_spending.count() == 5
rows = {r["customer_id"]: r for r in customer_spending.collect()}
assert rows[1]["total_orders"] == 4
assert rows[1]["total_spent"] == 2750.0
spending_vals = [r["total_spent"] for r in customer_spending.collect()]
assert spending_vals == sorted(spending_vals, reverse=True)
print("Task 2.3 passed!")"""),

        md("## Solution 2.4: Pivot — spending by product per customer"),
        code("""\
pivot_df = orders_df.groupBy("customer_id") \\
    .pivot("product") \\
    .agg(F.sum("amount")) \\
    .fillna(0) \\
    .orderBy("customer_id")
pivot_df.show()

# TEST — Do not modify
rows = {r["customer_id"]: r for r in pivot_df.collect()}
assert rows[1]["Laptop"] == 1200.0
assert rows[1]["Headphones"] == 150.0
assert rows[4]["Laptop"] == 0  # or 0.0
assert pivot_df.count() == 5
print("Task 2.4 passed!")"""),

        md("## Solution 2.5: Product revenue ranking"),
        code("""\
from pyspark.sql import Window

product_agg = orders_df.groupBy("product").agg(
    F.sum("amount").alias("total_revenue"),
    F.count("*").alias("order_count"),
)

w = Window.orderBy(F.col("total_revenue").desc())
product_ranking = product_agg.withColumn("revenue_rank", F.row_number().over(w))
product_ranking.show()

# TEST — Do not modify
rows = {r["product"]: r for r in product_ranking.collect()}
assert rows["Laptop"]["revenue_rank"] == 1  # highest revenue
assert rows["Laptop"]["total_revenue"] == 4950.0
assert rows["Laptop"]["order_count"] == 4
print("Task 2.5 passed!")"""),

        md("## Cleanup"),
        code(cleanup),
    ]
    save(s, "solutions/task_02_aggregations_joins_solution.ipynb")


# ==============================================================================
# TASK 03 — Spark SQL and Window Functions (+ Solution)
# ==============================================================================
def task_03():
    setup = """\
import os
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import StringType

spark = SparkSession.builder \\
    .master("local[*]") \\
    .appName("Task03") \\
    .config("spark.sql.shuffle.partitions", "4") \\
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))

emp_df = spark.read.csv(os.path.join(FIXTURES, "employees.csv"), header=True, inferSchema=True)
orders_df = spark.read.csv(os.path.join(FIXTURES, "orders.csv"), header=True, inferSchema=True)
customers_df = spark.read.csv(os.path.join(FIXTURES, "customers.csv"), header=True, inferSchema=True)

emp_df.createOrReplaceTempView("employees")
orders_df.createOrReplaceTempView("orders")
customers_df.createOrReplaceTempView("customers")
print("Setup complete. Views registered: employees, orders, customers")"""

    cleanup = 'spark.stop()\nprint("All tasks done!")'

    # --- TASK ---
    t = nb()
    t.cells = [
        md("# Task 03 — Spark SQL and Window Functions"),
        md("## Setup"),
        code(JAVA_SETUP),
        code(setup),

        md("""\
## Task 3.1: SQL — Department stats

Write a SQL query that returns `department`, `cnt` (count of employees), `avg_sal` (average salary rounded to 0).
Order by `avg_sal` descending. Store result in `dept_sql`."""),

        code("""\
# YOUR CODE HERE
dept_sql = spark.sql(\"\"\"

\"\"\")

# TEST — Do not modify
rows = {r["department"]: r for r in dept_sql.collect()}
assert len(rows) == 3
assert rows["Engineering"]["cnt"] == 6
assert rows["Engineering"]["avg_sal"] == round(sum([95000,88000,102000,91000,98000,105000])/6)
print("Task 3.1 passed!")"""),

        md("""\
## Task 3.2: SQL with JOIN — Customer order summary

Write a SQL query joining orders and customers to get:
`name` (customer), `city`, `total_orders`, `total_amount`.
Order by `total_amount` DESC. Store in `cust_sql`."""),

        code("""\
# YOUR CODE HERE
cust_sql = spark.sql(\"\"\"

\"\"\")

# TEST — Do not modify
assert cust_sql.count() == 5
rows = {r["name"]: r for r in cust_sql.collect()}
assert rows["Alice"]["total_orders"] == 4
assert rows["Alice"]["total_amount"] == 2750.0
print("Task 3.2 passed!")"""),

        md("""\
## Task 3.3: Window — Salary rank within department

Using DataFrame API (not SQL), add a `dept_rank` column to `emp_df` ranking employees by salary within each department (highest = rank 1).
Store result in `emp_ranked`. Keep all original columns plus `dept_rank`."""),

        code("""\
# YOUR CODE HERE
emp_ranked = ...

# TEST — Do not modify
assert "dept_rank" in emp_ranked.columns
noah = emp_ranked.filter(F.col("name") == "Noah").collect()[0]
assert noah["dept_rank"] == 1  # highest in Engineering
alice = emp_ranked.filter(F.col("name") == "Alice").collect()[0]
assert alice["dept_rank"] == 4  # 4th in Engineering
print("Task 3.3 passed!")"""),

        md("""\
## Task 3.4: Window — Running total per customer

Add a `running_total` column to orders showing cumulative sum of `amount` per `customer_id`, ordered by `order_date`.
Store in `orders_running`. Keep all original columns plus `running_total`."""),

        code("""\
# YOUR CODE HERE
orders_running = ...

# TEST — Do not modify
cust1 = orders_running.filter(F.col("customer_id") == 1).orderBy("order_date").collect()
assert cust1[0]["running_total"] == 1200.0  # first order
assert cust1[1]["running_total"] == 1700.0  # 1200 + 500
assert cust1[2]["running_total"] == 1850.0  # + 150
assert cust1[3]["running_total"] == 2750.0  # + 900
print("Task 3.4 passed!")"""),

        md("""\
## Task 3.5: UDF — Categorize orders

Create a UDF `size_category` that takes an amount and returns:
- "Small" if amount < 300
- "Medium" if 300 <= amount < 1000
- "Large" if amount >= 1000

Apply it to create `orders_categorized` with a new `size` column."""),

        code("""\
# YOUR CODE HERE
orders_categorized = ...

# TEST — Do not modify
rows = orders_categorized.select("order_id", "amount", "size").collect()
sizes = {r["order_id"]: r["size"] for r in rows}
assert sizes[101] == "Large"    # 1200
assert sizes[102] == "Medium"   # 800
assert sizes[105] == "Small"    # 150
print("Task 3.5 passed!")"""),

        md("""\
## Task 3.6: SQL CTE — Employees above department average

Write a SQL query using a CTE to find employees whose salary is above their department's average.
Return: `name`, `department`, `salary`, `dept_avg` (rounded to 0).
Order by `salary` DESC. Store in `above_avg`."""),

        code("""\
# YOUR CODE HERE
above_avg = spark.sql(\"\"\"

\"\"\")

# TEST — Do not modify
names = {r["name"] for r in above_avg.collect()}
assert "Noah" in names    # 105k, eng avg ~96.5k
assert "Eve" in names     # 102k
assert "Karen" in names   # 98k
assert "Diana" in names   # 78k, mkt avg ~73.5k
assert "Leo" in names     # 75k
assert "Jack" in names    # 73k, sales avg ~69.8k
assert "Grace" in names   # 71k
assert "Charlie" not in names  # 72k < 73.5k
print("Task 3.6 passed!")"""),

        md("## Cleanup"),
        code(cleanup),
    ]
    save(t, "tasks/task_03_spark_sql_window.ipynb")

    # --- SOLUTION ---
    s = nb()
    s.cells = [
        md("# Solution — Task 03: Spark SQL and Window Functions"),
        md("## Setup"),
        code(JAVA_SETUP),
        code(setup),

        md("## Solution 3.1: SQL — Department stats"),
        code("""\
dept_sql = spark.sql(\"\"\"
    SELECT department,
           COUNT(*) as cnt,
           ROUND(AVG(salary), 0) as avg_sal
    FROM employees
    GROUP BY department
    ORDER BY avg_sal DESC
\"\"\")
dept_sql.show()

# TEST — Do not modify
rows = {r["department"]: r for r in dept_sql.collect()}
assert len(rows) == 3
assert rows["Engineering"]["cnt"] == 6
assert rows["Engineering"]["avg_sal"] == round(sum([95000,88000,102000,91000,98000,105000])/6)
print("Task 3.1 passed!")"""),

        md("## Solution 3.2: SQL with JOIN — Customer order summary"),
        code("""\
cust_sql = spark.sql(\"\"\"
    SELECT c.name, c.city,
           COUNT(*) as total_orders,
           SUM(o.amount) as total_amount
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.name, c.city
    ORDER BY total_amount DESC
\"\"\")
cust_sql.show()

# TEST — Do not modify
assert cust_sql.count() == 5
rows = {r["name"]: r for r in cust_sql.collect()}
assert rows["Alice"]["total_orders"] == 4
assert rows["Alice"]["total_amount"] == 2750.0
print("Task 3.2 passed!")"""),

        md("## Solution 3.3: Window — Salary rank within department"),
        code("""\
w = Window.partitionBy("department").orderBy(F.col("salary").desc())
emp_ranked = emp_df.withColumn("dept_rank", F.rank().over(w))
emp_ranked.select("name", "department", "salary", "dept_rank").show()

# TEST — Do not modify
assert "dept_rank" in emp_ranked.columns
noah = emp_ranked.filter(F.col("name") == "Noah").collect()[0]
assert noah["dept_rank"] == 1  # highest in Engineering
alice = emp_ranked.filter(F.col("name") == "Alice").collect()[0]
assert alice["dept_rank"] == 4  # 4th in Engineering
print("Task 3.3 passed!")"""),

        md("## Solution 3.4: Window — Running total per customer"),
        code("""\
w = Window.partitionBy("customer_id").orderBy("order_date")
orders_running = orders_df.withColumn("running_total", F.sum("amount").over(w))
orders_running.filter(F.col("customer_id") == 1).orderBy("order_date").show()

# TEST — Do not modify
cust1 = orders_running.filter(F.col("customer_id") == 1).orderBy("order_date").collect()
assert cust1[0]["running_total"] == 1200.0  # first order
assert cust1[1]["running_total"] == 1700.0  # 1200 + 500
assert cust1[2]["running_total"] == 1850.0  # + 150
assert cust1[3]["running_total"] == 2750.0  # + 900
print("Task 3.4 passed!")"""),

        md("## Solution 3.5: UDF — Categorize orders"),
        code("""\
@F.udf(returnType=StringType())
def size_category(amount):
    if amount is None:
        return "Unknown"
    if amount < 300:
        return "Small"
    elif amount < 1000:
        return "Medium"
    return "Large"

orders_categorized = orders_df.withColumn("size", size_category(F.col("amount")))
orders_categorized.select("order_id", "amount", "size").show()

# TEST — Do not modify
rows = orders_categorized.select("order_id", "amount", "size").collect()
sizes = {r["order_id"]: r["size"] for r in rows}
assert sizes[101] == "Large"    # 1200
assert sizes[102] == "Medium"   # 800
assert sizes[105] == "Small"    # 150
print("Task 3.5 passed!")"""),

        md("## Solution 3.6: SQL CTE — Employees above department average"),
        code("""\
above_avg = spark.sql(\"\"\"
    WITH dept_avg AS (
        SELECT department, AVG(salary) as avg_salary
        FROM employees
        GROUP BY department
    )
    SELECT e.name, e.department, e.salary,
           ROUND(d.avg_salary, 0) as dept_avg
    FROM employees e
    JOIN dept_avg d ON e.department = d.department
    WHERE e.salary > d.avg_salary
    ORDER BY e.salary DESC
\"\"\")
above_avg.show()

# TEST — Do not modify
names = {r["name"] for r in above_avg.collect()}
assert "Noah" in names    # 105k, eng avg ~96.5k
assert "Eve" in names     # 102k
assert "Karen" in names   # 98k
assert "Diana" in names   # 78k, mkt avg ~73.5k
assert "Leo" in names     # 75k
assert "Jack" in names    # 73k, sales avg ~69.8k
assert "Grace" in names   # 71k
assert "Charlie" not in names  # 72k < 73.5k
print("Task 3.6 passed!")"""),

        md("## Cleanup"),
        code(cleanup),
    ]
    save(s, "solutions/task_03_spark_sql_window_solution.ipynb")


if __name__ == "__main__":
    print("Generating Module 12 notebooks...")
    learning_01()
    learning_02()
    learning_03()
    task_01()
    task_02()
    task_03()
    print("\nAll notebooks generated!")
