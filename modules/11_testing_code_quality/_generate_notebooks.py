"""Generate all notebooks for Module 11."""
import nbformat as nbf
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def nb():
    """Create a new notebook."""
    return nbf.v4.new_notebook(metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    })


def md(src):
    return nbf.v4.new_markdown_cell(src.strip())


def code(src):
    return nbf.v4.new_code_cell(src.strip())


def save(notebook, path):
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        nbf.write(notebook, f)
    print(f"  Created {path}")


# ─────────────────────────────────────────────
# Learning 01: pytest Fundamentals
# ─────────────────────────────────────────────
def learning_01():
    n = nb()
    n.cells = [
        md("""# 01 — pytest Fundamentals

This notebook covers:
- Running pytest from a notebook
- Writing test functions and classes
- Rich assertions, `pytest.raises`
- Markers: `skip`, `xfail`, `parametrize`
- Test discovery rules"""),

        md("## Setup"),
        code("""import subprocess, sys, os, textwrap, tempfile, pathlib

# Helper: write a temp test file and run pytest on it
def run_pytest(test_code: str, extra_args: list[str] | None = None):
    \"\"\"Write test_code to a temp file and run pytest. Returns (stdout, returncode).\"\"\"
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "test_tmp.py"
        p.write_text(textwrap.dedent(test_code))
        cmd = [sys.executable, "-m", "pytest", str(p), "-v", "--tb=short", "--no-header"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        print(output)
        return result.returncode

print("Helper ready.")"""),

        md("## 1. Your First Test"),
        code("""rc = run_pytest('''
def test_addition():
    assert 1 + 1 == 2

def test_string():
    assert "hello".upper() == "HELLO"
''')
assert rc == 0, "Tests should pass"
print("\\nAll passed!")"""),

        md("""## 2. Rich Assertion Messages

pytest rewrites `assert` to show detailed diffs — no need for `assertEqual`."""),
        code("""# This test FAILS on purpose to show the diff output
rc = run_pytest('''
def test_list_diff():
    result = [1, 2, 3, 4]
    expected = [1, 2, 99, 4]
    assert result == expected
''')
assert rc != 0, "Should fail"
print("\\n^ Notice how pytest shows the exact index that differs.")"""),

        md("## 3. Testing Exceptions with `pytest.raises`"),
        code("""rc = run_pytest('''
import pytest

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def test_divide_ok():
    assert divide(10, 2) == 5.0

def test_divide_error():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(1, 0)

def test_type_error():
    with pytest.raises(TypeError):
        divide("a", "b")
''')
assert rc == 0
print("\\nException tests passed!")"""),

        md("""## 4. Markers

Markers let you tag tests and run subsets."""),
        code("""rc = run_pytest('''
import pytest

@pytest.mark.skip(reason="not implemented yet")
def test_future():
    assert False  # never runs

@pytest.mark.xfail(reason="known bug")
def test_known_bug():
    assert 1 == 2  # expected to fail

def test_normal():
    assert True
''')
assert rc == 0
print("\\nskip -> skipped, xfail -> xfail (expected failure), normal -> passed")"""),

        md("## 5. `parametrize` — Multiple Inputs, One Test"),
        code("""rc = run_pytest('''
import pytest

@pytest.mark.parametrize("x, y, expected", [
    (1, 1, 2),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(x, y, expected):
    assert x + y == expected
''')
assert rc == 0
print("\\n4 parameterized cases, all passed!")"""),

        md("""## 6. Test Classes

Group related tests. No `__init__` needed."""),
        code("""rc = run_pytest('''
class TestStringMethods:
    def test_upper(self):
        assert "hello".upper() == "HELLO"

    def test_split(self):
        assert "a,b,c".split(",") == ["a", "b", "c"]

    def test_strip(self):
        assert "  hi  ".strip() == "hi"
''')
assert rc == 0
print("\\nTest class with 3 methods — all passed!")"""),

        md("""## 7. Test Discovery Rules

pytest finds tests automatically by convention:
- Files: `test_*.py` or `*_test.py`
- Functions: `test_*`
- Classes: `Test*` (no `__init__`)

Files NOT matched: `helper.py`, `utils.py`, `conftest.py` (special purpose)."""),

        md("""## Key Takeaways

1. Plain `assert` gives rich diffs — no `assertEqual` needed
2. `pytest.raises(ExcType, match=...)` for exception testing
3. `@pytest.mark.parametrize` eliminates copy-paste tests
4. Markers (`skip`, `xfail`, custom) control which tests run
5. Test discovery is convention-based — follow naming rules"""),
    ]
    save(n, "learning/01_pytest_fundamentals.ipynb")


# ─────────────────────────────────────────────
# Learning 02: Fixtures & Parametrize
# ─────────────────────────────────────────────
def learning_02():
    n = nb()
    n.cells = [
        md("""# 02 — Fixtures & Parametrize

This notebook covers:
- `@pytest.fixture` with scopes
- Built-in fixtures: `tmp_path`, `capsys`, `monkeypatch`
- `conftest.py` for shared fixtures
- Advanced parametrize patterns
- Fixture factories"""),

        md("## Setup"),
        code("""import subprocess, sys, textwrap, tempfile, pathlib

def run_pytest(test_code: str, extra_args: list[str] | None = None, extra_files: dict[str, str] | None = None):
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        p = td_path / "test_tmp.py"
        p.write_text(textwrap.dedent(test_code))
        if extra_files:
            for name, content in extra_files.items():
                (td_path / name).write_text(textwrap.dedent(content))
        cmd = [sys.executable, "-m", "pytest", str(td), "-v", "--tb=short", "--no-header"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout + result.stderr)
        return result.returncode

print("Helper ready.")"""),

        md("## 1. Basic Fixtures"),
        code("""rc = run_pytest('''
import pytest

@pytest.fixture
def sample_list():
    \"\"\"Provides a fresh list for each test.\"\"\"
    return [1, 2, 3, 4, 5]

def test_length(sample_list):
    assert len(sample_list) == 5

def test_sum(sample_list):
    assert sum(sample_list) == 15

def test_mutate(sample_list):
    sample_list.append(6)
    assert len(sample_list) == 6  # 6 here...

def test_still_fresh(sample_list):
    assert len(sample_list) == 5  # ...but 5 here — fixture is recreated
''')
assert rc == 0
print("Each test gets a FRESH copy of the fixture (function scope by default).")"""),

        md("## 2. Fixture with Teardown (`yield`)"),
        code("""rc = run_pytest('''
import pytest

log = []

@pytest.fixture
def resource():
    log.append("setup")
    yield "the_resource"
    log.append("teardown")

def test_one(resource):
    assert resource == "the_resource"
    log.append("test_one_ran")

def test_two(resource):
    assert resource == "the_resource"
    log.append("test_two_ran")

def test_log():
    # After test_one: setup, test_one_ran, teardown
    # After test_two: setup, test_two_ran, teardown
    assert log == ["setup", "test_one_ran", "teardown", "setup", "test_two_ran", "teardown"]
''')
assert rc == 0
print("yield separates setup from teardown — cleanup is guaranteed.")"""),

        md("## 3. Fixture Scopes"),
        code("""rc = run_pytest('''
import pytest

call_count = {"func": 0, "mod": 0}

@pytest.fixture
def func_fixture():
    call_count["func"] += 1
    return call_count["func"]

@pytest.fixture(scope="module")
def mod_fixture():
    call_count["mod"] += 1
    return call_count["mod"]

def test_a(func_fixture, mod_fixture):
    assert func_fixture == 1  # fresh per test
    assert mod_fixture == 1   # shared across module

def test_b(func_fixture, mod_fixture):
    assert func_fixture == 2  # incremented again
    assert mod_fixture == 1   # still 1 — module scope
''')
assert rc == 0
print("function scope: per test. module scope: once per file. session scope: once per run.")"""),

        md("## 4. Built-in Fixture: `tmp_path`"),
        code("""rc = run_pytest('''
def test_write_file(tmp_path):
    # tmp_path is a pathlib.Path to a unique temp directory
    f = tmp_path / "data.txt"
    f.write_text("hello world")
    assert f.read_text() == "hello world"
    assert f.exists()

def test_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,age" + chr(10) + "Alice,30" + chr(10) + "Bob,25")
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
''')
assert rc == 0
print("tmp_path: auto-cleaned temp dir, unique per test.")"""),

        md("## 5. Built-in Fixture: `monkeypatch`"),
        code("""rc = run_pytest('''
import os

def get_env_setting():
    return os.environ.get("MY_SETTING", "default")

def test_default():
    assert get_env_setting() == "default"

def test_override(monkeypatch):
    monkeypatch.setenv("MY_SETTING", "custom_value")
    assert get_env_setting() == "custom_value"

def test_still_default():
    # monkeypatch auto-reverts after the test
    assert get_env_setting() == "default"
''')
assert rc == 0
print("monkeypatch sets env vars, attributes, dict items — auto-reverted after test.")"""),

        md("## 6. `conftest.py` — Shared Fixtures"),
        code("""conftest_code = '''
import pytest

@pytest.fixture
def sample_texts():
    return ["hello world", "foo bar", "test input"]

@pytest.fixture
def empty_text():
    return ""
'''

test_code = '''
def test_texts_count(sample_texts):
    assert len(sample_texts) == 3

def test_empty(empty_text):
    assert empty_text == ""
'''

rc = run_pytest(test_code, extra_files={"conftest.py": conftest_code})
assert rc == 0
print("Fixtures from conftest.py are auto-discovered — no imports needed.")"""),

        md("## 7. Advanced Parametrize"),
        code("""rc = run_pytest('''
import pytest

# Using pytest.param for IDs and markers
@pytest.mark.parametrize("text, expected", [
    pytest.param("hello world", 2, id="two_words"),
    pytest.param("", 0, id="empty"),
    pytest.param("one", 1, id="single_word"),
    pytest.param("  spaces  between  ", 2, id="extra_spaces"),
])
def test_word_count(text, expected):
    words = text.split() if text.strip() else []
    assert len(words) == expected

# Stacking parametrize for combinations
@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", [10, 20])
def test_multiply(a, b):
    assert a * b > 0  # 4 combinations: (1,10), (1,20), (2,10), (2,20)
''')
assert rc == 0
print("pytest.param(id=...) makes output readable. Stacked parametrize = combinatorial.")"""),

        md("## 8. Fixture Factories"),
        code("""rc = run_pytest('''
import pytest

@pytest.fixture
def make_user():
    \"\"\"Factory fixture: returns a function that creates users.\"\"\"
    created = []
    def _make(name="Alice", age=30):
        user = {"name": name, "age": age}
        created.append(user)
        return user
    yield _make
    # teardown: clean up all created users
    created.clear()

def test_default_user(make_user):
    user = make_user()
    assert user == {"name": "Alice", "age": 30}

def test_custom_user(make_user):
    user = make_user("Bob", 25)
    assert user["name"] == "Bob"

def test_multiple(make_user):
    u1 = make_user("A", 1)
    u2 = make_user("B", 2)
    assert u1 != u2
''')
assert rc == 0
print("Factory fixtures return callables — flexible and reusable.")"""),

        md("""## Key Takeaways

1. **Fixtures** = dependency injection for tests. Request only what you need.
2. **Scopes**: `function` (default, safe) → `module` → `session` (fast, shared)
3. **`yield`** for setup+teardown in one place
4. **Built-ins**: `tmp_path`, `capsys`, `monkeypatch` — use them!
5. **`conftest.py`** shares fixtures without imports
6. **Factories** for when you need multiple instances per test"""),
    ]
    save(n, "learning/02_fixtures_and_parametrize.ipynb")


# ─────────────────────────────────────────────
# Learning 03: Mocking & Patching
# ─────────────────────────────────────────────
def learning_03():
    n = nb()
    n.cells = [
        md("""# 03 — Mocking & Patching

This notebook covers:
- `unittest.mock.patch` and `MagicMock`
- `monkeypatch` for env vars and attributes
- `responses` library for HTTP mocking
- When to mock vs. when to use real objects"""),

        md("## Setup"),
        code("""import subprocess, sys, textwrap, tempfile, pathlib

def run_pytest(test_code: str, extra_args: list[str] | None = None, extra_files: dict[str, str] | None = None):
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        p = td_path / "test_tmp.py"
        p.write_text(textwrap.dedent(test_code))
        if extra_files:
            for name, content in extra_files.items():
                (td_path / name).write_text(textwrap.dedent(content))
        cmd = [sys.executable, "-m", "pytest", str(td), "-v", "--tb=short", "--no-header"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout + result.stderr)
        return result.returncode

print("Helper ready.")"""),

        md("""## 1. MagicMock Basics

`MagicMock` creates objects that record how they're used."""),
        code("""from unittest.mock import MagicMock

# Create a mock
model = MagicMock()

# Call it — doesn't crash, returns another MagicMock
result = model.predict(["hello"])
print(f"result type: {type(result)}")

# Configure return value
model.predict.return_value = [1, 0, 1]
assert model.predict(["any", "input"]) == [1, 0, 1]

# Check it was called
model.predict.assert_called()
model.predict.assert_called_with(["any", "input"])
print(f"call count: {model.predict.call_count}")

print("\\nMagicMock: records calls, configurable return values, never crashes.")"""),

        md("## 2. `patch` — Replacing Objects During Tests"),
        code("""# Module to test: a function that calls an external API
module_code = '''
import requests

def fetch_user(user_id: int) -> dict:
    resp = requests.get(f"https://api.example.com/users/{user_id}")
    resp.raise_for_status()
    return resp.json()
'''

test_code = '''
from unittest.mock import patch, MagicMock
from mymodule import fetch_user

@patch("mymodule.requests.get")
def test_fetch_user(mock_get):
    # Configure the mock
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 1, "name": "Alice"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Call the function — it uses mock instead of real requests
    result = fetch_user(1)

    assert result == {"id": 1, "name": "Alice"}
    mock_get.assert_called_once_with("https://api.example.com/users/1")
'''

rc = run_pytest(test_code, extra_files={"mymodule.py": module_code})
assert rc == 0
print("patch replaces requests.get with a mock — no real HTTP call!")"""),

        md("""## 3. The Patch Target Rule

**Critical**: patch where the name is *looked up*, not where it's *defined*.

```python
# mymodule.py
from datetime import datetime  # <-- datetime is now in mymodule's namespace

# WRONG: @patch("datetime.datetime")
# RIGHT: @patch("mymodule.datetime")
```"""),
        code("""module_code = '''
from datetime import datetime

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    return "Good afternoon"
'''

test_code = '''
from unittest.mock import patch
from datetime import datetime
from mymod import get_greeting

@patch("mymod.datetime")
def test_morning(mock_dt):
    mock_dt.now.return_value = datetime(2024, 1, 1, 8, 0)
    assert get_greeting() == "Good morning"

@patch("mymod.datetime")
def test_afternoon(mock_dt):
    mock_dt.now.return_value = datetime(2024, 1, 1, 15, 0)
    assert get_greeting() == "Good afternoon"
'''

rc = run_pytest(test_code, extra_files={"mymod.py": module_code})
assert rc == 0
print("Patched mymod.datetime (where it's looked up), not datetime.datetime!")"""),

        md("## 4. `monkeypatch` for Environment & Attributes"),
        code("""rc = run_pytest('''
import os

def get_model_path():
    return os.environ.get("MODEL_PATH", "/default/model.bin")

def test_default_path():
    assert get_model_path() == "/default/model.bin"

def test_custom_path(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/custom/model.bin")
    assert get_model_path() == "/custom/model.bin"

def test_unset(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    assert get_model_path() == "/default/model.bin"

# monkeypatch for attributes
class Config:
    DEBUG = False

def test_debug_mode(monkeypatch):
    monkeypatch.setattr(Config, "DEBUG", True)
    assert Config.DEBUG is True
''')
assert rc == 0
print("monkeypatch: env vars, attributes, dict items — auto-reverted after test.")"""),

        md("## 5. `responses` — HTTP Mocking Made Easy"),
        code("""rc = run_pytest('''
import responses
import requests

@responses.activate
def test_api_call():
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"items": [1, 2, 3]},
        status=200,
    )

    resp = requests.get("https://api.example.com/data")
    assert resp.status_code == 200
    assert resp.json()["items"] == [1, 2, 3]

@responses.activate
def test_api_error():
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"error": "not found"},
        status=404,
    )

    resp = requests.get("https://api.example.com/data")
    assert resp.status_code == 404

@responses.activate
def test_multiple_calls():
    # First call returns 500, second returns 200
    responses.add(responses.GET, "https://api.example.com/data", status=500)
    responses.add(responses.GET, "https://api.example.com/data", json={"ok": True}, status=200)

    r1 = requests.get("https://api.example.com/data")
    r2 = requests.get("https://api.example.com/data")
    assert r1.status_code == 500
    assert r2.status_code == 200
''')
assert rc == 0
print("responses: clean HTTP mocking with @responses.activate decorator.")"""),

        md("""## 6. When to Mock vs. When to Use Real Objects

| Mock | Don't Mock |
|------|------------|
| HTTP APIs, LLM calls | Pure functions |
| Database in unit tests | Simple data classes |
| File system (sometimes) | Your own utilities |
| Time, randomness | Anything fast & deterministic |

**Rule of thumb**: mock at boundaries (network, disk, external services). Test your own code with real objects."""),

        md("""## Key Takeaways

1. **MagicMock**: records calls, configurable returns, never crashes
2. **`@patch("where.its.looked.up")`** — not where it's defined
3. **`monkeypatch`** for simple value swaps (env vars, attributes)
4. **`mock.patch`** when you need to verify interactions (assert_called)
5. **`responses`** for clean HTTP mocking
6. Mock at **boundaries**, test internals with real objects"""),
    ]
    save(n, "learning/03_mocking_and_patching.ipynb")


# ─────────────────────────────────────────────
# Learning 04: Testing ML & APIs
# ─────────────────────────────────────────────
def learning_04():
    n = nb()
    n.cells = [
        md("""# 04 — Testing ML Pipelines & APIs

This notebook covers:
- Testing ML pipelines with reproducible seeds
- FastAPI TestClient (with lifespan!)
- Mocking ML model predictions
- Coverage reports"""),

        md("## Setup"),
        code("""import subprocess, sys, textwrap, tempfile, pathlib, os

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(".")),
                        "11_testing_code_quality", "fixtures", "input")
# Fallback for running from within the module dir
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.join(os.path.abspath("."), "fixtures", "input")
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.join(os.path.abspath(".."), "fixtures", "input")

print(f"Fixtures dir: {FIXTURES}")
assert os.path.exists(FIXTURES), f"Fixtures not found at {FIXTURES}"

sys.path.insert(0, FIXTURES)"""),

        md("## 1. Testing the Sample Module"),
        code("""from sample_module import clean_text, tokenize, count_words, extract_emails, is_palindrome

# Direct testing — no pytest needed for exploration
assert clean_text("  hello   world  ") == "hello world"
assert tokenize("Hello World") == ["hello", "world"]
assert tokenize("") == []
assert count_words("the cat sat on the mat") == {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
assert extract_emails("email me at user@example.com or admin@test.org") == ["user@example.com", "admin@test.org"]
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("hello") == False

print("All sample_module functions work correctly!")"""),

        md("## 2. Testing ML Pipeline — Reproducibility"),
        code("""import numpy as np
from sample_ml_pipeline import (
    create_pipeline, train_pipeline, predict, evaluate,
    SAMPLE_TEXTS, SAMPLE_LABELS, preprocess_texts
)

# Train with fixed seed — results should be reproducible
np.random.seed(42)
pipe1 = create_pipeline(max_features=100)
train_pipeline(pipe1, SAMPLE_TEXTS, SAMPLE_LABELS)
preds1 = predict(pipe1, SAMPLE_TEXTS)

np.random.seed(42)
pipe2 = create_pipeline(max_features=100)
train_pipeline(pipe2, SAMPLE_TEXTS, SAMPLE_LABELS)
preds2 = predict(pipe2, SAMPLE_TEXTS)

assert np.array_equal(preds1, preds2), "Same seed should give same predictions"
print(f"Predictions (run 1): {preds1}")
print(f"Predictions (run 2): {preds2}")
print("Reproducibility confirmed!")"""),

        code("""# Evaluate metrics
metrics = evaluate(SAMPLE_LABELS, preds1.tolist())
print(f"Metrics: {metrics}")
assert 0 <= metrics["accuracy"] <= 1
assert 0 <= metrics["f1_macro"] <= 1
print("Metrics are in valid range.")"""),

        md("""## 3. Testing FastAPI with TestClient

**Important**: Starlette 0.50+ requires context manager for lifespan events."""),
        code("""from fastapi.testclient import TestClient
from sample_fastapi_app import app

# CORRECT: context manager triggers lifespan (model loads)
with TestClient(app) as client:
    # Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    print(f"Health: {data}")

    # Predict
    resp = client.post("/predict", json={"text": "This is great and amazing"})
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["label"] in ("positive", "negative", "neutral")
    assert 0 <= pred["score"] <= 1
    print(f"Predict: {pred}")

    # Batch predict
    resp = client.post("/predict/batch", json={"texts": ["good", "bad", "ok"]})
    assert resp.status_code == 200
    batch = resp.json()
    assert len(batch["predictions"]) == 3
    print(f"Batch: {len(batch['predictions'])} predictions")

print("\\nAll API tests passed!")"""),

        md("## 4. Testing Validation Errors"),
        code("""with TestClient(app) as client:
    # Empty text — should fail validation (min_length=1)
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422
    print(f"Empty text -> {resp.status_code} (validation error)")

    # Missing field
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
    print(f"Missing field -> {resp.status_code}")

    # Wrong type
    resp = client.post("/predict", json={"text": 123})
    assert resp.status_code == 422
    print(f"Wrong type -> {resp.status_code}")

print("\\nValidation error handling works correctly!")"""),

        md("## 5. Mocking the ML Model in API Tests"),
        code("""from unittest.mock import MagicMock
from sample_fastapi_app import ml_models

with TestClient(app) as client:
    # Replace the real model with a mock
    mock_model = MagicMock()
    mock_model.predict.return_value = ("positive", 0.99)
    ml_models["sentiment"] = mock_model

    resp = client.post("/predict", json={"text": "anything"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "positive"
    assert resp.json()["score"] == 0.99

    # Verify the mock was called
    mock_model.predict.assert_called_once_with("anything")

print("Mock model works — we control predictions in tests!")"""),

        md("## 6. Running pytest with Coverage"),
        code("""# Write a proper test file and run with coverage
test_content = textwrap.dedent(f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import clean_text, tokenize, extract_emails

def test_clean():
    assert clean_text("  hi  there  ") == "hi there"

def test_tokenize():
    assert tokenize("Hello World") == ["hello", "world"]

def test_tokenize_empty():
    assert tokenize("") == []

def test_emails():
    assert extract_emails("a@b.com") == ["a@b.com"]
    assert extract_emails("no email") == []
''')

with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "test_cov.py"
    p.write_text(test_content)
    cmd = [
        sys.executable, "-m", "pytest", str(p), "-v",
        f"--cov={FIXTURES}/sample_module.py",  # not a package, but works for demo
        "--cov-report=term-missing",
        "--tb=short", "--no-header",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)"""),

        md("""## Key Takeaways

1. **Reproducibility**: always set `np.random.seed()` for ML tests
2. **TestClient context manager**: required for lifespan events (Starlette 0.50+)
3. **Mock ML models** to control predictions and speed up tests
4. **Test validation errors** (422) — they're part of your API contract
5. **Coverage**: `--cov-report=term-missing` shows exactly which lines need tests"""),
    ]
    save(n, "learning/04_testing_ml_and_apis.ipynb")


# ─────────────────────────────────────────────
# Task 01: pytest Basics
# ─────────────────────────────────────────────
def task_01():
    n = nb()
    n.cells = [
        md("""# Task 01 — pytest Basics

Write tests for functions in `sample_module.py`. Each cell has instructions and assert-based validation.

Functions to test: `clean_text`, `tokenize`, `count_words`, `extract_emails`, `truncate`, `is_palindrome`, `mask_pii`"""),

        md("## Setup"),
        code("""import subprocess, sys, os, textwrap, tempfile, pathlib

FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
sys.path.insert(0, FIXTURES)

from sample_module import clean_text, tokenize, count_words, extract_emails, truncate, is_palindrome, mask_pii

def run_pytest(test_code: str, extra_args: list[str] | None = None):
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "test_tmp.py"
        p.write_text(textwrap.dedent(test_code))
        cmd = [sys.executable, "-m", "pytest", str(p), "-v", "--tb=short", "--no-header"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout + result.stderr)
        return result.returncode

print("Setup complete. Fixtures at:", FIXTURES)"""),

        md("""## Task 1.1: Test `clean_text`

Write a test file with at least 3 tests for `clean_text`:
- Normal text with extra spaces
- Empty string
- Text with tabs and newlines"""),
        code("""# YOUR CODE HERE
# Write a string containing pytest test functions for clean_text
# The string will be written to a temp file and run with pytest.
# You must import clean_text inside the test string since it runs in a subprocess.

test_clean_text = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import clean_text

# Write your tests here:
# def test_...():
#     assert clean_text(...) == ...
'''

# TEST — Do not modify
rc = run_pytest(test_clean_text)
assert rc == 0, "Tests should pass"
print("Task 1.1 passed!")"""),

        md("""## Task 1.2: Test `tokenize` with `parametrize`

Use `@pytest.mark.parametrize` to test `tokenize` with at least 5 different inputs covering:
- Normal text
- Empty string
- Single word
- Mixed case
- Extra whitespace"""),
        code("""# YOUR CODE HERE
test_tokenize = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")
from sample_module import tokenize

# Write parametrized tests:
# @pytest.mark.parametrize("text, expected", [...])
# def test_tokenize(text, expected):
#     ...
'''

# TEST — Do not modify
rc = run_pytest(test_tokenize)
assert rc == 0, "Tests should pass"
print("Task 1.2 passed!")"""),

        md("""## Task 1.3: Test Exceptions with `pytest.raises`

Write tests that verify:
- `clean_text(123)` raises `TypeError`
- `tokenize(None)` raises `TypeError`
- `truncate("hello", max_length=-1)` raises `ValueError`"""),
        code("""# YOUR CODE HERE
test_exceptions = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")
from sample_module import clean_text, tokenize, truncate

# Write exception tests:
# def test_clean_text_type_error():
#     with pytest.raises(TypeError):
#         ...
'''

# TEST — Do not modify
rc = run_pytest(test_exceptions)
assert rc == 0, "Tests should pass"
print("Task 1.3 passed!")"""),

        md("""## Task 1.4: Test `extract_emails` and `mask_pii`

Write tests for:
- `extract_emails` with 0, 1, and multiple emails
- `mask_pii` replacing emails and phone numbers"""),
        code("""# YOUR CODE HERE
test_pii = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import extract_emails, mask_pii

# Write your tests:
# def test_extract_no_emails():
#     assert extract_emails("no email here") == []
#
# def test_mask_pii_email():
#     assert mask_pii("contact user@example.com") == "contact [EMAIL]"
'''

# TEST — Do not modify
rc = run_pytest(test_pii)
assert rc == 0, "Tests should pass"
print("Task 1.4 passed!")"""),

        md("""## Task 1.5: Test `is_palindrome` and `truncate`

Write at least 2 tests for each function."""),
        code("""# YOUR CODE HERE
test_misc = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import is_palindrome, truncate

# Write your tests here
'''

# TEST — Do not modify
rc = run_pytest(test_misc)
assert rc == 0, "Tests should pass"
print("Task 1.5 passed!")"""),
    ]
    save(n, "tasks/task_01_pytest_basics.ipynb")


# ─────────────────────────────────────────────
# Task 02: Fixtures & Parametrize
# ─────────────────────────────────────────────
def task_02():
    n = nb()
    n.cells = [
        md("""# Task 02 — Fixtures & Parametrize

Practice writing fixtures, using `conftest.py`, and advanced parametrize patterns."""),

        md("## Setup"),
        code("""import subprocess, sys, os, textwrap, tempfile, pathlib

FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
sys.path.insert(0, FIXTURES)

def run_pytest(test_code: str, extra_files: dict[str, str] | None = None, extra_args: list[str] | None = None):
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        p = td_path / "test_tmp.py"
        p.write_text(textwrap.dedent(test_code))
        if extra_files:
            for name, content in extra_files.items():
                (td_path / name).write_text(textwrap.dedent(content))
        cmd = [sys.executable, "-m", "pytest", str(td), "-v", "--tb=short", "--no-header"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout + result.stderr)
        return result.returncode

print("Setup complete.")"""),

        md("""## Task 2.1: Create Fixtures for Text Processing

Write a `conftest.py` with these fixtures:
- `sample_texts` — returns a list of 5 diverse text strings
- `empty_text` — returns ""
- `text_with_emails` — returns a string containing 2+ email addresses

Then write tests that use them."""),
        code("""# YOUR CODE HERE
conftest_code = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")

# @pytest.fixture
# def sample_texts():
#     ...
'''

test_code = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import tokenize, extract_emails

# def test_tokenize_samples(sample_texts):
#     for text in sample_texts:
#         tokens = tokenize(text)
#         assert isinstance(tokens, list)
#
# def test_extract_from_email_text(text_with_emails):
#     emails = extract_emails(text_with_emails)
#     assert len(emails) >= 2
'''

# TEST — Do not modify
rc = run_pytest(test_code, extra_files={"conftest.py": conftest_code})
assert rc == 0, "Tests should pass"
print("Task 2.1 passed!")"""),

        md("""## Task 2.2: Parametrize ML Preprocessing

Use `@pytest.mark.parametrize` to test `preprocess_texts` from `sample_ml_pipeline.py` with different inputs.

Test cases should include:
- Normal text → lowercased and stripped
- Text with URLs → URLs removed
- Text with extra whitespace → normalized"""),
        code("""# YOUR CODE HERE
test_preprocess = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")
from sample_ml_pipeline import preprocess_texts

# @pytest.mark.parametrize("texts, expected", [
#     (["Hello World"], ["hello world"]),
#     (["Check https://example.com out"], ["check out"]),
#     ...
# ])
# def test_preprocess(texts, expected):
#     assert preprocess_texts(texts) == expected
'''

# TEST — Do not modify
rc = run_pytest(test_preprocess)
assert rc == 0, "Tests should pass"
print("Task 2.2 passed!")"""),

        md("""## Task 2.3: Use `tmp_path` for File-Based Tests

Write a test that:
1. Creates a CSV file in `tmp_path` with text data
2. Reads it back with pandas
3. Applies `clean_text` to each row
4. Asserts the results are correct"""),
        code("""# YOUR CODE HERE
test_file = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
import pandas as pd
from sample_module import clean_text

# def test_csv_processing(tmp_path):
#     # 1. Create CSV
#     csv_path = tmp_path / "data.csv"
#     csv_path.write_text("text" + chr(10) + "  hello   world  " + chr(10) + "  foo   bar  " + chr(10))
#     # 2. Read with pandas
#     df = pd.read_csv(csv_path)
#     # 3. Apply clean_text
#     df["cleaned"] = df["text"].apply(clean_text)
#     # 4. Assert
#     assert df["cleaned"].tolist() == ["hello world", "foo bar"]
'''

# TEST — Do not modify
rc = run_pytest(test_file)
assert rc == 0, "Tests should pass"
print("Task 2.3 passed!")"""),

        md("""## Task 2.4: Fixture with Yield (Setup + Teardown)

Write a fixture that:
1. Creates a trained ML pipeline (setup)
2. Yields it
3. Logs "pipeline cleaned up" to a list (teardown)

Then write a test using this fixture."""),
        code("""# YOUR CODE HERE
test_yield = f'''
import sys, pytest, numpy as np
sys.path.insert(0, "{FIXTURES}")
from sample_ml_pipeline import create_pipeline, train_pipeline, predict, SAMPLE_TEXTS, SAMPLE_LABELS

cleanup_log = []

# @pytest.fixture
# def trained_pipe():
#     np.random.seed(42)
#     pipe = create_pipeline(max_features=100)
#     train_pipeline(pipe, SAMPLE_TEXTS, SAMPLE_LABELS)
#     yield pipe
#     cleanup_log.append("pipeline cleaned up")

# def test_predict(trained_pipe):
#     preds = predict(trained_pipe, ["great product"])
#     assert len(preds) == 1

# def test_cleanup_happened():
#     assert "pipeline cleaned up" in cleanup_log
'''

# TEST — Do not modify
rc = run_pytest(test_yield)
assert rc == 0, "Tests should pass"
print("Task 2.4 passed!")"""),
    ]
    save(n, "tasks/task_02_fixtures_parametrize.ipynb")


# ─────────────────────────────────────────────
# Task 03: Testing ML API
# ─────────────────────────────────────────────
def task_03():
    n = nb()
    n.cells = [
        md("""# Task 03 — Testing ML API End-to-End

Test the FastAPI app from `sample_fastapi_app.py` using TestClient and mocking."""),

        md("## Setup"),
        code("""import sys, os

FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
sys.path.insert(0, FIXTURES)

from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from sample_fastapi_app import app, ml_models

print("Setup complete.")"""),

        md("""## Task 3.1: Test Health Endpoint

Write tests that verify:
- `/health` returns 200
- Response has `status` = "ok" and `model_loaded` = True
- Use `TestClient` with context manager (required for lifespan!)"""),
        code("""# YOUR CODE HERE

# with TestClient(app) as client:
#     ...

# TEST — Do not modify
with TestClient(app) as client:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True
print("Task 3.1 passed!")"""),

        md("""## Task 3.2: Test Predict Endpoint

Write tests for `/predict`:
- Positive text returns a valid prediction
- Response has `label`, `score`, and `text` fields
- Score is between 0 and 1"""),
        code("""# YOUR CODE HERE
# Test the /predict endpoint with a positive text

# TEST — Do not modify
with TestClient(app) as client:
    resp = client.post("/predict", json={"text": "This is amazing and great"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "score" in data
    assert "text" in data
    assert 0 <= data["score"] <= 1
    assert data["text"] == "This is amazing and great"
print("Task 3.2 passed!")"""),

        md("""## Task 3.3: Test Validation Errors

Write tests that verify the API returns 422 for:
- Empty text `{"text": ""}`
- Missing text field `{}`
- Text too long (over 5000 chars)"""),
        code("""# YOUR CODE HERE

# TEST — Do not modify
with TestClient(app) as client:
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 5001}).status_code == 422
print("Task 3.3 passed!")"""),

        md("""## Task 3.4: Test Batch Predict

Write tests for `/predict/batch`:
- Send 3 texts, get 3 predictions back
- Each prediction has the required fields"""),
        code("""# YOUR CODE HERE

# TEST — Do not modify
with TestClient(app) as client:
    texts = ["good product", "terrible service", "it was okay"]
    resp = client.post("/predict/batch", json={"texts": texts})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert "label" in p and "score" in p and "text" in p
print("Task 3.4 passed!")"""),

        md("""## Task 3.5: Mock the ML Model

Replace the sentiment model with a `MagicMock` that always returns `("positive", 0.99)`.
Verify that the mock is used and the response matches."""),
        code("""# YOUR CODE HERE
# Hint:
# mock_model = MagicMock()
# mock_model.predict.return_value = ("positive", 0.99)
# ml_models["sentiment"] = mock_model
# Then make a request and verify the response

# TEST — Do not modify
with TestClient(app) as client:
    mock_model = MagicMock()
    mock_model.predict.return_value = ("positive", 0.99)
    ml_models["sentiment"] = mock_model

    resp = client.post("/predict", json={"text": "any text here"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "positive"
    assert resp.json()["score"] == 0.99
    mock_model.predict.assert_called_once_with("any text here")
print("Task 3.5 passed!")"""),

        md("""## Task 3.6: Test Model Not Loaded (503)

Clear `ml_models` so there's no model, then verify `/predict` returns 503."""),
        code("""# YOUR CODE HERE
# Hint: manually clear ml_models after lifespan loads
# ml_models.clear()

# TEST — Do not modify
with TestClient(app) as client:
    ml_models.clear()
    resp = client.post("/predict", json={"text": "hello"})
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()
print("Task 3.6 passed!")"""),
    ]
    save(n, "tasks/task_03_testing_ml_api.ipynb")


# ─────────────────────────────────────────────
# Solution 01
# ─────────────────────────────────────────────
def solution_01():
    n = nb()
    n.cells = [
        md("""# Solution — Task 01: pytest Basics"""),

        md("## Setup"),
        code("""import subprocess, sys, os, textwrap, tempfile, pathlib

FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
sys.path.insert(0, FIXTURES)

from sample_module import clean_text, tokenize, count_words, extract_emails, truncate, is_palindrome, mask_pii

def run_pytest(test_code: str, extra_args: list[str] | None = None):
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "test_tmp.py"
        p.write_text(textwrap.dedent(test_code))
        cmd = [sys.executable, "-m", "pytest", str(p), "-v", "--tb=short", "--no-header"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout + result.stderr)
        return result.returncode

print("Setup complete. Fixtures at:", FIXTURES)"""),

        md("## Solution 1.1: Test `clean_text`"),
        code("""test_clean_text = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import clean_text

def test_extra_spaces():
    assert clean_text("  hello   world  ") == "hello world"

def test_empty_string():
    assert clean_text("") == ""

def test_tabs_and_newlines():
    assert clean_text("hello\\\\tworld\\\\nfoo") == "hello world foo"

def test_single_word():
    assert clean_text("  word  ") == "word"
'''

rc = run_pytest(test_clean_text)
assert rc == 0, "Tests should pass"
print("Task 1.1 passed!")"""),

        md("## Solution 1.2: Test `tokenize` with `parametrize`"),
        code("""test_tokenize = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")
from sample_module import tokenize

@pytest.mark.parametrize("text, expected", [
    ("hello world", ["hello", "world"]),
    ("", []),
    ("word", ["word"]),
    ("Hello World", ["hello", "world"]),
    ("  extra   spaces  ", ["extra", "spaces"]),
    ("ALL CAPS", ["all", "caps"]),
])
def test_tokenize(text, expected):
    assert tokenize(text) == expected

def test_tokenize_no_lowercase():
    assert tokenize("Hello World", lowercase=False) == ["Hello", "World"]
'''

rc = run_pytest(test_tokenize)
assert rc == 0, "Tests should pass"
print("Task 1.2 passed!")"""),

        md("## Solution 1.3: Test Exceptions"),
        code("""test_exceptions = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")
from sample_module import clean_text, tokenize, truncate

def test_clean_text_type_error():
    with pytest.raises(TypeError):
        clean_text(123)

def test_tokenize_type_error():
    with pytest.raises(TypeError):
        tokenize(None)

def test_truncate_negative_length():
    with pytest.raises(ValueError, match="non-negative"):
        truncate("hello", max_length=-1)
'''

rc = run_pytest(test_exceptions)
assert rc == 0, "Tests should pass"
print("Task 1.3 passed!")"""),

        md("## Solution 1.4: Test `extract_emails` and `mask_pii`"),
        code("""test_pii = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import extract_emails, mask_pii

def test_extract_no_emails():
    assert extract_emails("no email here") == []

def test_extract_one_email():
    assert extract_emails("contact user@example.com please") == ["user@example.com"]

def test_extract_multiple_emails():
    text = "reach a@b.com or c@d.org"
    assert extract_emails(text) == ["a@b.com", "c@d.org"]

def test_mask_pii_email():
    assert mask_pii("contact user@example.com") == "contact [EMAIL]"

def test_mask_pii_phone():
    assert mask_pii("call 123-456-7890") == "call [PHONE]"

def test_mask_pii_both():
    text = "email: a@b.com phone: 123.456.7890"
    result = mask_pii(text)
    assert "[EMAIL]" in result
    assert "[PHONE]" in result
    assert "a@b.com" not in result
'''

rc = run_pytest(test_pii)
assert rc == 0, "Tests should pass"
print("Task 1.4 passed!")"""),

        md("## Solution 1.5: Test `is_palindrome` and `truncate`"),
        code("""test_misc = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import is_palindrome, truncate

def test_palindrome_true():
    assert is_palindrome("racecar") is True

def test_palindrome_with_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama") is True

def test_palindrome_false():
    assert is_palindrome("hello") is False

def test_truncate_short():
    assert truncate("hi", max_length=10) == "hi"

def test_truncate_long():
    assert truncate("hello world", max_length=8) == "hello..."

def test_truncate_exact():
    assert truncate("hello", max_length=5) == "hello"
'''

rc = run_pytest(test_misc)
assert rc == 0, "Tests should pass"
print("Task 1.5 passed!")"""),
    ]
    save(n, "solutions/task_01_pytest_basics_solution.ipynb")


# ─────────────────────────────────────────────
# Solution 02
# ─────────────────────────────────────────────
def solution_02():
    n = nb()
    n.cells = [
        md("""# Solution — Task 02: Fixtures & Parametrize"""),

        md("## Setup"),
        code("""import subprocess, sys, os, textwrap, tempfile, pathlib

FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
sys.path.insert(0, FIXTURES)

def run_pytest(test_code: str, extra_files: dict[str, str] | None = None, extra_args: list[str] | None = None):
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        p = td_path / "test_tmp.py"
        p.write_text(textwrap.dedent(test_code))
        if extra_files:
            for name, content in extra_files.items():
                (td_path / name).write_text(textwrap.dedent(content))
        cmd = [sys.executable, "-m", "pytest", str(td), "-v", "--tb=short", "--no-header"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout + result.stderr)
        return result.returncode

print("Setup complete.")"""),

        md("## Solution 2.1: Fixtures in conftest.py"),
        code("""conftest_code = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")

@pytest.fixture
def sample_texts():
    return [
        "Hello world",
        "  extra   spaces  ",
        "UPPERCASE TEXT",
        "hello\\\\tworld\\\\nfoo",
        "simple",
    ]

@pytest.fixture
def empty_text():
    return ""

@pytest.fixture
def text_with_emails():
    return "Contact alice@example.com and bob@test.org for info."
'''

test_code = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
from sample_module import tokenize, extract_emails, clean_text

def test_tokenize_samples(sample_texts):
    for text in sample_texts:
        tokens = tokenize(text)
        assert isinstance(tokens, list)
        if text.strip():
            assert len(tokens) > 0

def test_clean_empty(empty_text):
    assert clean_text(empty_text) == ""

def test_extract_from_email_text(text_with_emails):
    emails = extract_emails(text_with_emails)
    assert len(emails) >= 2
    assert "alice@example.com" in emails
    assert "bob@test.org" in emails
'''

rc = run_pytest(test_code, extra_files={"conftest.py": conftest_code})
assert rc == 0, "Tests should pass"
print("Task 2.1 passed!")"""),

        md("## Solution 2.2: Parametrize ML Preprocessing"),
        code("""test_preprocess = f'''
import sys, pytest
sys.path.insert(0, "{FIXTURES}")
from sample_ml_pipeline import preprocess_texts

@pytest.mark.parametrize("texts, expected", [
    (["Hello World"], ["hello world"]),
    (["Check https://example.com out"], ["check out"]),
    (["  extra   spaces  "], ["extra spaces"]),
    (["UPPER case Text"], ["upper case text"]),
    (["no change"], ["no change"]),
])
def test_preprocess(texts, expected):
    assert preprocess_texts(texts) == expected

def test_preprocess_multiple():
    texts = ["Hello", "  World  "]
    result = preprocess_texts(texts)
    assert result == ["hello", "world"]
'''

rc = run_pytest(test_preprocess)
assert rc == 0, "Tests should pass"
print("Task 2.2 passed!")"""),

        md("## Solution 2.3: `tmp_path` for File-Based Tests"),
        code("""test_file = f'''
import sys
sys.path.insert(0, "{FIXTURES}")
import pandas as pd
from sample_module import clean_text

def test_csv_processing(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("text" + chr(10) + "  hello   world  " + chr(10) + "  foo   bar  " + chr(10))
    df = pd.read_csv(csv_path)
    df["cleaned"] = df["text"].apply(clean_text)
    assert df["cleaned"].tolist() == ["hello world", "foo bar"]

def test_json_roundtrip(tmp_path):
    import json
    data = {{"texts": ["hello", "world"], "labels": [1, 0]}}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(data))
    loaded = json.loads(json_path.read_text())
    assert loaded == data
'''

rc = run_pytest(test_file)
assert rc == 0, "Tests should pass"
print("Task 2.3 passed!")"""),

        md("## Solution 2.4: Fixture with Yield"),
        code("""test_yield = f'''
import sys, pytest, numpy as np
sys.path.insert(0, "{FIXTURES}")
from sample_ml_pipeline import create_pipeline, train_pipeline, predict, SAMPLE_TEXTS, SAMPLE_LABELS

cleanup_log = []

@pytest.fixture
def trained_pipe():
    np.random.seed(42)
    pipe = create_pipeline(max_features=100)
    train_pipeline(pipe, SAMPLE_TEXTS, SAMPLE_LABELS)
    yield pipe
    cleanup_log.append("pipeline cleaned up")

def test_predict(trained_pipe):
    preds = predict(trained_pipe, ["great product"])
    assert len(preds) == 1

def test_cleanup_happened():
    assert "pipeline cleaned up" in cleanup_log
'''

rc = run_pytest(test_yield)
assert rc == 0, "Tests should pass"
print("Task 2.4 passed!")"""),
    ]
    save(n, "solutions/task_02_fixtures_parametrize_solution.ipynb")


# ─────────────────────────────────────────────
# Solution 03
# ─────────────────────────────────────────────
def solution_03():
    n = nb()
    n.cells = [
        md("""# Solution — Task 03: Testing ML API"""),

        md("## Setup"),
        code("""import sys, os
from unittest.mock import MagicMock

FIXTURES = os.path.abspath(os.path.join("..", "fixtures", "input"))
if not os.path.exists(FIXTURES):
    FIXTURES = os.path.abspath(os.path.join("fixtures", "input"))
sys.path.insert(0, FIXTURES)

from fastapi.testclient import TestClient
from sample_fastapi_app import app, ml_models

print("Setup complete.")"""),

        md("## Solution 3.1: Test Health Endpoint"),
        code("""with TestClient(app) as client:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    print(f"Health response: {data}")

# TEST — Do not modify
with TestClient(app) as client:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True
print("Task 3.1 passed!")"""),

        md("## Solution 3.2: Test Predict Endpoint"),
        code("""with TestClient(app) as client:
    # Test positive text
    resp = client.post("/predict", json={"text": "This is amazing and great"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("positive", "negative", "neutral")
    assert 0 <= data["score"] <= 1
    assert data["text"] == "This is amazing and great"
    print(f"Prediction: {data}")

    # Test negative text
    resp = client.post("/predict", json={"text": "Terrible and awful"})
    assert resp.status_code == 200
    data = resp.json()
    print(f"Negative prediction: {data}")

# TEST — Do not modify
with TestClient(app) as client:
    resp = client.post("/predict", json={"text": "This is amazing and great"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "score" in data
    assert "text" in data
    assert 0 <= data["score"] <= 1
    assert data["text"] == "This is amazing and great"
print("Task 3.2 passed!")"""),

        md("## Solution 3.3: Test Validation Errors"),
        code("""with TestClient(app) as client:
    # Empty text
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422
    print(f"Empty text: {resp.status_code}")

    # Missing field
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
    print(f"Missing field: {resp.status_code}")

    # Too long
    resp = client.post("/predict", json={"text": "x" * 5001})
    assert resp.status_code == 422
    print(f"Too long: {resp.status_code}")

# TEST — Do not modify
with TestClient(app) as client:
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 5001}).status_code == 422
print("Task 3.3 passed!")"""),

        md("## Solution 3.4: Test Batch Predict"),
        code("""with TestClient(app) as client:
    texts = ["good product", "terrible service", "it was okay"]
    resp = client.post("/predict/batch", json={"texts": texts})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 3
    for pred in preds:
        assert "label" in pred
        assert "score" in pred
        assert "text" in pred
    print(f"Batch predictions: {[p['label'] for p in preds]}")

# TEST — Do not modify
with TestClient(app) as client:
    texts = ["good product", "terrible service", "it was okay"]
    resp = client.post("/predict/batch", json={"texts": texts})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert "label" in p and "score" in p and "text" in p
print("Task 3.4 passed!")"""),

        md("## Solution 3.5: Mock the ML Model"),
        code("""with TestClient(app) as client:
    mock_model = MagicMock()
    mock_model.predict.return_value = ("positive", 0.99)
    ml_models["sentiment"] = mock_model

    resp = client.post("/predict", json={"text": "any text here"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "positive"
    assert resp.json()["score"] == 0.99
    mock_model.predict.assert_called_once_with("any text here")
    print(f"Mock prediction: {resp.json()}")

# TEST — Do not modify
with TestClient(app) as client:
    mock_model = MagicMock()
    mock_model.predict.return_value = ("positive", 0.99)
    ml_models["sentiment"] = mock_model

    resp = client.post("/predict", json={"text": "any text here"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "positive"
    assert resp.json()["score"] == 0.99
    mock_model.predict.assert_called_once_with("any text here")
print("Task 3.5 passed!")"""),

        md("## Solution 3.6: Test Model Not Loaded"),
        code("""with TestClient(app) as client:
    ml_models.clear()
    resp = client.post("/predict", json={"text": "hello"})
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()
    print(f"503 response: {resp.json()}")

# TEST — Do not modify
with TestClient(app) as client:
    ml_models.clear()
    resp = client.post("/predict", json={"text": "hello"})
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()
print("Task 3.6 passed!")"""),
    ]
    save(n, "solutions/task_03_testing_ml_api_solution.ipynb")


# ─────────────────────────────────────────────
# Generate all
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating notebooks...")
    learning_01()
    learning_02()
    learning_03()
    learning_04()
    task_01()
    task_02()
    task_03()
    solution_01()
    solution_02()
    solution_03()
    print("\nDone! All notebooks generated.")
