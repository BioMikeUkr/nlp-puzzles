# Module 11: Testing & Code Quality

## Overview

Testing is what separates prototypes from production ML systems. This module covers **pytest** (the de facto Python testing framework), **mocking** strategies for ML components, **FastAPI testing**, and code quality tools like **ruff** and **mypy**.

### Learning Objectives
- Write and organize tests with pytest (fixtures, parametrize, markers)
- Mock external dependencies: APIs, ML models, databases
- Test FastAPI applications with `TestClient`
- Use coverage reports to find untested paths
- Integrate ruff + mypy into your workflow

---

## 1. pytest Basics

### Test Discovery
pytest auto-discovers tests by convention:
- Files: `test_*.py` or `*_test.py`
- Functions: `test_*`
- Classes: `Test*` (no `__init__`)

```python
# test_example.py
def test_addition():
    assert 1 + 1 == 2

class TestMath:
    def test_multiply(self):
        assert 2 * 3 == 6
```

### Rich Assertions
No need for `assertEqual` — plain `assert` gives detailed diffs:

```python
def test_list():
    result = [1, 2, 3]
    assert result == [1, 2, 4]
    # pytest shows:  E  At index 2 diff: 3 != 4
```

### Expected Exceptions
```python
import pytest

def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0

def test_error_message():
    with pytest.raises(ValueError, match="must be positive"):
        validate(-1)
```

### Markers
```python
@pytest.mark.skip(reason="not implemented yet")
def test_future_feature(): ...

@pytest.mark.xfail(reason="known bug #123")
def test_known_bug(): ...

@pytest.mark.slow
def test_train_large_model(): ...
```

Run subsets: `pytest -m "not slow"`

---

## 2. Fixtures

Fixtures replace setup/teardown with dependency injection:

```python
@pytest.fixture
def sample_df():
    return pd.DataFrame({"text": ["hello", "world"], "label": [1, 0]})

def test_shape(sample_df):
    assert sample_df.shape == (2, 2)
```

### Scopes
| Scope | Created | Destroyed |
|-------|---------|-----------|
| `function` (default) | Per test | After test |
| `class` | Per class | After class |
| `module` | Per file | After file |
| `session` | Once | At end |

Use `session` scope for expensive resources (model loading, DB connections).

### Built-in Fixtures
- **`tmp_path`** — temporary directory (per-test)
- **`capsys`** — capture stdout/stderr
- **`monkeypatch`** — modify objects/env vars temporarily

### conftest.py
Shared fixtures go in `conftest.py` — pytest auto-discovers them. No imports needed.

```
tests/
├── conftest.py          # shared fixtures
├── test_api.py
└── test_pipeline.py
```

---

## 3. Parametrize

Test multiple inputs without copy-pasting:

```python
@pytest.mark.parametrize("input_text, expected", [
    ("hello world", ["hello", "world"]),
    ("", []),
    ("  spaces  ", ["spaces"]),
])
def test_tokenize(input_text, expected):
    assert tokenize(input_text) == expected
```

Stack decorators for combinatorial testing:

```python
@pytest.mark.parametrize("model", ["lr", "svm"])
@pytest.mark.parametrize("max_features", [100, 1000])
def test_pipeline(model, max_features):  # runs 4 combinations
    ...
```

---

## 4. Mocking & Patching

### When to Mock
- External APIs (HTTP calls, LLM APIs)
- Heavy ML models (use lightweight stubs)
- Database connections
- Time, randomness, filesystem (when needed)

### When NOT to Mock
- Pure functions (just test them directly)
- Your own simple classes (test the real thing)
- Things that are fast and deterministic

### unittest.mock
```python
from unittest.mock import patch, MagicMock

@patch("mymodule.requests.get")
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {"result": "ok"}
    result = my_function()
    assert result == "ok"
    mock_get.assert_called_once()
```

### monkeypatch (pytest-native)
```python
def test_with_env_var(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key-123")
    assert get_api_key() == "test-key-123"
```

### responses library (HTTP mocking)
```python
import responses

@responses.activate
def test_external_api():
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"items": [1, 2, 3]},
        status=200,
    )
    result = fetch_data()
    assert result == [1, 2, 3]
```

---

## 5. Testing FastAPI

```python
from fastapi.testclient import TestClient
from myapp import app

# IMPORTANT: use context manager for lifespan events (Starlette 0.50+)
with TestClient(app) as client:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

### Testing Validation
```python
with TestClient(app) as client:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error
```

---

## 6. Testing ML Pipelines

### Reproducibility
Always set seeds:
```python
@pytest.fixture
def trained_pipeline():
    np.random.seed(42)
    pipe = create_pipeline()
    pipe.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return pipe
```

### Smoke Tests vs. Performance Tests
```python
# Smoke test — does it run without crashing?
def test_pipeline_runs(trained_pipeline):
    preds = trained_pipeline.predict(["test input"])
    assert len(preds) == 1

# Performance test — mark as slow, run separately
@pytest.mark.slow
def test_pipeline_accuracy(trained_pipeline):
    preds = trained_pipeline.predict(TEST_TEXTS)
    assert accuracy_score(TEST_LABELS, preds) > 0.7
```

### Snapshot/Golden File Testing
Save expected outputs, compare on each run:
```python
def test_predictions_snapshot(trained_pipeline, tmp_path):
    preds = trained_pipeline.predict(TEXTS)
    result_path = tmp_path / "preds.json"
    # Compare with stored golden file
    assert preds.tolist() == EXPECTED_PREDS
```

---

## 7. Code Quality Tools

### ruff (linter + formatter)
```bash
ruff check .           # lint
ruff check --fix .     # auto-fix
ruff format .          # format (replaces black)
```

### mypy (type checker)
```bash
mypy src/ --strict
```

Key settings in `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
disallow_untyped_defs = true
```

### pre-commit
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
      - id: ruff-format
```

---

## 8. Coverage

```bash
pytest --cov=src --cov-report=term-missing
pytest --cov=src --cov-report=html    # open htmlcov/index.html
```

### Coverage Traps
- **100% coverage ≠ bug-free** — coverage only tells you which lines ran, not that they're correct
- **Branch coverage** (`--cov-branch`) catches untested `if/else` paths
- Don't chase 100% — focus on critical paths

---

## 9. Best Practices

1. **Test behavior, not implementation** — test what a function does, not how
2. **One assert per concept** — each test should verify one thing
3. **Fast tests by default** — mark slow tests with `@pytest.mark.slow`
4. **Fixtures > setup/teardown** — more composable, more explicit
5. **Don't mock what you don't own** — wrap third-party code, mock the wrapper
6. **Use parametrize for edge cases** — cleaner than copy-pasting tests
7. **Name tests descriptively** — `test_tokenize_handles_empty_string` > `test_tokenize_1`
