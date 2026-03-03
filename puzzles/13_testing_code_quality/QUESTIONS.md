# Module 11: Testing & Code Quality — Deep Questions

## Architecture & Design (Q1–Q7)

### Q1: pytest fixtures vs. `setUp`/`tearDown` — why did pytest win?

**`setUp`/`tearDown` (unittest style):**
```python
class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.db = connect_db()
        self.model = load_model()       # what if I only need the model?
    def tearDown(self):
        self.db.close()
```

Problems: every test pays the cost of *all* setup, fixtures are implicit (what does `self.db` contain?), and sharing between files requires inheritance.

**pytest fixtures:**
```python
@pytest.fixture(scope="session")
def model():
    return load_model()

@pytest.fixture
def db():
    conn = connect_db()
    yield conn
    conn.close()

def test_predict(model):           # only requests what it needs
    assert model.predict(["hi"])

def test_store(db, model):         # composes freely
    db.store(model.predict(["hi"]))
```

Key advantages:
- **Explicit dependency injection** — each test declares exactly what it needs
- **Scoping** — `session`-scoped model loads once across all tests; `function`-scoped db resets per test
- **`yield` for teardown** — setup and teardown live together, impossible to forget cleanup
- **conftest.py sharing** — no inheritance, fixtures available to all tests in the directory tree
- **Composability** — fixtures can depend on other fixtures, forming a DAG

The real insight: pytest fixtures are essentially a lightweight DI container scoped to your test suite.

---

### Q2: How should you structure tests for a project with both an ML pipeline and an API?

```
tests/
├── conftest.py              # shared fixtures (sample data, trained model stub)
├── unit/
│   ├── test_preprocessing.py    # pure functions, fast
│   ├── test_model.py            # model logic with mocked weights
│   └── test_schemas.py          # Pydantic model validation
├── integration/
│   ├── test_api.py              # TestClient + real app
│   └── test_pipeline.py         # end-to-end pipeline (train → predict)
└── e2e/
    └── test_api_live.py         # against running server (CI only)
```

**Key decisions:**
- **Unit tests** mock everything external, run in <1s total
- **Integration tests** use real components but local resources (TestClient, SQLite)
- **E2E tests** hit real services, marked `@pytest.mark.e2e`, run only in CI
- `conftest.py` at root provides lightweight fixtures; expensive fixtures go in `integration/conftest.py` with `session` scope

Run fast by default: `pytest tests/unit`. Full suite in CI: `pytest`.

---

### Q3: What's the right fixture scope for an ML model in tests?

**`function` scope** — loads model per test. Safe but brutally slow if model loading takes seconds.

**`session` scope** — loads once, shared across all tests. Fast, but:
- Tests must NOT mutate the model (no `model.fit()` in tests)
- If one test corrupts state, all subsequent tests fail (cascading failures)

**Recommended pattern:**
```python
@pytest.fixture(scope="session")
def base_model():
    """Expensive: loads once."""
    return load_model("weights.bin")

@pytest.fixture
def model(base_model):
    """Cheap: returns a copy per test if model is mutable."""
    import copy
    return copy.deepcopy(base_model)
```

For sklearn pipelines, `deepcopy` is cheap (~ms). For large neural nets, you often can't deepcopy — in that case, use `session` scope and write tests that only call `.predict()` (read-only).

---

### Q4: How do you test code that calls an LLM API (OpenAI, Anthropic)?

Never call real LLM APIs in tests — they're slow, expensive, non-deterministic, and rate-limited.

**Strategy 1: Mock at the HTTP level**
```python
@responses.activate
def test_llm_call():
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "Paris"}}]},
    )
    assert ask_llm("Capital of France?") == "Paris"
```

**Strategy 2: Inject the client**
```python
def classify(text: str, client=None):
    client = client or openai.OpenAI()
    ...

def test_classify():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = ...
    assert classify("hello", client=mock_client) == "greeting"
```

**Strategy 3: Golden file tests** — record real API responses once, replay in tests. Libraries like `vcrpy` or `responses` can record/replay HTTP interactions.

For structured output (JSON mode), test your Pydantic parsing separately from the API call.

---

### Q5: `monkeypatch` vs `unittest.mock.patch` — when to use which?

| Feature | `monkeypatch` | `mock.patch` |
|---------|--------------|--------------|
| Scope | pytest-native, auto-cleanup | needs decorator/context manager |
| Env vars | `monkeypatch.setenv("KEY", "val")` | `patch.dict(os.environ, {...})` |
| Attributes | `monkeypatch.setattr(obj, "attr", val)` | `patch.object(obj, "attr", val)` |
| Return values | Manual: set to a lambda/value | `MagicMock` with `.return_value` |
| Call tracking | No built-in | `.assert_called_once_with()` etc. |

**Use `monkeypatch`** when you just need to swap a value (env var, config, simple attribute).

**Use `mock.patch`** when you need to verify interactions (was it called? how many times? with what args?).

```python
# monkeypatch: simple value swap
def test_config(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/tmp/test_model")
    assert load_config().model_path == "/tmp/test_model"

# mock.patch: verify behavior
@patch("myapp.send_notification")
def test_on_error(mock_send):
    process_with_error()
    mock_send.assert_called_once_with("Error occurred", severity="high")
```

---

### Q6: How do you design tests for a FastAPI app that uses lifespan events?

Since Starlette 0.50+, `TestClient(app)` without context manager does **NOT** trigger lifespan events. This is a common trap.

```python
# WRONG — lifespan not triggered, ml_models is empty
client = TestClient(app)
response = client.post("/predict", json={"text": "hello"})
# 503: Model not loaded

# CORRECT — lifespan runs, model loads
with TestClient(app) as client:
    response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 200
```

**Testing with a mock model:**
```python
@pytest.fixture
def client():
    # Override lifespan to inject mock model
    from myapp import app, ml_models
    with TestClient(app) as c:
        ml_models["sentiment"] = MockSentimentModel()
        yield c
```

Or use FastAPI's dependency override system if the model is provided via `Depends()`.

---

### Q7: What's the role of `hypothesis` (property-based testing) in ML testing?

Instead of writing individual test cases, you describe *properties* and let hypothesis generate inputs:

```python
from hypothesis import given, strategies as st

@given(st.text())
def test_clean_text_idempotent(s):
    """Cleaning twice gives the same result as cleaning once."""
    result = clean_text(s)
    assert clean_text(result) == result

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_normalize_range(values):
    """Normalized values should be in [0, 1]."""
    normed = min_max_normalize(values)
    assert all(0 <= v <= 1 for v in normed)
```

Hypothesis is excellent for finding edge cases you wouldn't think of: empty strings, Unicode, extreme floats, huge lists. It's complementary to example-based tests, not a replacement.

---

## Implementation & Coding (Q8–Q14)

### Q8: Write a `conftest.py` that provides fixtures for testing an ML API.

```python
# tests/conftest.py
import pytest
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def sample_texts():
    return [
        "This product is great!",
        "Terrible experience.",
        "It was okay, nothing special.",
    ]


@pytest.fixture(scope="session")
def sample_labels():
    return [1, 0, 1]


@pytest.fixture(scope="session")
def trained_pipeline(sample_texts, sample_labels):
    """Train once, reuse across all tests."""
    from sample_ml_pipeline import create_pipeline, train_pipeline
    np.random.seed(42)
    pipe = create_pipeline(max_features=100)
    train_pipeline(pipe, sample_texts * 3, sample_labels * 3)  # need enough data
    return pipe


@pytest.fixture
def client():
    from sample_fastapi_app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_model():
    """Lightweight model substitute."""
    from unittest.mock import MagicMock
    model = MagicMock()
    model.predict.return_value = ("positive", 0.95)
    return model
```

Note `session` scope for the trained pipeline (expensive), `function` scope for the client (cheap, fresh per test).

---

### Q9: How do you use `parametrize` to test edge cases systematically?

```python
@pytest.mark.parametrize("text, expected", [
    # Normal cases
    ("hello world", ["hello", "world"]),
    ("Hello World", ["hello", "world"]),

    # Edge cases
    ("", []),                           # empty string
    ("   ", []),                         # whitespace only
    ("word", ["word"]),                  # single word
    ("  leading trailing  ", ["leading", "trailing"]),

    # Unicode
    ("café résumé", ["café", "résumé"]),

    # Special characters
    ("hello\nworld\ttab", ["hello", "world", "tab"]),
])
def test_tokenize(text, expected):
    assert tokenize(text) == expected


# Use IDs for readability in test output
@pytest.mark.parametrize("text, expected", [
    pytest.param("", [], id="empty"),
    pytest.param("a b", ["a", "b"], id="simple"),
    pytest.param("  x  ", ["x"], id="whitespace"),
])
def test_tokenize_with_ids(text, expected):
    assert tokenize(text) == expected
```

The `id` parameter makes test output readable: `test_tokenize_with_ids[empty] PASSED` instead of `test_tokenize_with_ids[-[]]`.

---

### Q10: How do you test that a function raises the right exception with the right message?

```python
def test_train_empty_data():
    pipe = create_pipeline()
    with pytest.raises(ValueError, match="Cannot train on empty data"):
        train_pipeline(pipe, [], [])

def test_train_mismatched_lengths():
    pipe = create_pipeline()
    with pytest.raises(ValueError, match=r"\d+ != \d+"):
        train_pipeline(pipe, ["a", "b"], [1])

# Capture and inspect the exception object
def test_exception_details():
    with pytest.raises(ValueError) as exc_info:
        train_pipeline(create_pipeline(), [], [])
    assert "empty" in str(exc_info.value)
    assert exc_info.type is ValueError
```

The `match` parameter takes a **regex**, so you can use patterns like `r"\d+"` for dynamic parts of error messages.

---

### Q11: How do you mock `datetime.now()` or `time.time()` in tests?

```python
from unittest.mock import patch
from datetime import datetime

# If your code does: from datetime import datetime; datetime.now()
@patch("mymodule.datetime")
def test_time_dependent(mock_dt):
    mock_dt.now.return_value = datetime(2024, 1, 15, 10, 30)
    assert get_greeting() == "Good morning"

# monkeypatch approach (often cleaner)
def test_cache_expiry(monkeypatch):
    current_time = [1000.0]
    monkeypatch.setattr("time.time", lambda: current_time[0])

    cache.set("key", "value", ttl=60)
    assert cache.get("key") == "value"

    current_time[0] = 1061.0  # advance past TTL
    assert cache.get("key") is None
```

Important: you must patch `datetime` **where it's imported**, not where it's defined. If `mymodule.py` has `from datetime import datetime`, patch `mymodule.datetime`, not `datetime.datetime`.

---

### Q12: How do you write tests that create temporary files?

```python
def test_save_and_load_model(tmp_path):
    """tmp_path is a pytest built-in fixture — unique per test, auto-cleaned."""
    model_path = tmp_path / "model.pkl"

    # Save
    pipe = create_pipeline()
    train_pipeline(pipe, TEXTS, LABELS)
    import joblib
    joblib.dump(pipe, model_path)

    # Load and verify
    loaded = joblib.load(model_path)
    preds = loaded.predict(["test"])
    assert len(preds) == 1

def test_csv_processing(tmp_path):
    # Create test CSV
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("text,label\nhello,1\nbye,0\n")

    df = process_csv(csv_path)
    assert len(df) == 2
```

`tmp_path` gives you a `pathlib.Path` to a unique temp directory. `tmp_path_factory` (session-scoped) lets you share temp dirs across tests.

---

### Q13: How do you test async FastAPI endpoints?

```python
# Option 1: TestClient (synchronous wrapper) — simplest
def test_predict_sync():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "great product"})
        assert resp.status_code == 200
        assert resp.json()["label"] == "positive"

# Option 2: httpx.AsyncClient — truly async
import pytest
import httpx

@pytest.mark.asyncio
async def test_predict_async():
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/predict", json={"text": "great product"})
        assert resp.status_code == 200
```

For most cases, `TestClient` (sync) is sufficient. Use `httpx.AsyncClient` when testing WebSocket-like behavior or when you need concurrent test requests.

---

### Q14: How do you use `responses` to mock HTTP calls in tests?

```python
import responses

@responses.activate
def test_fetch_embeddings():
    # Register mock response
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/embeddings",
        json={
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"total_tokens": 5},
        },
        status=200,
    )

    result = get_embedding("hello world")
    assert len(result) == 3
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-key"


@responses.activate
def test_api_retry_on_500():
    """Test retry logic by returning 500 first, then 200."""
    responses.add(responses.GET, "https://api.example.com/data", status=500)
    responses.add(responses.GET, "https://api.example.com/data", json={"ok": True}, status=200)

    result = fetch_with_retry("https://api.example.com/data")
    assert result == {"ok": True}
    assert len(responses.calls) == 2  # retried once
```

---

## Debugging & Troubleshooting (Q15–Q17)

### Q15: A test passes when run alone but fails when run with other tests. What's happening?

**Shared mutable state** — the classic flaky test cause.

Common culprits:
1. **Module-level mutable objects** — a dict or list modified by one test, not reset
2. **Session-scoped fixtures mutated** — test A modifies the fixture, test B sees the mutation
3. **Import-time side effects** — importing a module sets global state
4. **Database state** — test A inserts rows, test B assumes empty table

**Debugging:**
```bash
# Run the failing test in isolation — if it passes, it's a state leak
pytest tests/test_failing.py::test_specific -v

# Run with random order to catch order-dependent tests
pip install pytest-randomly
pytest --randomly-seed=12345
```

**Fix:** use `function`-scoped fixtures, ensure teardown/cleanup, avoid global mutable state. If you *must* share state, use `deepcopy` in fixtures.

---

### Q16: Tests are slow. How do you profile and speed them up?

```bash
# Find the slowest tests
pytest --durations=10

# Typical output:
# 5.2s  test_integration.py::test_full_pipeline
# 3.1s  test_model.py::test_train_and_evaluate
# 0.01s test_utils.py::test_clean_text
```

**Common speed fixes:**
1. **Scope expensive fixtures to `session`** — model loading, DB setup
2. **Use lighter models for tests** — `max_features=100` instead of 10000
3. **Mock I/O and network** — don't hit real APIs
4. **Mark slow tests** and skip by default: `pytest -m "not slow"`
5. **Parallel execution** with `pytest-xdist`: `pytest -n auto`

**Rule of thumb:** unit tests should take <1s total. If a single test takes >1s, it's probably an integration test — mark it accordingly.

---

### Q17: `mock.patch` target is wrong — the test passes but doesn't actually mock anything. How to debug?

This is the #1 mocking mistake. You must patch where the name is **looked up**, not where it's **defined**.

```python
# myapp.py
from datetime import datetime    # <-- imports datetime into myapp namespace

def get_hour():
    return datetime.now().hour

# WRONG: patches the original, but myapp already imported its own reference
@patch("datetime.datetime")
def test_wrong(mock_dt): ...

# CORRECT: patches the reference in myapp
@patch("myapp.datetime")
def test_correct(mock_dt):
    mock_dt.now.return_value = datetime(2024, 1, 1, 14, 0)
    assert get_hour() == 14
```

**Debugging tips:**
- Add `print(type(the_thing))` inside the function to check if it's a `MagicMock`
- If it prints the real type, your patch target is wrong
- Use `@patch.object(module, "name")` for clarity

---

## Trade-offs & Decisions (Q18–Q20)

### Q18: What coverage percentage should you target? Is 100% worth it?

**80-90% is the sweet spot for most projects.**

**Why not 100%:**
- Last 10% often covers defensive code that's hard to trigger (except clauses for rare errors)
- Chasing 100% leads to brittle tests that test implementation details
- Time spent on 95→100% is better spent on property-based tests or integration tests

**What matters more than the number:**
- **Branch coverage** (`--cov-branch`) catches untested else-paths
- **Critical path coverage** — auth, payment, data validation MUST be 100%
- **Mutation testing** (with `mutmut`) — verifies tests actually catch bugs, not just run lines

```bash
# Good: check what you're missing
pytest --cov=src --cov-report=term-missing --cov-branch

# The "missing" column shows exactly which lines/branches need tests
```

---

### Q19: `mypy --strict` vs. gradual typing — what's the right level?

| Approach | Effort | Benefit | Best for |
|----------|--------|---------|----------|
| No typing | None | None | Scripts, throwaway code |
| Gradual (`mypy src/`) | Low | Catches obvious bugs | Most projects |
| `--strict` | High | Catches subtle bugs | Libraries, critical systems |

**Practical recommendation:** start gradual, add `--strict` to new modules.

```toml
# pyproject.toml — strict for new code, lenient for legacy
[tool.mypy]
python_version = "3.12"

[[tool.mypy.overrides]]
module = "legacy_module.*"
ignore_errors = true

[[tool.mypy.overrides]]
module = "new_module.*"
disallow_untyped_defs = true
warn_return_any = true
```

ML-specific pain points: numpy/pandas typing is verbose and sometimes wrong. `pandas-stubs` helps but isn't perfect. Don't fight the type checker on DataFrame operations — use `# type: ignore` pragmatically.

---

### Q20: When should you use `hypothesis` vs. hand-written parametrize tests?

| | `parametrize` | `hypothesis` |
|---|---|---|
| **Control** | You pick exact inputs | Framework generates inputs |
| **Edge cases** | Only what you think of | Finds things you'd never think of |
| **Reproducibility** | Deterministic | Uses seed, shrinks to minimal example |
| **Speed** | Fast (fixed N tests) | Slower (default 100 examples) |
| **Best for** | Known edge cases, regression tests | Property validation, input fuzzing |

**Use `parametrize` when:** you know the exact inputs and expected outputs.

**Use `hypothesis` when:** you can state a property ("output is always positive", "function is idempotent", "roundtrip encode/decode returns original").

**Combine them:**
```python
# parametrize for known regressions
@pytest.mark.parametrize("input,expected", [
    ("bug-123-input", "bug-123-output"),
])
def test_known_cases(input, expected): ...

# hypothesis for property-based exploration
@given(st.text())
def test_roundtrip(s):
    assert decode(encode(s)) == s
```

The best test suites use both: parametrize for documentation and regression, hypothesis for exploration.
