# Module 10 Fixtures

## Running FastAPI in Jupyter Notebooks

FastAPI servers in notebooks can be tricky. Here are two approaches:

### Approach 1: Run Server in Terminal (Recommended)

1. Save your FastAPI app to a Python file:
```python
# app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello"}
```

2. Run in terminal:
```bash
uvicorn app:app --reload --port 8000
```

3. Test from notebook:
```python
import httpx

response = httpx.get("http://localhost:8000/")
print(response.json())
```

### Approach 2: Run in Notebook (For Learning)

Use threading to run server in background:

```python
from fastapi import FastAPI
import uvicorn
from threading import Thread
import time

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello"}

# Run server in background thread
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

thread = Thread(target=run_server, daemon=True)
thread.start()
time.sleep(2)  # Wait for server to start

print("Server running at http://127.0.0.1:8001")
print("Docs at http://127.0.0.1:8001/docs")
```

**Note:** Daemon threads will stop when notebook kernel stops.

### Testing Endpoints

Use `httpx` for making requests:

```python
import httpx

# Synchronous requests
response = httpx.get("http://localhost:8001/")
print(response.status_code)
print(response.json())

# POST request
response = httpx.post(
    "http://localhost:8001/predict",
    json={"text": "test input"}
)
print(response.json())
```

## Fixtures

### sample_texts.json
Sample texts for sentiment classification training/testing:
- 5 texts with sentiment labels (positive/negative/neutral)
- Used in text classification tasks

### test_file.txt
Sample text file for file upload testing:
- Simple text content
- Used to test file upload endpoints

## Common Issues

### Port Already in Use
```bash
# Find process using port 8000
lsof -ti:8000

# Kill process
kill -9 $(lsof -ti:8000)
```

### Server Not Starting
- Check if port is available
- Verify FastAPI and uvicorn are installed
- Check for syntax errors in app definition

### 422 Validation Errors
- Verify request body matches Pydantic model
- Check required fields are provided
- Validate data types match model definition
