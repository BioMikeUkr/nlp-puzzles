# Module 10: FastAPI for Machine Learning

> Building production-ready ML APIs with FastAPI

## Overview

FastAPI is the modern Python framework for building APIs, particularly well-suited for ML applications. This module covers building production-ready ML APIs with request validation, async processing, error handling, and best practices.

## Learning Objectives

By the end of this module, you will be able to:
- Build RESTful APIs with FastAPI
- Implement request/response validation with Pydantic
- Handle async operations for concurrent requests
- Deploy ML models as API endpoints
- Implement authentication and authorization
- Handle file uploads for ML predictions
- Stream responses for LLM applications
- Write API tests with pytest
- Deploy FastAPI applications

## Key Concepts

### 1. Why FastAPI for ML?

**Advantages:**
- ✅ **Fast**: One of the fastest Python frameworks (based on Starlette/Pydantic)
- ✅ **Type hints**: Built-in validation with Python type hints
- ✅ **Async support**: Handle concurrent requests efficiently
- ✅ **Auto documentation**: Swagger UI and ReDoc generated automatically
- ✅ **Production ready**: Used by companies like Microsoft, Uber, Netflix

**FastAPI vs Flask:**

| Feature | FastAPI | Flask |
|---------|---------|-------|
| Performance | Very Fast | Moderate |
| Async support | Built-in | Via extensions |
| Validation | Automatic (Pydantic) | Manual |
| Documentation | Auto-generated | Manual |
| Type hints | Required | Optional |
| Learning curve | Moderate | Easy |

### 2. FastAPI Basics

**Minimal API:**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

**Run server:**
```bash
uvicorn main:app --reload
```

**Automatic documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Request/Response Models with Pydantic

**Define data models:**

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    model_name: Optional[str] = "default"

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    processing_time_ms: float
```

**Use in endpoint:**

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # FastAPI automatically validates request
    # and serializes response
    result = model.predict(request.text)
    return PredictionResponse(
        prediction=result['label'],
        confidence=result['confidence'],
        model_used=request.model_name,
        processing_time_ms=result['time']
    )
```

### 4. ML Model Integration

**Load model at startup:**

```python
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

# Global model variable
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    ml_models["embedder"] = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded!")
    yield
    # Cleanup on shutdown
    ml_models.clear()
    print("Model unloaded!")

app = FastAPI(lifespan=lifespan)

@app.post("/embed")
def create_embedding(text: str):
    embedder = ml_models["embedder"]
    embedding = embedder.encode(text)
    return {"embedding": embedding.tolist()}
```

### 5. Async Operations

**Why async for ML APIs:**
- Handle multiple requests concurrently
- I/O-bound operations (database, file reads) don't block
- Better resource utilization

**Async endpoint:**

```python
import asyncio
from typing import List

@app.post("/predict-batch")
async def predict_batch(texts: List[str]):
    # Process multiple texts concurrently
    tasks = [process_text_async(text) for text in texts]
    results = await asyncio.gather(*tasks)
    return {"predictions": results}

async def process_text_async(text: str):
    # Simulate async processing
    await asyncio.sleep(0.1)  # I/O operation

    # CPU-bound work still blocks, but I/O doesn't
    embedding = ml_models["embedder"].encode(text)
    return {"text": text, "embedding": embedding.tolist()}
```

**When to use async:**
- ✅ Multiple I/O operations (database, external APIs)
- ✅ File operations with aiofiles
- ✅ Batch processing with concurrent requests
- ❌ Pure CPU-bound operations (use background tasks instead)

### 6. Background Tasks

For long-running operations, use background tasks:

```python
from fastapi import BackgroundTasks

def process_large_file(file_path: str):
    # Long-running task
    time.sleep(10)
    print(f"Processed {file_path}")

@app.post("/process-file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile
):
    file_path = f"/tmp/{file.filename}"

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process in background
    background_tasks.add_task(process_large_file, file_path)

    return {"message": "File uploaded, processing in background"}
```

### 7. File Uploads

**Handle file uploads:**

```python
from fastapi import File, UploadFile
import aiofiles

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")

    # Validate file size
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(400, "File too large")

    # Save file asynchronously
    async with aiofiles.open(f"uploads/{file.filename}", 'wb') as f:
        await f.write(contents)

    # Process image
    prediction = predict_image(contents)

    return {"filename": file.filename, "prediction": prediction}
```

### 8. Error Handling

**Custom exception handlers:**

```python
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

class ModelNotLoadedError(Exception):
    pass

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"error": "Model not loaded", "detail": str(exc)}
    )

@app.get("/predict/{text}")
def predict(text: str):
    if "embedder" not in ml_models:
        raise ModelNotLoadedError("Embedding model not initialized")

    try:
        result = ml_models["embedder"].encode(text)
        return {"embedding": result.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
```

### 9. Authentication

**API Key authentication:**

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@app.post("/predict", dependencies=[Security(verify_api_key)])
def predict_authenticated(request: PredictionRequest):
    # Only accessible with valid API key
    return perform_prediction(request)
```

**JWT tokens:**

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends
from fastapi.security import HTTPBearer

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(403, "Invalid token")

@app.post("/login")
def login(username: str, password: str):
    # Verify credentials (simplified)
    if username == "admin" and password == "password":
        token = create_access_token({"sub": username})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(401, "Invalid credentials")

@app.get("/protected")
def protected_route(payload: dict = Depends(verify_token)):
    return {"message": f"Hello {payload['sub']}"}
```

### 10. Streaming Responses

**For LLM applications:**

```python
from fastapi.responses import StreamingResponse
import asyncio

async def generate_text_stream(prompt: str):
    # Simulate LLM streaming
    words = ["This", "is", "a", "streaming", "response"]
    for word in words:
        await asyncio.sleep(0.5)
        yield f"data: {word}\n\n"

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    return StreamingResponse(
        generate_text_stream(prompt),
        media_type="text/event-stream"
    )
```

### 11. CORS Configuration

**Enable CORS for web apps:**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 12. Testing

**Test with TestClient:**

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"text": "test input", "model_name": "default"}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()

def test_invalid_input():
    response = client.post(
        "/predict",
        json={"text": ""}  # Empty text
    )
    assert response.status_code == 422  # Validation error
```

### 13. Deployment

**Production server:**

```bash
# Use Gunicorn with Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Docker deployment:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Environment variables:**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    model_path: str = "models/"
    max_batch_size: int = 32

    class Config:
        env_file = ".env"

settings = Settings()
```

## Common Patterns for ML APIs

### 1. Model Registry Pattern

```python
from typing import Dict
from enum import Enum

class ModelName(str, Enum):
    SENTIMENT = "sentiment"
    NER = "ner"
    EMBEDDINGS = "embeddings"

class ModelRegistry:
    def __init__(self):
        self.models: Dict = {}

    def register(self, name: str, model):
        self.models[name] = model

    def get(self, name: str):
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        return self.models[name]

registry = ModelRegistry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models
    registry.register("sentiment", load_sentiment_model())
    registry.register("ner", load_ner_model())
    yield
    registry.models.clear()

@app.post("/predict/{model_name}")
def predict(model_name: ModelName, text: str):
    model = registry.get(model_name.value)
    return model.predict(text)
```

### 2. Batch Processing Pattern

```python
@app.post("/predict-batch")
async def predict_batch(
    texts: List[str],
    batch_size: int = 32
):
    # Process in batches to avoid memory issues
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = ml_models["embedder"].encode(batch)
        results.extend(batch_results.tolist())

    return {"predictions": results, "count": len(results)}
```

### 3. Caching Pattern

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> List[float]:
    return ml_models["embedder"].encode(text).tolist()

@app.post("/embed-cached")
def embed_cached(text: str):
    # Automatically cached for repeated requests
    embedding = get_cached_embedding(text)
    return {"embedding": embedding}
```

### 4. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_limited(request: Request, text: str):
    return {"prediction": ml_models["model"].predict(text)}
```

## Best Practices

### 1. Model Loading
- Load models at startup, not per request
- Use lifespan events for initialization
- Consider model lazy loading for memory efficiency

### 2. Request Validation
- Use Pydantic for all request/response models
- Validate file sizes and types
- Set reasonable limits (text length, batch size)

### 3. Error Handling
- Return meaningful error messages
- Use appropriate HTTP status codes
- Log errors for debugging

### 4. Performance
- Use async for I/O operations
- Batch predictions when possible
- Implement caching for repeated requests
- Use background tasks for long operations

### 5. Security
- Never expose API keys in code
- Use environment variables
- Implement rate limiting
- Validate all inputs

### 6. Documentation
- Write clear endpoint descriptions
- Provide example requests/responses
- Document error codes

## Real-World Use Cases

### 1. Text Classification API
```python
@app.post("/classify")
async def classify_text(text: str):
    embedding = ml_models["embedder"].encode(text)
    category = ml_models["classifier"].predict([embedding])[0]
    return {"text": text, "category": category}
```

### 2. Semantic Search API
```python
@app.post("/search")
async def semantic_search(
    query: str,
    top_k: int = 10
):
    query_embedding = ml_models["embedder"].encode(query)
    results = vector_db.search(query_embedding, top_k)
    return {"results": results}
```

### 3. Batch Processing API
```python
@app.post("/process-batch")
async def process_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile
):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(process_file_async, file, job_id)
    return {"job_id": job_id, "status": "processing"}

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    status = check_job_status(job_id)
    return {"job_id": job_id, "status": status}
```

## Documentation Links

- [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Server](https://www.uvicorn.org/)
- [Starlette Framework](https://www.starlette.io/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)

## Next Steps

After completing this module:
1. Module 11: Testing & Code Quality - test your FastAPI apps
2. Module 6: RAG - build RAG APIs with FastAPI
3. Module 12: Spark - scale ML pipelines
4. Module 15: NER with GLiNER - deploy NER models as APIs
