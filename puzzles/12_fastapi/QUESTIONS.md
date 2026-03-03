# Module 10: FastAPI for ML - Deep Dive Questions

## Architecture & Design (Q1-Q10)

### Q1: You're deploying a sentiment analysis model via FastAPI. The model takes 2 seconds to load and uses 500MB RAM. You expect 100 requests/second. How would you design the service to handle this load efficiently?

**Answer:**

**Key challenges:**
1. Model loading is expensive (2s + 500MB)
2. High throughput requirement (100 req/s)
3. Need to avoid loading model per request

**Solution: Global model instance with proper lifecycle management**

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
from transformers import pipeline

# Global variable for model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown"""
    global model

    # Startup: Load model once
    print("Loading sentiment model...")
    model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model loaded successfully")

    yield  # Server runs here

    # Shutdown: Clean up
    print("Cleaning up model...")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(text: str):
    """Predict sentiment using globally loaded model"""
    result = model(text)[0]
    return {
        "label": result["label"],
        "score": result["score"]
    }
```

**For higher throughput (100 req/s), add batching:**

```python
from fastapi import BackgroundTasks
import asyncio
from typing import List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PendingRequest:
    text: str
    future: asyncio.Future
    timestamp: datetime

# Global batch queue
batch_queue: List[PendingRequest] = []
batch_lock = asyncio.Lock()
MAX_BATCH_SIZE = 32
MAX_WAIT_MS = 50  # Maximum wait time before processing

async def batch_processor():
    """Background task that processes requests in batches"""
    while True:
        await asyncio.sleep(0.01)  # Check every 10ms

        async with batch_lock:
            if not batch_queue:
                continue

            # Check if we should process (batch full or timeout)
            should_process = (
                len(batch_queue) >= MAX_BATCH_SIZE or
                (datetime.now() - batch_queue[0].timestamp).total_seconds() * 1000 > MAX_WAIT_MS
            )

            if not should_process:
                continue

            # Extract batch
            batch = batch_queue[:MAX_BATCH_SIZE]
            batch_queue[:MAX_BATCH_SIZE] = []

            # Process batch
            texts = [req.text for req in batch]
            results = model(texts)

            # Return results to waiting requests
            for req, result in zip(batch, results):
                req.future.set_result({
                    "label": result["label"],
                    "score": result["score"]
                })

@app.on_event("startup")
async def start_batch_processor():
    """Start background batch processing"""
    asyncio.create_task(batch_processor())

@app.post("/predict")
async def predict_batched(text: str):
    """Add request to batch queue and wait for result"""
    future = asyncio.Future()

    async with batch_lock:
        batch_queue.append(PendingRequest(
            text=text,
            future=future,
            timestamp=datetime.now()
        ))

    # Wait for batch processor to handle this request
    result = await future
    return result
```

**Deployment configuration (using Gunicorn + Uvicorn):**

```bash
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download model
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

COPY . .

# Run with 4 workers (each loads model once)
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

**Performance comparison:**

| Approach | Throughput | Latency (p95) | Memory |
|----------|-----------|---------------|--------|
| Load per request | 0.5 req/s | 2000ms | 500MB |
| Global model | 50 req/s | 20ms | 500MB |
| Global + batching | 200 req/s | 80ms | 500MB |
| 4 workers + batching | 800 req/s | 100ms | 2GB |

**Recommendation:**
- **< 50 req/s:** Global model, single worker
- **50-500 req/s:** Global model + batching, multiple workers
- **> 500 req/s:** Add load balancer + horizontal scaling

---

### Q2: Design a FastAPI service that serves 3 different models (classification, NER, summarization). Should you use one app with multiple endpoints or separate services? Explain trade-offs with code examples.

**Answer:**

**Option 1: Monolithic - All models in one service (SIMPLE)**

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import pipeline

class ModelManager:
    def __init__(self):
        self.classifier = None
        self.ner = None
        self.summarizer = None

    def load_all(self):
        print("Loading all models...")
        self.classifier = pipeline("text-classification")
        self.ner = pipeline("ner")
        self.summarizer = pipeline("summarization")
        print("All models loaded")

models = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load_all()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/classify")
async def classify(text: str):
    return models.classifier(text)

@app.post("/ner")
async def ner(text: str):
    return models.ner(text)

@app.post("/summarize")
async def summarize(text: str, max_length: int = 130):
    return models.summarizer(text, max_length=max_length)
```

**Pros:**
- Simple deployment (one container)
- Shared code and utilities
- Single endpoint for health checks

**Cons:**
- All models always loaded (high memory even if only using one)
- Can't scale models independently
- One model crash kills all services
- Deployment requires reloading all models

**Option 2: Microservices - Separate service per model (SCALABLE)**

```python
# classifier_service.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import pipeline

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = pipeline("text-classification")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(text: str):
    return model(text)

# ner_service.py - Similar structure
# summarization_service.py - Similar structure
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  classifier:
    build: ./classifier
    ports:
      - "8001:8000"
    deploy:
      resources:
        limits:
          memory: 1G

  ner:
    build: ./ner
    ports:
      - "8002:8000"
    deploy:
      resources:
        limits:
          memory: 2G

  summarizer:
    build: ./summarizer
    ports:
      - "8003:8000"
    deploy:
      resources:
        limits:
          memory: 3G

  gateway:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

```nginx
# nginx.conf - API Gateway
http {
  upstream classifier {
    server classifier:8000;
  }

  upstream ner {
    server ner:8000;
  }

  upstream summarizer {
    server summarizer:8000;
  }

  server {
    listen 80;

    location /classify {
      proxy_pass http://classifier/predict;
    }

    location /ner {
      proxy_pass http://ner/predict;
    }

    location /summarize {
      proxy_pass http://summarizer/predict;
    }
  }
}
```

**Pros:**
- Independent scaling (scale only high-traffic models)
- Isolated failures
- Independent deployments
- Right-sized resources per model

**Cons:**
- More complex deployment
- Need API gateway or service mesh
- More containers to manage

**Option 3: Hybrid - One app with lazy loading (BALANCED)**

```python
from fastapi import FastAPI
from typing import Optional
from functools import lru_cache

class LazyModelManager:
    def __init__(self):
        self._classifier: Optional[Any] = None
        self._ner: Optional[Any] = None
        self._summarizer: Optional[Any] = None

    @property
    def classifier(self):
        if self._classifier is None:
            print("Loading classifier on first use...")
            self._classifier = pipeline("text-classification")
        return self._classifier

    @property
    def ner(self):
        if self._ner is None:
            print("Loading NER on first use...")
            self._ner = pipeline("ner")
        return self._ner

    @property
    def summarizer(self):
        if self._summarizer is None:
            print("Loading summarizer on first use...")
            self._summarizer = pipeline("summarization")
        return self._summarizer

models = LazyModelManager()

app = FastAPI()

@app.post("/classify")
async def classify(text: str):
    return models.classifier(text)

@app.post("/ner")
async def ner(text: str):
    return models.ner(text)

@app.post("/summarize")
async def summarize(text: str):
    return models.summarizer(text)
```

**Comparison:**

| Aspect | Monolithic | Microservices | Hybrid |
|--------|-----------|---------------|--------|
| Memory (all models) | 6GB always | 6GB total (distributed) | 6GB if all used |
| Memory (1 model) | 6GB | 2GB | 2GB |
| First request latency | Fast | Fast | Slow (lazy load) |
| Scaling flexibility | ❌ | ✅ | ⚠️ |
| Deployment complexity | ✅ Simple | ❌ Complex | ✅ Simple |
| Fault isolation | ❌ | ✅ | ❌ |

**Recommendation:**

| Scenario | Choice |
|----------|--------|
| Prototype/MVP | Monolithic |
| All models used equally | Monolithic |
| Different traffic patterns | Microservices |
| One dominant model | Microservices |
| Limited DevOps resources | Hybrid (lazy) |
| Need independent scaling | Microservices |

---

### Q3: Your ML API receives file uploads (images, PDFs) for processing. Design the endpoint to handle: (1) Large files (500MB), (2) Multiple files in one request, (3) Progress tracking. Show implementation with streaming and background tasks.

**Answer:**

**Implementation with streaming, chunking, and background tasks:**

```python
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import aiofiles
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

app = FastAPI()

# Configuration
UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: float  # 0-100
    files_processed: int
    total_files: int
    result: dict = {}
    error: str = None

# In-memory job store (use Redis in production)
jobs: dict[str, JobInfo] = {}

async def save_upload_file(upload_file: UploadFile, destination: Path) -> tuple[int, str]:
    """
    Save uploaded file in chunks and return size + hash
    """
    size = 0
    hash_md5 = hashlib.md5()

    async with aiofiles.open(destination, 'wb') as f:
        while chunk := await upload_file.read(CHUNK_SIZE):
            size += len(chunk)

            # Check file size limit
            if size > MAX_FILE_SIZE:
                await f.close()
                destination.unlink()  # Delete partial file
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024**2):.0f}MB"
                )

            await f.write(chunk)
            hash_md5.update(chunk)

    return size, hash_md5.hexdigest()

async def process_file(file_path: Path, job_id: str, file_idx: int, total_files: int):
    """
    Background task to process uploaded file (placeholder for ML inference)
    """
    try:
        # Update status
        jobs[job_id].status = JobStatus.PROCESSING

        # Simulate ML processing
        import asyncio
        await asyncio.sleep(2)  # Replace with actual model inference

        # Update progress
        jobs[job_id].files_processed = file_idx + 1
        jobs[job_id].progress = ((file_idx + 1) / total_files) * 100

        # Simulate result
        result = {
            "filename": file_path.name,
            "predictions": ["class_A", "class_B"],
            "confidence": 0.95
        }

        jobs[job_id].result[file_path.name] = result

        # Mark complete if last file
        if file_idx + 1 == total_files:
            jobs[job_id].status = JobStatus.COMPLETED

    except Exception as e:
        jobs[job_id].status = JobStatus.FAILED
        jobs[job_id].error = str(e)

@app.post("/upload/single")
async def upload_single_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a single file"""

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}"
        )

    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        files_processed=0,
        total_files=1
    )

    # Save file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    try:
        size, file_hash = await save_upload_file(file, file_path)
    except HTTPException as e:
        del jobs[job_id]
        raise e

    # Process in background
    background_tasks.add_task(process_file, file_path, job_id, 0, 1)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "size_mb": size / (1024**2),
        "hash": file_hash,
        "message": "File uploaded. Processing started.",
        "status_url": f"/jobs/{job_id}"
    }

@app.post("/upload/batch")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process multiple files in parallel"""

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per request"
        )

    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        files_processed=0,
        total_files=len(files)
    )

    # Save all files
    saved_files = []
    for idx, file in enumerate(files):
        file_path = UPLOAD_DIR / f"{job_id}_{idx}_{file.filename}"

        try:
            size, file_hash = await save_upload_file(file, file_path)
            saved_files.append({
                "path": file_path,
                "filename": file.filename,
                "size_mb": size / (1024**2),
                "hash": file_hash
            })
        except HTTPException as e:
            # Clean up already saved files
            for saved in saved_files:
                saved["path"].unlink(missing_ok=True)
            del jobs[job_id]
            raise e

    # Process all files in background
    for idx, saved in enumerate(saved_files):
        background_tasks.add_task(
            process_file,
            saved["path"],
            job_id,
            idx,
            len(files)
        )

    return {
        "job_id": job_id,
        "files_uploaded": len(files),
        "total_size_mb": sum(f["size_mb"] for f in saved_files),
        "files": saved_files,
        "message": "Files uploaded. Processing started.",
        "status_url": f"/jobs/{job_id}"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status and progress"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]

@app.post("/upload/stream")
async def upload_stream(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload with real-time progress tracking (for very large files)
    Uses Server-Sent Events (SSE) for progress updates
    """
    from fastapi.responses import StreamingResponse
    import json

    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    async def progress_generator():
        """Stream progress updates"""
        size = 0
        hash_md5 = hashlib.md5()

        try:
            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(CHUNK_SIZE):
                    size += len(chunk)
                    await f.write(chunk)
                    hash_md5.update(chunk)

                    # Send progress update
                    progress = {
                        "bytes_uploaded": size,
                        "mb_uploaded": size / (1024**2)
                    }
                    yield f"data: {json.dumps(progress)}\n\n"

            # Upload complete
            yield f"data: {json.dumps({'status': 'complete', 'hash': hash_md5.hexdigest()})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream"
    )

@app.delete("/jobs/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up job files and data"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete files
    for file_path in UPLOAD_DIR.glob(f"{job_id}_*"):
        file_path.unlink()

    # Remove job
    del jobs[job_id]

    return {"message": "Job cleaned up"}
```

**Client usage example:**

```python
import requests

# Single file upload
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload/single",
        files={"file": f}
    )
    job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
print(f"Progress: {status['progress']:.1f}%")

# Multiple files
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("document.pdf", "rb"))
]
response = requests.post(
    "http://localhost:8000/upload/batch",
    files=files
)
```

**Key features:**
- Chunked reading (handles large files without OOM)
- Size validation during upload (fails fast)
- File hash for integrity verification
- Background processing with progress tracking
- Batch upload support
- SSE streaming for real-time progress
- Proper cleanup

---

### Q4: Compare synchronous vs asynchronous endpoints for ML inference. When should you use async? Show examples with I/O-bound (API calls) and CPU-bound (model inference) operations.

**Answer:**

**Key differences:**

| Aspect | Sync (`def`) | Async (`async def`) |
|--------|--------------|---------------------|
| Execution | Blocks thread | Non-blocking |
| Best for | CPU-bound tasks | I/O-bound tasks |
| Concurrency | One at a time per worker | Many concurrent per worker |
| Model inference | ✅ Good | ⚠️ Depends |
| API calls | ❌ Blocks | ✅ Excellent |
| File I/O | ⚠️ Blocks | ✅ Good |
| Database queries | ⚠️ Blocks | ✅ Good (with async driver) |

**Example 1: CPU-bound ML inference (use SYNC)**

```python
from fastapi import FastAPI
import torch
from transformers import pipeline

app = FastAPI()
model = pipeline("text-classification")

# ✅ CORRECT: Sync for CPU-bound work
@app.post("/predict/sync")
def predict_sync(text: str):
    """Synchronous endpoint for model inference"""
    # CPU-bound: model inference blocks
    result = model(text)[0]
    return result

# ❌ WRONG: Async doesn't help CPU-bound work
@app.post("/predict/async-wrong")
async def predict_async_wrong(text: str):
    """Async doesn't help here - model inference still blocks!"""
    result = model(text)[0]  # Still blocks the event loop
    return result

# ✅ CORRECT: Async with thread pool for CPU work
from concurrent.futures import ThreadPoolExecutor
import asyncio

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict/async-correct")
async def predict_async_correct(text: str):
    """Properly offload CPU work to thread pool"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: model(text)[0]
    )
    return result
```

**Performance comparison:**

```python
import asyncio
import time
import httpx

async def benchmark():
    urls = ["http://localhost:8000/predict/sync",
            "http://localhost:8000/predict/async-wrong",
            "http://localhost:8000/predict/async-correct"]

    for url in urls:
        start = time.time()

        async with httpx.AsyncClient() as client:
            # Send 10 concurrent requests
            tasks = [
                client.post(url, json={"text": f"Test {i}"})
                for i in range(10)
            ]
            await asyncio.gather(*tasks)

        elapsed = time.time() - start
        print(f"{url}: {elapsed:.2f}s")

# Results (single Uvicorn worker):
# /predict/sync: 2.0s (processes sequentially)
# /predict/async-wrong: 2.0s (async but blocks event loop)
# /predict/async-correct: 2.0s (offloads to threads, but no real benefit)

# With multiple Uvicorn workers (4 workers):
# /predict/sync: 0.5s (parallel across workers)
# /predict/async-wrong: 0.5s (parallel across workers)
# /predict/async-correct: 0.6s (thread pool overhead)
```

**Verdict for CPU-bound:** Use **sync** with multiple Uvicorn workers.

---

**Example 2: I/O-bound operations (use ASYNC)**

```python
import httpx
import asyncio

# ❌ WRONG: Sync blocks during API call
@app.post("/enrich/sync")
def enrich_sync(text: str):
    """Blocks entire worker during API call"""
    # Synchronous API call - blocks for network I/O
    response = httpx.get(f"https://api.example.com/analyze?text={text}")
    return response.json()

# ✅ CORRECT: Async for I/O-bound work
@app.post("/enrich/async")
async def enrich_async(text: str):
    """Non-blocking during API call - can handle other requests"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/analyze?text={text}"
        )
    return response.json()
```

**Performance comparison:**

```python
# 10 concurrent requests, each waits 1s for external API

# Sync (1 worker): 10s (sequential)
# Sync (10 workers): 1s (parallel workers)

# Async (1 worker): 1s (concurrent on single worker)
# Async (10 workers): 1s (same, single worker handles all)
```

**Verdict for I/O-bound:** Use **async** - single worker handles many concurrent requests.

---

**Example 3: Mixed workload (I/O + CPU)**

```python
from typing import List

# Scenario: Call external API, then run local model

# ❌ BAD: All sync
@app.post("/pipeline/all-sync")
def pipeline_all_sync(text: str):
    # Blocks during API call
    enriched = httpx.get(f"https://api.example.com/enrich?text={text}").json()

    # Blocks during inference
    result = model(enriched["text"])[0]
    return result

# ⚠️ OKAY: Async I/O, sync CPU
@app.post("/pipeline/hybrid")
async def pipeline_hybrid(text: str):
    # Non-blocking API call
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/enrich?text={text}")
        enriched = response.json()

    # Blocks during inference (but I/O was async)
    result = model(enriched["text"])[0]
    return result

# ✅ BEST: Async I/O, offload CPU to thread pool
@app.post("/pipeline/best")
async def pipeline_best(text: str):
    # Non-blocking API call
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/enrich?text={text}")
        enriched = response.json()

    # Offload CPU work to thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: model(enriched["text"])[0]
    )
    return result

# ✅ EXCELLENT: Parallel API calls + batch inference
@app.post("/pipeline/batch")
async def pipeline_batch(texts: List[str]):
    # Call APIs in parallel
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f"https://api.example.com/enrich?text={text}")
            for text in texts
        ]
        responses = await asyncio.gather(*tasks)
        enriched_texts = [r.json()["text"] for r in responses]

    # Batch inference (single model call)
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        executor,
        lambda: model(enriched_texts)
    )
    return results
```

---

**Decision tree:**

```
Is your endpoint I/O-bound (API calls, DB queries, file I/O)?
├─ YES → Use async
│   └─ Do you also have CPU work (model inference)?
│       ├─ YES → Use async + run_in_executor for CPU
│       └─ NO → Pure async is fine
│
└─ NO (pure CPU-bound) → Use sync
    └─ Scale with multiple Uvicorn workers
```

**Production recommendations:**

```python
# For ML API with external enrichment
# Run with: gunicorn -w 4 -k uvicorn.workers.UvicornWorker

@app.post("/predict")
async def predict(text: str):
    # Async for I/O
    async with httpx.AsyncClient() as client:
        enriched = await client.get(f"https://api/enrich?text={text}")

    # Sync CPU work is OK - Gunicorn has multiple workers
    result = model(enriched.json()["text"])[0]
    return result
```

**Summary:**
- **Pure ML inference:** Use sync endpoints with multiple workers
- **External API calls:** Use async endpoints
- **Mixed workload:** Use async for I/O, accept blocking CPU or offload to executor
- **Batch inference:** Always better than individual predictions

---

### Q5: Design a rate limiting strategy for an ML API that has different tiers (free: 100 req/day, pro: 10k req/day, enterprise: unlimited). Include implementation with Redis and handling of burst traffic.

**Answer:**

**Implementation with Redis (token bucket algorithm):**

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import Optional
import hashlib

app = FastAPI()

# Redis connection
redis_client = None

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )

@app.on_event("shutdown")
async def shutdown():
    await redis_client.close()

# Rate limit tiers
class RateLimitTier:
    FREE = {
        "requests_per_day": 100,
        "requests_per_minute": 10,  # Burst protection
        "name": "free"
    }
    PRO = {
        "requests_per_day": 10_000,
        "requests_per_minute": 100,
        "name": "pro"
    }
    ENTERPRISE = {
        "requests_per_day": None,  # Unlimited
        "requests_per_minute": 1000,
        "name": "enterprise"
    }

async def get_user_tier(api_key: str) -> dict:
    """Look up user tier from API key (simplified)"""
    # In production, query database
    tier_map = {
        "free_key_123": RateLimitTier.FREE,
        "pro_key_456": RateLimitTier.PRO,
        "ent_key_789": RateLimitTier.ENTERPRISE,
    }
    return tier_map.get(api_key, RateLimitTier.FREE)

async def check_rate_limit(
    api_key: str,
    tier: dict,
    endpoint: str
) -> tuple[bool, dict]:
    """
    Token bucket rate limiting with Redis

    Returns: (allowed, info_dict)
    """
    # Keys for daily and per-minute counters
    day_key = f"ratelimit:day:{api_key}:{datetime.now().strftime('%Y-%m-%d')}"
    minute_key = f"ratelimit:minute:{api_key}:{datetime.now().strftime('%Y-%m-%d:%H:%M')}"

    # Check daily limit
    if tier["requests_per_day"] is not None:
        day_count = await redis_client.incr(day_key)

        if day_count == 1:
            # First request of the day - set expiry
            await redis_client.expire(day_key, timedelta(days=1))

        if day_count > tier["requests_per_day"]:
            remaining = 0
            reset_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

            return False, {
                "allowed": False,
                "limit": tier["requests_per_day"],
                "remaining": 0,
                "reset": reset_time.isoformat(),
                "retry_after": int((reset_time - datetime.now()).total_seconds())
            }
    else:
        day_count = 0

    # Check per-minute limit (burst protection)
    minute_count = await redis_client.incr(minute_key)

    if minute_count == 1:
        # First request of the minute - set expiry
        await redis_client.expire(minute_key, timedelta(minutes=1))

    if minute_count > tier["requests_per_minute"]:
        reset_time = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)

        return False, {
            "allowed": False,
            "limit": tier["requests_per_minute"],
            "remaining": 0,
            "reset": reset_time.isoformat(),
            "retry_after": int((reset_time - datetime.now()).total_seconds()),
            "reason": "Too many requests per minute"
        }

    # Calculate remaining
    day_remaining = (
        tier["requests_per_day"] - day_count
        if tier["requests_per_day"] else None
    )
    minute_remaining = tier["requests_per_minute"] - minute_count

    return True, {
        "allowed": True,
        "day_limit": tier["requests_per_day"],
        "day_remaining": day_remaining,
        "day_used": day_count,
        "minute_limit": tier["requests_per_minute"],
        "minute_remaining": minute_remaining,
        "tier": tier["name"]
    }

async def rate_limit_dependency(
    x_api_key: Optional[str] = Header(None)
):
    """Dependency to enforce rate limiting"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header."
        )

    # Get user tier
    tier = await get_user_tier(x_api_key)

    # Check rate limit
    allowed, info = await check_rate_limit(x_api_key, tier, "/predict")

    if not allowed:
        # Return 429 with retry info
        raise HTTPException(
            status_code=429,
            detail=info["reason"] if "reason" in info else "Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": info["reset"],
                "Retry-After": str(info["retry_after"])
            }
        )

    return info

@app.post("/predict")
async def predict(
    text: str,
    rate_limit_info: dict = Depends(rate_limit_dependency)
):
    """ML prediction endpoint with rate limiting"""
    # Your model inference here
    result = {"prediction": "positive", "confidence": 0.95}

    # Include rate limit info in response headers
    return JSONResponse(
        content=result,
        headers={
            "X-RateLimit-Limit": str(rate_limit_info.get("day_limit", "unlimited")),
            "X-RateLimit-Remaining": str(rate_limit_info.get("day_remaining", "unlimited")),
            "X-RateLimit-Tier": rate_limit_info["tier"]
        }
    )

@app.get("/rate-limit/status")
async def get_rate_limit_status(
    x_api_key: str = Header(...)
):
    """Check current rate limit status"""
    tier = await get_user_tier(x_api_key)

    day_key = f"ratelimit:day:{x_api_key}:{datetime.now().strftime('%Y-%m-%d')}"
    minute_key = f"ratelimit:minute:{x_api_key}:{datetime.now().strftime('%Y-%m-%d:%H:%M')}"

    day_count = int(await redis_client.get(day_key) or 0)
    minute_count = int(await redis_client.get(minute_key) or 0)

    return {
        "tier": tier["name"],
        "daily": {
            "limit": tier["requests_per_day"],
            "used": day_count,
            "remaining": (tier["requests_per_day"] - day_count) if tier["requests_per_day"] else None
        },
        "per_minute": {
            "limit": tier["requests_per_minute"],
            "used": minute_count,
            "remaining": tier["requests_per_minute"] - minute_count
        }
    }
```

**Advanced: Sliding window rate limiting (more accurate):**

```python
async def check_rate_limit_sliding_window(
    api_key: str,
    tier: dict,
    window_seconds: int = 60
) -> tuple[bool, dict]:
    """
    Sliding window rate limiter using sorted sets in Redis
    More accurate than fixed windows
    """
    key = f"ratelimit:sliding:{api_key}"
    now = datetime.now().timestamp()
    window_start = now - window_seconds

    # Remove old requests outside window
    await redis_client.zremrangebyscore(key, 0, window_start)

    # Count requests in window
    count = await redis_client.zcard(key)

    limit = tier["requests_per_minute"]

    if count >= limit:
        # Get oldest request in window to calculate retry_after
        oldest = await redis_client.zrange(key, 0, 0, withscores=True)
        if oldest:
            retry_after = int(oldest[0][1] + window_seconds - now)
        else:
            retry_after = window_seconds

        return False, {
            "allowed": False,
            "limit": limit,
            "remaining": 0,
            "retry_after": retry_after
        }

    # Add current request
    await redis_client.zadd(key, {str(now): now})
    await redis_client.expire(key, window_seconds)

    return True, {
        "allowed": True,
        "limit": limit,
        "remaining": limit - count - 1
    }
```

**Client-side handling:**

```python
import httpx
import time

async def call_api_with_retry(text: str, api_key: str):
    """Client that respects rate limits"""
    headers = {"X-API-Key": api_key}

    async with httpx.AsyncClient() as client:
        while True:
            response = await client.post(
                "http://localhost:8000/predict",
                json={"text": text},
                headers=headers
            )

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Retrying after {retry_after}s...")
                await asyncio.sleep(retry_after)

            else:
                response.raise_for_status()
```

**Kubernetes deployment with Redis:**

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          limits:
            memory: 512Mi
            cpu: 500m
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

**Monitoring rate limits:**

```python
@app.get("/admin/rate-limits")
async def get_all_rate_limits(admin_key: str = Header(...)):
    """Admin endpoint to view all rate limits"""
    # Verify admin key
    if admin_key != "admin_secret":
        raise HTTPException(status_code=403)

    # Get all rate limit keys
    keys = []
    async for key in redis_client.scan_iter("ratelimit:*"):
        value = await redis_client.get(key)
        ttl = await redis_client.ttl(key)
        keys.append({
            "key": key,
            "count": value,
            "ttl_seconds": ttl
        })

    return keys
```

**Summary:**
- **Fixed window:** Simple, but allows bursts at window boundaries
- **Sliding window:** More accurate, uses sorted sets
- **Token bucket:** Best for allowing controlled bursts
- **Multiple limits:** Combine daily + per-minute for best UX
- **Redis:** Required for distributed systems (multiple API instances)

---

### Q6: Design a caching strategy for an ML API where predictions are deterministic. Consider cache invalidation when the model is updated. Show implementation with Redis and cache headers.

**Answer:**

**Caching strategy for ML predictions:**

1. **Response caching** - Cache model predictions
2. **Model versioning** - Invalidate cache on model updates  
3. **TTL** - Set reasonable expiration
4. **Cache headers** - Help client-side caching

**Implementation:**

```python
from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()

# Redis client
redis_client = None
MODEL_VERSION = "v1.2.0"  # Increment on model updates

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url("redis://localhost:6379")

async def get_cache_key(endpoint: str, params: dict, model_version: str) -> str:
    """Generate cache key from endpoint, params, and model version"""
    # Include model version in key for automatic invalidation
    cache_data = {
        "endpoint": endpoint,
        "params": params,
        "model_version": model_version
    }
    key_string = json.dumps(cache_data, sort_keys=True)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    return f"cache:{endpoint}:{key_hash}"

async def get_cached_response(cache_key: str) -> Optional[dict]:
    """Get cached response from Redis"""
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

async def set_cached_response(
    cache_key: str,
    response: dict,
    ttl_seconds: int = 3600
):
    """Store response in Redis with TTL"""
    await redis_client.setex(
        cache_key,
        ttl_seconds,
        json.dumps(response)
    )

@app.post("/predict")
async def predict(
    text: str,
    response: Response,
    if_none_match: Optional[str] = Header(None)
):
    """
    Predict with caching
    
    Uses ETag for client-side caching validation
    """
    # Generate cache key
    cache_key = await get_cache_key(
        endpoint="/predict",
        params={"text": text},
        model_version=MODEL_VERSION
    )
    
    # Generate ETag (hash of input + model version)
    etag = hashlib.md5(
        f"{text}:{MODEL_VERSION}".encode()
    ).hexdigest()
    
    # Check if client has valid cached version (HTTP 304)
    if if_none_match == etag:
        response.status_code = 304  # Not Modified
        return None
    
    # Check server-side cache
    cached_result = await get_cached_response(cache_key)
    if cached_result:
        # Cache hit - return cached result
        response.headers["X-Cache"] = "HIT"
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "private, max-age=3600"
        return cached_result
    
    # Cache miss - run model inference
    result = {
        "text": text,
        "prediction": "positive",  # Your model inference here
        "confidence": 0.95,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat()
    }
    
    # Store in cache
    await set_cached_response(cache_key, result, ttl_seconds=3600)
    
    # Set response headers for caching
    response.headers["X-Cache"] = "MISS"
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=3600"
    
    return result

@app.post("/predict/batch")
async def predict_batch(texts: list[str], response: Response):
    """
    Batch prediction with partial caching
    
    Cache individual predictions, only run model on cache misses
    """
    results = []
    uncached_texts = []
    uncached_indices = []
    
    # Check cache for each text
    for idx, text in enumerate(texts):
        cache_key = await get_cache_key(
            endpoint="/predict",
            params={"text": text},
            model_version=MODEL_VERSION
        )
        
        cached = await get_cached_response(cache_key)
        if cached:
            results.append(cached)
        else:
            results.append(None)  # Placeholder
            uncached_texts.append(text)
            uncached_indices.append(idx)
    
    # Run model only on uncached texts
    if uncached_texts:
        # Batch inference (your model here)
        new_predictions = [
            {
                "text": text,
                "prediction": "positive",
                "confidence": 0.95,
                "model_version": MODEL_VERSION
            }
            for text in uncached_texts
        ]
        
        # Cache new predictions and fill results
        for idx, pred in zip(uncached_indices, new_predictions):
            results[idx] = pred
            
            cache_key = await get_cache_key(
                endpoint="/predict",
                params={"text": pred["text"]},
                model_version=MODEL_VERSION
            )
            await set_cached_response(cache_key, pred)
    
    cache_hit_rate = (len(texts) - len(uncached_texts)) / len(texts) * 100
    
    response.headers["X-Cache-Hit-Rate"] = f"{cache_hit_rate:.1f}%"
    response.headers["X-Cached-Items"] = str(len(texts) - len(uncached_texts))
    response.headers["X-Uncached-Items"] = str(len(uncached_texts))
    
    return {
        "results": results,
        "cache_stats": {
            "total": len(texts),
            "cached": len(texts) - len(uncached_texts),
            "uncached": len(uncached_texts),
            "hit_rate": f"{cache_hit_rate:.1f}%"
        }
    }

@app.post("/admin/cache/invalidate")
async def invalidate_cache(
    pattern: str = "*",
    admin_key: str = Header(...)
):
    """
    Admin endpoint to invalidate cache
    
    Call this when deploying a new model version
    """
    if admin_key != "admin_secret":
        raise HTTPException(status_code=403)
    
    # Delete cache keys matching pattern
    deleted = 0
    async for key in redis_client.scan_iter(f"cache:{pattern}"):
        await redis_client.delete(key)
        deleted += 1
    
    return {
        "message": f"Invalidated {deleted} cache entries",
        "pattern": pattern
    }

@app.get("/admin/cache/stats")
async def get_cache_stats(admin_key: str = Header(...)):
    """Get cache statistics"""
    if admin_key != "admin_secret":
        raise HTTPException(status_code=403)
    
    # Count cache keys by endpoint
    stats = {}
    async for key in redis_client.scan_iter("cache:*"):
        endpoint = key.decode().split(":")[1]
        stats[endpoint] = stats.get(endpoint, 0) + 1
    
    # Get Redis memory usage
    info = await redis_client.info("memory")
    
    return {
        "cache_entries": stats,
        "total_entries": sum(stats.values()),
        "memory_used_mb": info["used_memory"] / (1024 ** 2),
        "model_version": MODEL_VERSION
    }
```

**Advanced: Cache warming on model deployment:**

```python
@app.post("/admin/cache/warm")
async def warm_cache(
    common_inputs: list[str],
    admin_key: str = Header(...)
):
    """
    Pre-populate cache with common inputs after model deployment
    """
    if admin_key != "admin_secret":
        raise HTTPException(status_code=403)
    
    results = []
    for text in common_inputs:
        # Run prediction (will cache result)
        result = await predict(text, Response())
        results.append(result)
    
    return {
        "message": f"Warmed cache with {len(common_inputs)} entries",
        "model_version": MODEL_VERSION
    }
```

**Client usage with caching:**

```python
import httpx

async def call_with_cache(text: str):
    """Client that respects ETag caching"""
    headers = {}
    
    # Check if we have cached ETag
    cached_etag = get_local_cache(text)
    if cached_etag:
        headers["If-None-Match"] = cached_etag
    
    response = await httpx.post(
        "http://localhost:8000/predict",
        json={"text": text},
        headers=headers
    )
    
    if response.status_code == 304:
        # Use local cached response
        return get_local_cache_response(text)
    
    # Cache ETag and response locally
    etag = response.headers.get("ETag")
    if etag:
        save_local_cache(text, etag, response.json())
    
    return response.json()
```

**Deployment strategy for model updates:**

```python
# 1. Deploy new model with new version
MODEL_VERSION = "v1.3.0"  # Increment version

# 2. Cache automatically invalidates (different version in key)
# Old keys: cache:/predict:hash_v1.2.0
# New keys: cache:/predict:hash_v1.3.0

# 3. Optional: Explicitly clear old cache to free memory
@app.on_event("startup")
async def clear_old_cache():
    # Delete cache entries from previous versions
    old_versions = ["v1.0.0", "v1.1.0", "v1.2.0"]
    for version in old_versions:
        async for key in redis_client.scan_iter(f"cache:*:{version}"):
            await redis_client.delete(key)
```

**Cache key strategies:**

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| Hash(input) | Simple | No model version tracking | Single model, no updates |
| Hash(input + version) | Auto-invalidation on updates | More cache misses | Frequent model updates |
| Hash(input + version + user) | User-specific caching | Higher memory usage | Personalized models |
| Semantic similarity | Cache similar inputs | Requires embedding comparison | Fuzzy matching |

**Recommendation:**
- Use `Hash(input + model_version)` for deterministic models
- Set TTL to 1 hour for active models, 1 day for stable models
- Implement cache warming for common queries after deployment
- Monitor cache hit rate (target > 70% for production traffic)

---

### Q7: You need to serve models that require GPU. Design a system that efficiently shares GPU resources across multiple requests. Consider: (1) Batching, (2) Queue management, (3) Timeouts, (4) Graceful degradation.

**Answer:**

**Challenge:** GPUs are expensive and underutilized with single-request inference.

**Solution: GPU batching with request queue**

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import torch
from transformers import pipeline
from collections import deque
from enum import Enum

app = FastAPI()

# Configuration
MAX_BATCH_SIZE = 32
MAX_WAIT_MS = 100  # Max time to wait for batch to fill
REQUEST_TIMEOUT_SECONDS = 30
GPU_MEMORY_LIMIT_GB = 8

class RequestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class GPURequest:
    request_id: str
    text: str
    future: asyncio.Future
    created_at: datetime
    status: RequestStatus = RequestStatus.PENDING
    
# Global queue and model
request_queue = deque()
queue_lock = asyncio.Lock()
model = None
gpu_stats = {
    "total_requests": 0,
    "batches_processed": 0,
    "avg_batch_size": 0,
    "gpu_utilization": 0.0
}

@app.on_event("startup")
async def startup():
    global model
    
    # Load model on GPU
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline(
        "text-classification",
        model="distilbert-base-uncased",
        device=device,
        batch_size=MAX_BATCH_SIZE
    )
    
    # Start background batch processor
    asyncio.create_task(gpu_batch_processor())
    asyncio.create_task(timeout_monitor())
    
    print(f"Model loaded on {'GPU' if device == 0 else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

async def gpu_batch_processor():
    """
    Background task that batches requests and processes on GPU
    """
    while True:
        await asyncio.sleep(0.01)  # Check every 10ms
        
        async with queue_lock:
            if not request_queue:
                continue
            
            # Check if should process batch
            should_process = (
                len(request_queue) >= MAX_BATCH_SIZE or
                (datetime.now() - request_queue[0].created_at).total_seconds() * 1000 > MAX_WAIT_MS
            )
            
            if not should_process:
                continue
            
            # Extract batch
            batch_size = min(len(request_queue), MAX_BATCH_SIZE)
            batch = [request_queue.popleft() for _ in range(batch_size)]
            
        # Process batch on GPU (outside lock to avoid blocking queue)
        try:
            # Update status
            for req in batch:
                req.status = RequestStatus.PROCESSING
            
            # Run inference
            texts = [req.text for req in batch]
            
            # Measure GPU utilization
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()
            
            start_time = datetime.now()
            results = model(texts)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated()
                mem_used = (end_mem - start_mem) / 1e9
            
            # Return results to waiting requests
            for req, result in zip(batch, results):
                req.status = RequestStatus.COMPLETED
                req.future.set_result({
                    "label": result["label"],
                    "score": result["score"],
                    "batch_size": len(batch),
                    "inference_time_ms": inference_time * 1000
                })
            
            # Update stats
            gpu_stats["total_requests"] += len(batch)
            gpu_stats["batches_processed"] += 1
            gpu_stats["avg_batch_size"] = (
                gpu_stats["total_requests"] / gpu_stats["batches_processed"]
            )
            
            if torch.cuda.is_available():
                gpu_stats["gpu_utilization"] = torch.cuda.utilization()
            
        except Exception as e:
            # Handle batch failure
            for req in batch:
                req.status = RequestStatus.FAILED
                if not req.future.done():
                    req.future.set_exception(e)

async def timeout_monitor():
    """Monitor and timeout old requests"""
    while True:
        await asyncio.sleep(1)  # Check every second
        
        now = datetime.now()
        timed_out = []
        
        async with queue_lock:
            # Find timed out requests
            for req in list(request_queue):
                age = (now - req.created_at).total_seconds()
                if age > REQUEST_TIMEOUT_SECONDS:
                    request_queue.remove(req)
                    timed_out.append(req)
        
        # Timeout requests outside lock
        for req in timed_out:
            req.status = RequestStatus.TIMEOUT
            if not req.future.done():
                req.future.set_exception(
                    HTTPException(
                        status_code=504,
                        detail=f"Request timeout after {REQUEST_TIMEOUT_SECONDS}s"
                    )
                )

@app.post("/predict")
async def predict(text: str):
    """
    Predict endpoint with GPU batching
    
    Adds request to queue and waits for batch processing
    """
    request_id = str(uuid.uuid4())
    future = asyncio.Future()
    
    # Add to queue
    async with queue_lock:
        request = GPURequest(
            request_id=request_id,
            text=text,
            future=future,
            created_at=datetime.now()
        )
        request_queue.append(request)
        queue_size = len(request_queue)
    
    # Wait for result
    try:
        result = await asyncio.wait_for(
            future,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timeout after {REQUEST_TIMEOUT_SECONDS}s"
        )

@app.post("/predict/streaming")
async def predict_streaming(text: str):
    """
    Streaming response for long-running inference
    
    Returns progress updates via Server-Sent Events
    """
    request_id = str(uuid.uuid4())
    
    async def event_generator():
        future = asyncio.Future()
        
        async with queue_lock:
            request = GPURequest(
                request_id=request_id,
                text=text,
                future=future,
                created_at=datetime.now()
            )
            request_queue.append(request)
            queue_size = len(request_queue)
        
        # Send queued event
        yield f"data: {json.dumps({'status': 'queued', 'position': queue_size})}\n\n"
        
        # Wait for result with periodic updates
        while not future.done():
            await asyncio.sleep(0.5)
            
            # Send progress update
            async with queue_lock:
                position = list(request_queue).index(request) + 1 if request in request_queue else 0
            
            if position > 0:
                yield f"data: {json.dumps({'status': 'waiting', 'position': position})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'processing'})}\n\n"
        
        # Send final result
        try:
            result = await future
            yield f"data: {json.dumps({'status': 'completed', 'result': result})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'failed', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/gpu/stats")
async def get_gpu_stats():
    """Get GPU utilization and queue statistics"""
    async with queue_lock:
        queue_size = len(request_queue)
        
        # Calculate queue age distribution
        now = datetime.now()
        queue_ages = [
            (now - req.created_at).total_seconds()
            for req in request_queue
        ]
    
    stats = {
        "queue": {
            "size": queue_size,
            "avg_age_seconds": sum(queue_ages) / len(queue_ages) if queue_ages else 0,
            "max_age_seconds": max(queue_ages) if queue_ages else 0
        },
        "processing": gpu_stats,
    }
    
    if torch.cuda.is_available():
        stats["gpu"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "utilization_percent": torch.cuda.utilization()
        }
    
    return stats

@app.post("/gpu/health")
async def gpu_health_check():
    """
    Health check endpoint with graceful degradation
    
    Returns unhealthy if queue is too full or GPU is out of memory
    """
    async with queue_lock:
        queue_size = len(request_queue)
    
    # Check queue size
    if queue_size > MAX_BATCH_SIZE * 10:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": f"Queue overloaded ({queue_size} requests)"
            }
        )
    
    # Check GPU memory
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        if memory_used > 0.95:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "reason": f"GPU memory critical ({memory_used*100:.1f}%)"
                }
            )
    
    return {"status": "healthy", "queue_size": queue_size}
```

**Kubernetes deployment with GPU:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-gpu-api
spec:
  replicas: 2  # 2 GPU instances
  selector:
    matchLabels:
      app: ml-gpu-api
  template:
    metadata:
      labels:
        app: ml-gpu-api
    spec:
      containers:
      - name: api
        image: ml-gpu-api:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU per pod
            memory: 16Gi
            cpu: 4
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
            cpu: 2
        env:
        - name: MAX_BATCH_SIZE
          value: "32"
        - name: MAX_WAIT_MS
          value: "100"
        livenessProbe:
          httpGet:
            path: /gpu/health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /gpu/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-gpu-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-gpu-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: gpu  # Scale based on GPU utilization
      target:
        type: Utilization
        averageUtilization: 70
```

**Performance comparison:**

| Configuration | Throughput | Avg Latency | GPU Util | Cost/hour |
|---------------|-----------|-------------|----------|-----------|
| No batching | 10 req/s | 100ms | 20% | $2.40 |
| Batch=8, wait=50ms | 80 req/s | 150ms | 60% | $2.40 |
| Batch=32, wait=100ms | 200 req/s | 200ms | 85% | $2.40 |
| 2 GPUs, batch=32 | 400 req/s | 200ms | 85% | $4.80 |

**Recommendation:**
- Start with batch=16, wait=50ms for balanced latency/throughput
- Monitor GPU utilization (target 70-80%)
- Use HPA to scale GPU pods based on queue size or GPU utilization
- Implement circuit breaker to fail fast when queue is full

---

### Q8: Design an ML API that supports both synchronous predictions and asynchronous batch jobs. Show how to handle long-running inference (5+ minutes) with job status tracking and result retrieval.

**Answer:**

**Architecture:**
1. **Sync endpoint** - Immediate response for fast inference (< 30s)
2. **Async endpoint** - Submit job, get job_id, poll for results
3. **Webhook callback** - Notify client when job completes
4. **Result storage** - Store results in database/S3

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from enum import Enum
from datetime import datetime
import asyncio
import httpx
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid

app = FastAPI()

# Database setup
DATABASE_URL = "postgresql://user:pass@localhost/mlapi"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(String, primary_key=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    input_data = Column(JSON)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    callback_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    progress = Column(JSON, nullable=True)  # {"current": 50, "total": 100}

Base.metadata.create_all(engine)

# Pydantic models
class PredictRequest(BaseModel):
    text: str

class BatchPredictRequest(BaseModel):
    texts: List[str]
    callback_url: Optional[HttpUrl] = None

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[dict] = None

# Sync endpoint for fast inference
@app.post("/predict")
async def predict_sync(request: PredictRequest):
    """
    Synchronous prediction endpoint
    
    Use for inference that completes in < 30 seconds
    """
    # Run model inference
    result = run_model_inference(request.text)  # Your model here
    
    return {
        "prediction": result,
        "execution_time_ms": 100
    }

# Async endpoint for batch/long-running jobs
@app.post("/predict/async", response_model=JobResponse)
async def predict_async(
    request: BatchPredictRequest,
    background_tasks: BackgroundTasks
):
    """
    Asynchronous batch prediction endpoint
    
    Returns job_id immediately, processes in background
    """
    # Create job
    job_id = str(uuid.uuid4())
    
    db = SessionLocal()
    job = Job(
        job_id=job_id,
        input_data=request.dict(),
        status=JobStatus.PENDING,
        callback_url=str(request.callback_url) if request.callback_url else None
    )
    db.add(job)
    db.commit()
    db.close()
    
    # Process in background
    background_tasks.add_task(
        process_batch_job,
        job_id,
        request.texts,
        request.callback_url
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=job.created_at
    )

async def process_batch_job(
    job_id: str,
    texts: List[str],
    callback_url: Optional[str] = None
):
    """
    Background task to process batch inference
    """
    db = SessionLocal()
    
    try:
        # Update status to processing
        job = db.query(Job).filter_by(job_id=job_id).first()
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()
        db.commit()
        
        # Process each text with progress tracking
        results = []
        total = len(texts)
        
        for idx, text in enumerate(texts):
            # Simulate long-running inference
            await asyncio.sleep(0.5)  # Replace with actual model inference
            result = {
                "text": text,
                "prediction": "positive",
                "confidence": 0.95
            }
            results.append(result)
            
            # Update progress
            job.progress = {"current": idx + 1, "total": total}
            db.commit()
        
        # Job completed
        job.status = JobStatus.COMPLETED
        job.result = {"predictions": results}
        job.completed_at = datetime.now()
        db.commit()
        
        # Send callback if provided
        if callback_url:
            await send_callback(callback_url, job_id, job.result)
        
    except Exception as e:
        # Job failed
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()
        db.commit()
        
        # Send failure callback
        if callback_url:
            await send_callback(callback_url, job_id, error=str(e))
    
    finally:
        db.close()

async def send_callback(callback_url: str, job_id: str, result: dict = None, error: str = None):
    """Send HTTP callback to notify job completion"""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                callback_url,
                json={
                    "job_id": job_id,
                    "status": "completed" if result else "failed",
                    "result": result,
                    "error": error
                },
                timeout=30
            )
    except Exception as e:
        print(f"Failed to send callback: {e}")

@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get job status and results
    
    Poll this endpoint to check job progress
    """
    db = SessionLocal()
    job = db.query(Job).filter_by(job_id=job_id).first()
    db.close()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
        progress=job.progress
    )

@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Stream job progress via Server-Sent Events
    
    Client connects once and receives real-time updates
    """
    from fastapi.responses import StreamingResponse
    import json
    
    async def event_generator():
        db = SessionLocal()
        
        while True:
            job = db.query(Job).filter_by(job_id=job_id).first()
            
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            # Send progress update
            data = {
                "status": job.status.value,
                "progress": job.progress,
                "result": job.result if job.status == JobStatus.COMPLETED else None,
                "error": job.error if job.status == JobStatus.FAILED else None
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            # Stop streaming if job is done
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            
            await asyncio.sleep(1)  # Update every second
        
        db.close()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a pending/running job
    """
    db = SessionLocal()
    job = db.query(Job).filter_by(job_id=job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Job already finished")
    
    job.status = JobStatus.FAILED
    job.error = "Cancelled by user"
    job.completed_at = datetime.now()
    db.commit()
    db.close()
    
    return {"message": "Job cancelled"}

@app.get("/jobs")
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all jobs with pagination
    """
    db = SessionLocal()
    
    query = db.query(Job)
    if status:
        query = query.filter_by(status=status)
    
    jobs = query.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()
    total = query.count()
    
    db.close()
    
    return {
        "jobs": [
            JobResponse(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                progress=job.progress
            )
            for job in jobs
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }
```

**Client usage patterns:**

**1. Polling pattern:**
```python
import httpx
import time

async def process_batch_polling(texts: List[str]):
    """Submit job and poll for results"""
    async with httpx.AsyncClient() as client:
        # Submit job
        response = await client.post(
            "http://localhost:8000/predict/async",
            json={"texts": texts}
        )
        job_id = response.json()["job_id"]
        print(f"Job submitted: {job_id}")
        
        # Poll for completion
        while True:
            response = await client.get(f"http://localhost:8000/jobs/{job_id}")
            job = response.json()
            
            print(f"Status: {job['status']}, Progress: {job.get('progress')}")
            
            if job["status"] in ["completed", "failed"]:
                return job
            
            await asyncio.sleep(2)  # Poll every 2 seconds
```

**2. Webhook callback pattern:**
```python
# Client server to receive callbacks
from flask import Flask, request

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    job_id = data["job_id"]
    status = data["status"]
    result = data.get("result")
    
    print(f"Job {job_id} finished with status {status}")
    print(f"Result: {result}")
    
    return {"received": True}

# Submit job with callback
async def process_batch_webhook(texts: List[str]):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/predict/async",
            json={
                "texts": texts,
                "callback_url": "http://client-server.com/webhook"
            }
        )
        job_id = response.json()["job_id"]
        print(f"Job submitted: {job_id}. Will receive callback when done.")
```

**3. Server-Sent Events streaming:**
```python
import httpx

async def process_batch_streaming(texts: List[str]):
    """Submit job and stream progress"""
    async with httpx.AsyncClient() as client:
        # Submit job
        response = await client.post(
            "http://localhost:8000/predict/async",
            json={"texts": texts}
        )
        job_id = response.json()["job_id"]
        
        # Stream progress
        async with client.stream(
            "GET",
            f"http://localhost:8000/jobs/{job_id}/stream"
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    print(f"Update: {data}")
                    
                    if data.get("status") in ["completed", "failed"]:
                        return data
```

**Summary:**

| Pattern | Pros | Cons | Use Case |
|---------|------|------|----------|
| Sync | Simple, immediate | Timeouts on long jobs | Fast inference (< 30s) |
| Async + Polling | Simple client | Inefficient (constant requests) | Batch jobs, infrequent checks |
| Async + Webhook | Efficient, real-time | Requires public endpoint | Production batch processing |
| Async + SSE | Real-time, efficient | Requires persistent connection | UI with progress bar |

---

### Q9: Compare FastAPI, Flask, and Django for ML model deployment. When would you choose each? Include specific scenarios with code examples showing key differences.

**Answer:**

**High-level comparison:**

| Aspect | FastAPI | Flask | Django |
|--------|---------|-------|--------|
| Performance | ⚡ Fastest (async) | Medium | Slowest |
| Learning curve | Medium | Easy | Hard |
| Async support | ✅ Native | ⚠️ Limited (Flask 2.0+) | ⚠️ Limited (Django 4.1+) |
| Auto docs | ✅ Built-in (Swagger) | ❌ Manual | ⚠️ DRF only |
| Type checking | ✅ Pydantic | ❌ | ⚠️ Serializers |
| Best for | Modern ML APIs | Simple prototypes | Full web apps |

---

**Scenario 1: Simple sentiment analysis API**

**FastAPI:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
model = pipeline("sentiment-analysis")

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictRequest):
    result = model(request.text)[0]
    return result

# Auto-generated docs at /docs
# Type validation built-in
# Run with: uvicorn main:app --reload
```

**Flask:**
```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
model = pipeline("sentiment-analysis")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Manual validation
    if "text" not in data:
        return jsonify({"error": "Missing text"}), 400
    
    result = model(data["text"])[0]
    return jsonify(result)

# No auto docs
# Manual validation
# Run with: flask run
```

**Django (with DRF):**
```python
# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import serializers
from transformers import pipeline

model = pipeline("sentiment-analysis")

class PredictSerializer(serializers.Serializer):
    text = serializers.CharField()

class PredictView(APIView):
    def post(self, request):
        serializer = PredictSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        result = model(serializer.validated_data["text"])[0]
        return Response(result)

# urls.py
from django.urls import path

urlpatterns = [
    path("predict/", PredictView.as_view())
]

# Requires settings.py, manage.py, etc.
# Most boilerplate code
```

**Verdict:** FastAPI wins for simplicity + features.

---

**Scenario 2: High-throughput image classification (1000 req/s)**

**FastAPI (async) - BEST:**
```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import asyncio
from typing import List

app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Async file reading (non-blocking)
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    # Offload CPU inference to thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.predict, image)
    
    return result

# Single worker handles many concurrent uploads
# Run with: uvicorn main:app --workers 4
```

**Flask (sync) - OKAY:**
```python
from flask import Flask, request
from PIL import Image

app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify():
    # Sync file reading (blocks)
    file = request.files["file"]
    image = Image.open(file.stream)
    
    # Sync inference (blocks)
    result = model.predict(image)
    
    return result

# Each worker handles one request at a time
# Run with: gunicorn -w 16 main:app  # Need more workers
```

**Benchmark:**
```
FastAPI (4 workers): 1000 req/s, 100ms p95
Flask (4 workers): 200 req/s, 500ms p95
Flask (16 workers): 800 req/s, 150ms p95 (uses 4x more memory)
```

**Verdict:** FastAPI handles async I/O better, needs fewer workers.

---

**Scenario 3: ML platform with user management, admin panel, job scheduling**

**Django - BEST:**
```python
# Django provides batteries-included solution

# models.py
from django.db import models
from django.contrib.auth.models import User

class MLJob(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20)
    input_data = models.JSONField()
    result = models.JSONField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ["-created_at"]

# admin.py - Auto-generated admin panel
from django.contrib import admin
admin.site.register(MLJob)

# views.py
from django.contrib.auth.decorators import login_required

@login_required
def submit_job(request):
    # Built-in auth, sessions, CSRF protection
    job = MLJob.objects.create(
        user=request.user,
        input_data=request.POST
    )
    return JsonResponse({"job_id": job.id})

# Celery integration for background jobs
from celery import shared_task

@shared_task
def process_job(job_id):
    job = MLJob.objects.get(id=job_id)
    # Process...
    job.result = result
    job.save()

# Built-in:
# - User authentication
# - Admin panel (/admin)
# - ORM with migrations
# - Celery integration
# - Scheduled tasks (django-crontab)
```

**FastAPI - OKAY (but more work):**
```python
# Need to add everything manually

from fastapi import FastAPI, Depends
from fastapi_users import FastAPIUsers  # External lib
from sqlalchemy import create_engine  # External lib
from celery import Celery  # External lib

app = FastAPI()

# Manual user management
# Manual admin panel (FastAPI-Admin)
# Manual ORM setup
# Manual Celery setup
# Manual task scheduling

# More flexible but more setup
```

**Verdict:** Django is better for full-stack ML platforms with web UI.

---

**Scenario 4: Prototype/MVP for demo**

**Flask - BEST:**
```python
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
model = pipeline("sentiment-analysis")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    result = model(text)[0]
    return render_template("result.html", result=result)

# templates/index.html
# <form method="POST" action="/predict">
#   <textarea name="text"></textarea>
#   <button>Analyze</button>
# </form>

# Simplest to get started
# Run: flask run
```

**FastAPI - Good but overkill:**
```python
# Need to set up templates manually
# More features than needed for simple demo
```

**Verdict:** Flask is fastest to prototype.

---

**Decision matrix:**

| Use Case | Framework | Why |
|----------|-----------|-----|
| REST API for ML model | FastAPI | Auto docs, validation, async |
| High-throughput API | FastAPI | Async, fewer workers needed |
| WebSocket streaming | FastAPI | Native async WebSocket support |
| Simple prototype/demo | Flask | Easiest to start |
| Legacy integration | Flask | Well-established, stable |
| Full ML platform | Django | User management, admin panel, ORM |
| Need background jobs | Django + Celery | Built-in integration |
| Complex business logic | Django | ORM, validation, transactions |

**Production recommendations:**

```python
# Modern ML API: FastAPI + Postgres + Redis + Celery
# ├── FastAPI (API layer)
# ├── PostgreSQL (persistent storage)
# ├── Redis (caching + message broker)
# └── Celery (background jobs)

# Full ML platform: Django + Celery + FastAPI
# ├── Django (web UI, user management, admin)
# ├── FastAPI (high-performance inference API)
# ├── Celery (background jobs)
# └── PostgreSQL (shared database)
```

**Summary:**
- **FastAPI:** Modern ML APIs, high performance, async
- **Flask:** Prototypes, simple APIs, microservices
- **Django:** Full platforms with web UI, user management, complex business logic

---

### Q10: Design an API versioning strategy for ML models. How do you handle: (1) Multiple model versions in production, (2) Gradual rollout of new models, (3) A/B testing, (4) Backward compatibility?

**Answer:**

**Challenge:** ML models evolve faster than traditional software. Need to:
- Serve multiple versions simultaneously
- Gracefully migrate users to new versions
- Allow A/B testing
- Maintain backward compatibility

**Strategy 1: URL Path Versioning**

```python
from fastapi import FastAPI, Header
from enum import Enum
from typing import Optional

app = FastAPI()

class ModelVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

# Load multiple model versions
models = {
    "v1": load_model("models/sentiment-v1.0.0"),
    "v2": load_model("models/sentiment-v2.0.0"),
    "v3": load_model("models/sentiment-v3.0.0"),  # Latest
}

# Version in URL path
@app.post("/v1/predict")
async def predict_v1(text: str):
    """Legacy endpoint for v1 model"""
    result = models["v1"](text)
    return {"prediction": result, "model_version": "v1"}

@app.post("/v2/predict")
async def predict_v2(text: str):
    """Current stable endpoint"""
    result = models["v2"](text)
    return {"prediction": result, "model_version": "v2"}

@app.post("/v3/predict")
async def predict_v3(text: str):
    """Beta endpoint for new model"""
    result = models["v3"](text)
    return {"prediction": result, "model_version": "v3"}

# Default route (latest stable)
@app.post("/predict")
async def predict_default(text: str):
    """Default endpoint - redirects to latest stable (v2)"""
    return await predict_v2(text)
```

**Pros:**
- Clear and explicit
- Easy to deprecate old versions
- Simple client-side migration

**Cons:**
- URL bloat
- Need separate endpoints per version

---

**Strategy 2: Header-based Versioning**

```python
from fastapi import FastAPI, Header
from typing import Optional

app = FastAPI()

@app.post("/predict")
async def predict(
    text: str,
    x_model_version: Optional[str] = Header("v2")  # Default to v2
):
    """
    Single endpoint with version in header
    
    Usage:
    curl -X POST http://api/predict \
      -H "X-Model-Version: v3" \
      -d '{"text": "Great product!"}'
    """
    # Route to appropriate model
    if x_model_version not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model version: {x_model_version}. Available: {list(models.keys())}"
        )
    
    model = models[x_model_version]
    result = model(text)
    
    return {
        "prediction": result,
        "model_version": x_model_version
    }
```

**Pros:**
- Single endpoint URL
- Easy to version via headers
- Backward compatible (default version)

**Cons:**
- Less discoverable than URL versioning
- Harder to cache (headers affect caching)

---

**Strategy 3: Gradual Rollout with Feature Flags**

```python
from fastapi import FastAPI, Depends, Header
import random

app = FastAPI()

# Rollout configuration
ROLLOUT_CONFIG = {
    "v3": {
        "enabled": True,
        "percentage": 10,  # 10% of traffic
        "allowlist": ["user_123", "user_456"],  # Specific users
        "blocklist": []
    }
}

def get_model_version(
    x_api_key: str = Header(...),
    x_user_id: Optional[str] = Header(None)
) -> str:
    """
    Determine model version based on rollout config
    
    Priority:
    1. Blocklist -> use stable version
    2. Allowlist -> use new version
    3. Random percentage -> use new version
    4. Default -> use stable version
    """
    config = ROLLOUT_CONFIG["v3"]
    
    if not config["enabled"]:
        return "v2"  # Stable version
    
    # Check blocklist
    if x_user_id in config["blocklist"]:
        return "v2"
    
    # Check allowlist
    if x_user_id in config["allowlist"]:
        return "v3"
    
    # Random percentage
    if random.random() * 100 < config["percentage"]:
        return "v3"
    
    return "v2"

@app.post("/predict")
async def predict(
    text: str,
    model_version: str = Depends(get_model_version)
):
    """
    Automatic model version selection based on rollout
    """
    model = models[model_version]
    result = model(text)
    
    return {
        "prediction": result,
        "model_version": model_version
    }

# Admin endpoint to update rollout
@app.post("/admin/rollout")
async def update_rollout(
    version: str,
    percentage: int,
    admin_key: str = Header(...)
):
    """
    Update rollout percentage for gradual deployment
    
    Example:
    - Start: 1%
    - Monitor metrics for 24h
    - Increase: 10%, 25%, 50%, 100%
    """
    if admin_key != "admin_secret":
        raise HTTPException(status_code=403)
    
    ROLLOUT_CONFIG[version]["percentage"] = percentage
    
    return {
        "message": f"Updated {version} rollout to {percentage}%",
        "config": ROLLOUT_CONFIG[version]
    }
```

**Gradual rollout plan:**

```python
# Day 1: Internal testing (allowlist)
ROLLOUT_CONFIG["v3"]["percentage"] = 0
ROLLOUT_CONFIG["v3"]["allowlist"] = ["internal_user_1", "internal_user_2"]

# Day 2: 1% of prod traffic
ROLLOUT_CONFIG["v3"]["percentage"] = 1

# Day 3: Monitor metrics, increase to 10%
ROLLOUT_CONFIG["v3"]["percentage"] = 10

# Day 5: 50% traffic (A/B testing)
ROLLOUT_CONFIG["v3"]["percentage"] = 50

# Day 7: 100% traffic (full rollout)
ROLLOUT_CONFIG["v3"]["percentage"] = 100

# Day 10: Remove v2, make v3 default
del models["v2"]
ROLLOUT_CONFIG["v3"]["percentage"] = 100
```

---

**Strategy 4: A/B Testing with Metrics**

```python
from fastapi import FastAPI
import random
from datetime import datetime

app = FastAPI()

# A/B test configuration
AB_TEST_CONFIG = {
    "sentiment_v2_vs_v3": {
        "enabled": True,
        "variant_a": "v2",  # Control
        "variant_b": "v3",  # Treatment
        "split": 0.5,  # 50/50 split
        "start_date": "2024-02-01",
        "end_date": "2024-02-14"
    }
}

# Store experiment results
experiment_results = []

def assign_variant(user_id: str, experiment: str) -> str:
    """
    Consistent variant assignment
    
    Same user always gets same variant (based on hash)
    """
    config = AB_TEST_CONFIG[experiment]
    
    # Hash user_id for consistent assignment
    hash_val = hash(f"{user_id}:{experiment}") % 100
    
    if hash_val < config["split"] * 100:
        return config["variant_a"]
    else:
        return config["variant_b"]

@app.post("/predict")
async def predict(
    text: str,
    x_user_id: str = Header(...)
):
    """
    Predict with A/B testing
    """
    # Assign variant
    experiment = "sentiment_v2_vs_v3"
    variant = assign_variant(x_user_id, experiment)
    
    # Run prediction
    start_time = datetime.now()
    model = models[variant]
    result = model(text)
    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    # Log experiment result
    experiment_results.append({
        "experiment": experiment,
        "user_id": x_user_id,
        "variant": variant,
        "input": text,
        "output": result,
        "latency_ms": latency_ms,
        "timestamp": datetime.now()
    })
    
    return {
        "prediction": result,
        "model_version": variant,
        "experiment": experiment
    }

@app.get("/admin/experiments/{experiment}/results")
async def get_experiment_results(
    experiment: str,
    admin_key: str = Header(...)
):
    """
    Analyze A/B test results
    """
    if admin_key != "admin_secret":
        raise HTTPException(status_code=403)
    
    # Filter results for experiment
    results = [r for r in experiment_results if r["experiment"] == experiment]
    
    # Calculate metrics per variant
    variants = {}
    for result in results:
        variant = result["variant"]
        if variant not in variants:
            variants[variant] = {
                "count": 0,
                "total_latency": 0,
                "predictions": []
            }
        
        variants[variant]["count"] += 1
        variants[variant]["total_latency"] += result["latency_ms"]
        variants[variant]["predictions"].append(result["output"])
    
    # Summary statistics
    summary = {}
    for variant, data in variants.items():
        summary[variant] = {
            "count": data["count"],
            "avg_latency_ms": data["total_latency"] / data["count"],
            # Add business metrics (accuracy, user satisfaction, etc.)
        }
    
    return {
        "experiment": experiment,
        "config": AB_TEST_CONFIG[experiment],
        "summary": summary
    }
```

---

**Strategy 5: Backward Compatibility with Adapters**

```python
from pydantic import BaseModel
from typing import Union

# V1 schema
class PredictRequestV1(BaseModel):
    text: str

# V2 schema (added language param)
class PredictRequestV2(BaseModel):
    text: str
    language: str = "en"

# V3 schema (changed structure)
class PredictRequestV3(BaseModel):
    inputs: list[str]
    options: dict = {}

# V1 response
class PredictResponseV1(BaseModel):
    label: str
    score: float

# V2 response (added explanation)
class PredictResponseV2(BaseModel):
    label: str
    score: float
    explanation: str

# Adapter pattern for backward compatibility
def adapt_v1_to_v3(request: PredictRequestV1) -> PredictRequestV3:
    """Convert V1 request to V3 format"""
    return PredictRequestV3(
        inputs=[request.text],
        options={"language": "en"}
    )

def adapt_v3_to_v1(response: list) -> PredictResponseV1:
    """Convert V3 response to V1 format"""
    return PredictResponseV1(
        label=response[0]["label"],
        score=response[0]["score"]
    )

@app.post("/v1/predict", response_model=PredictResponseV1)
async def predict_v1(request: PredictRequestV1):
    """
    V1 endpoint (legacy)
    
    Internally uses V3 model but adapts request/response
    """
    # Adapt request to V3 format
    v3_request = adapt_v1_to_v3(request)
    
    # Use V3 model
    v3_response = models["v3"](v3_request.inputs, **v3_request.options)
    
    # Adapt response to V1 format
    v1_response = adapt_v3_to_v1(v3_response)
    
    return v1_response

@app.post("/v3/predict")
async def predict_v3(request: PredictRequestV3):
    """V3 endpoint (latest)"""
    result = models["v3"](request.inputs, **request.options)
    return result
```

---

**Complete versioning system:**

```python
from fastapi import FastAPI, Header, Depends
from typing import Optional
import random

app = FastAPI()

# Model registry
MODEL_REGISTRY = {
    "v1.0.0": {"status": "deprecated", "sunset_date": "2024-06-01"},
    "v1.1.0": {"status": "deprecated", "sunset_date": "2024-06-01"},
    "v2.0.0": {"status": "stable", "traffic_percentage": 50},
    "v2.1.0": {"status": "stable", "traffic_percentage": 50},
    "v3.0.0": {"status": "beta", "traffic_percentage": 10},
}

def get_model_version(
    x_model_version: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_api_key: str = Header(...)
) -> str:
    """
    Smart model version selection
    
    Priority:
    1. Explicit version in header
    2. User-specific assignment (A/B test)
    3. Default stable version
    """
    # Explicit version
    if x_model_version:
        if x_model_version not in MODEL_REGISTRY:
            raise HTTPException(400, f"Unknown version: {x_model_version}")
        
        if MODEL_REGISTRY[x_model_version]["status"] == "deprecated":
            # Return warning header
            return x_model_version  # Still allow, but warn
        
        return x_model_version
    
    # A/B test assignment
    if x_user_id:
        # Consistent hashing for user
        hash_val = hash(x_user_id) % 100
        
        # Assign based on traffic percentages
        cumulative = 0
        for version, config in MODEL_REGISTRY.items():
            if config["status"] == "deprecated":
                continue
            
            cumulative += config.get("traffic_percentage", 0)
            if hash_val < cumulative:
                return version
    
    # Default to latest stable
    stable_versions = [
        v for v, c in MODEL_REGISTRY.items()
        if c["status"] == "stable"
    ]
    return max(stable_versions)  # Latest stable version

@app.post("/predict")
async def predict(
    text: str,
    model_version: str = Depends(get_model_version),
    response: Response
):
    """
    Universal predict endpoint with smart versioning
    """
    model = models[model_version]
    result = model(text)
    
    # Add version info to response headers
    response.headers["X-Model-Version"] = model_version
    response.headers["X-Model-Status"] = MODEL_REGISTRY[model_version]["status"]
    
    # Warn if deprecated
    if MODEL_REGISTRY[model_version]["status"] == "deprecated":
        sunset_date = MODEL_REGISTRY[model_version]["sunset_date"]
        response.headers["Warning"] = f"299 - \"Model version deprecated. Sunset date: {sunset_date}\""
    
    return result
```

**Summary:**

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| URL path | Clear versioning | Explicit, easy to cache | URL bloat |
| Header-based | Single endpoint | Clean URLs | Less discoverable |
| Feature flags | Gradual rollout | Safe deployment | Complex logic |
| A/B testing | Model comparison | Data-driven decisions | Requires metrics |
| Adapters | Backward compat | Smooth migration | Maintenance overhead |

**Recommendation:**
- Use **URL path** for major versions (v1, v2, v3)
- Use **feature flags** for gradual rollout within major version
- Use **A/B testing** to compare model performance
- Use **adapters** to maintain backward compatibility during migration


## Implementation & Coding (Q11-Q20)

### Q11: Implement a FastAPI endpoint with Pydantic validation that accepts nested JSON for batch text classification. Include custom validators for text length, language detection, and profanity filtering.

**Answer:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict
from enum import Enum
import re
from langdetect import detect

app = FastAPI()

class Language(str, Enum):
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    AUTO = "auto"

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    metadata: Optional[Dict[str, str]] = Field(default={}, description="Optional metadata")
    
    @validator("text")
    def validate_text_not_empty(cls, v):
        """Check text is not just whitespace"""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()
    
    @validator("text")
    def validate_no_profanity(cls, v):
        """Check for profanity (simple example)"""
        profanity_list = ["badword1", "badword2", "offensive"]  # In production, use better-words or profanity-check
        
        words = v.lower().split()
        found_profanity = [word for word in words if word in profanity_list]
        
        if found_profanity:
            raise ValueError(f"Text contains prohibited words: {', '.join(found_profanity)}")
        
        return v
    
    @validator("text")
    def validate_text_quality(cls, v):
        """Check text quality (not just repeated characters)"""
        # Check for excessive repeated characters
        if re.search(r'(.)\1{10,}', v):  # Same character 10+ times
            raise ValueError("Text contains excessive repeated characters")
        
        # Check for minimum word count
        words = v.split()
        if len(words) < 2:
            raise ValueError("Text must contain at least 2 words")
        
        return v

class BatchRequest(BaseModel):
    inputs: List[TextInput] = Field(..., min_items=1, max_items=100, description="Batch of texts to classify")
    language: Language = Field(default=Language.AUTO, description="Expected language of texts")
    model_version: str = Field(default="v1", regex=r"^v\d+$", description="Model version (e.g., v1, v2)")
    options: Dict[str, any] = Field(default={}, description="Additional options")
    
    @validator("inputs")
    def validate_unique_texts(cls, v):
        """Check for duplicate texts in batch"""
        texts = [item.text for item in v]
        if len(texts) != len(set(texts)):
            raise ValueError("Batch contains duplicate texts")
        return v
    
    @root_validator
    def validate_language_consistency(cls, values):
        """Check all texts are in expected language"""
        inputs = values.get("inputs")
        expected_lang = values.get("language")
        
        if expected_lang == Language.AUTO:
            return values
        
        # Detect language for each text
        for idx, item in enumerate(inputs):
            try:
                detected = detect(item.text)
                if detected != expected_lang.value:
                    raise ValueError(
                        f"Input {idx} detected language '{detected}' does not match expected '{expected_lang.value}'"
                    )
            except Exception as e:
                # Language detection failed
                pass  # Allow if detection fails
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {"text": "This product is amazing! Highly recommend.", "metadata": {"source": "review"}},
                    {"text": "Terrible experience, will not buy again.", "metadata": {"source": "review"}}
                ],
                "language": "en",
                "model_version": "v2",
                "options": {"return_probabilities": True}
            }
        }

class Prediction(BaseModel):
    text: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = None
    metadata: Dict[str, str] = {}

class BatchResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str
    processing_time_ms: float
    warnings: List[str] = []

@app.post("/classify/batch", response_model=BatchResponse)
async def classify_batch(request: BatchRequest):
    """
    Batch text classification with comprehensive validation
    
    Validation includes:
    - Text length (1-5000 chars)
    - No profanity
    - Text quality (no spam, repeated chars)
    - Language consistency
    - No duplicates in batch
    - Batch size limits (1-100)
    """
    import time
    start_time = time.time()
    
    warnings = []
    
    # Validate model version exists
    available_models = ["v1", "v2", "v3"]
    if request.model_version not in available_models:
        warnings.append(f"Unknown model version '{request.model_version}', using default 'v1'")
        request.model_version = "v1"
    
    # Process each text
    predictions = []
    for item in request.inputs:
        # Your model inference here
        result = {
            "text": item.text,
            "label": "positive",  # Mock prediction
            "confidence": 0.95,
            "metadata": item.metadata
        }
        
        # Include probabilities if requested
        if request.options.get("return_probabilities"):
            result["probabilities"] = {
                "positive": 0.95,
                "negative": 0.03,
                "neutral": 0.02
            }
        
        predictions.append(Prediction(**result))
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchResponse(
        predictions=predictions,
        model_version=request.model_version,
        processing_time_ms=processing_time,
        warnings=warnings
    )

# Custom error handling for validation errors
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Custom validation error response with detailed messages
    """
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        value = error.get("input")
        
        errors.append({
            "field": field,
            "message": message,
            "invalid_value": str(value)[:100] if value else None,
            "error_type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation failed",
            "errors": errors
        }
    )
```

**Example usage:**

```python
import httpx

# Valid request
valid_request = {
    "inputs": [
        {"text": "Great product, love it!", "metadata": {"source": "twitter"}},
        {"text": "Not satisfied with the quality.", "metadata": {"source": "email"}}
    ],
    "language": "en",
    "model_version": "v2"
}

response = httpx.post("http://localhost:8000/classify/batch", json=valid_request)
print(response.json())

# Invalid request - profanity
invalid_request = {
    "inputs": [
        {"text": "This is badword1 content"}  # Contains profanity
    ]
}

response = httpx.post("http://localhost:8000/classify/batch", json=invalid_request)
# Returns 422: "Text contains prohibited words: badword1"

# Invalid request - duplicate texts
duplicate_request = {
    "inputs": [
        {"text": "Same text"},
        {"text": "Same text"}  # Duplicate
    ]
}
# Returns 422: "Batch contains duplicate texts"
```

---

### Q12: Implement file upload handling for PDF, DOCX, and images with: (1) Size validation, (2) Content-type checking, (3) Virus scanning, (4) Async processing, (5) Progress tracking.

**Answer:**

```python
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
from enum import Enum
import aiofiles
import hashlib
import magic  # python-magic for content type detection
import subprocess
import uuid
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

class FileType(str, Enum):
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"

class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    SCANNING = "scanning"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Configuration
UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_TYPES = [FileType.PDF, FileType.DOCX, FileType.IMAGE_PNG, FileType.IMAGE_JPEG]
CHUNK_SIZE = 1024 * 1024  # 1MB

# Job tracking
jobs = {}

class JobInfo(BaseModel):
    job_id: str
    status: ProcessingStatus
    filename: str
    file_size: int
    content_type: str
    progress: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

async def validate_file_type(file: UploadFile) -> str:
    """
    Validate file type using magic numbers (not just extension)
    
    More secure than trusting Content-Type header or file extension
    """
    # Read first 2KB for magic number detection
    content = await file.read(2048)
    await file.seek(0)  # Reset file pointer
    
    # Detect actual content type
    mime = magic.from_buffer(content, mime=True)
    
    if mime not in [t.value for t in ALLOWED_TYPES]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {mime}. Allowed: {[t.value for t in ALLOWED_TYPES]}"
        )
    
    return mime

async def scan_virus(file_path: Path) -> bool:
    """
    Scan file for viruses using ClamAV
    
    Install ClamAV: apt-get install clamav clamav-daemon
    """
    try:
        # Run ClamAV scan
        result = subprocess.run(
            ["clamscan", "--no-summary", str(file_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check result
        if result.returncode == 0:
            return True  # Clean
        elif result.returncode == 1:
            return False  # Virus found
        else:
            raise Exception(f"Virus scan failed: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        raise Exception("Virus scan timeout")
    except FileNotFoundError:
        # ClamAV not installed - skip scan in dev
        print("Warning: ClamAV not installed, skipping virus scan")
        return True

async def save_upload(file: UploadFile, job_id: str) -> tuple[Path, int, str]:
    """
    Save uploaded file with validation
    
    Returns: (file_path, size, hash)
    """
    # Validate content type
    content_type = await validate_file_type(file)
    
    # Generate safe filename
    safe_filename = f"{job_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    # Save file in chunks with size validation
    size = 0
    hash_sha256 = hashlib.sha256()
    
    async with aiofiles.open(file_path, 'wb') as f:
        while chunk := await file.read(CHUNK_SIZE):
            size += len(chunk)
            
            # Check size limit
            if size > MAX_FILE_SIZE:
                await f.close()
                file_path.unlink()  # Delete partial file
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {size / (1024**2):.2f}MB. Max: {MAX_FILE_SIZE / (1024**2):.0f}MB"
                )
            
            await f.write(chunk)
            hash_sha256.update(chunk)
    
    return file_path, size, hash_sha256.hexdigest()

async def process_file(job_id: str, file_path: Path):
    """
    Background task to process uploaded file
    """
    try:
        # Update status: scanning
        jobs[job_id].status = ProcessingStatus.SCANNING
        jobs[job_id].progress = 10
        
        # Virus scan
        is_clean = await scan_virus(file_path)
        if not is_clean:
            jobs[job_id].status = ProcessingStatus.FAILED
            jobs[job_id].error = "Virus detected in file"
            file_path.unlink()  # Delete infected file
            return
        
        jobs[job_id].progress = 30
        
        # Update status: processing
        jobs[job_id].status = ProcessingStatus.PROCESSING
        
        # Process based on file type
        content_type = jobs[job_id].content_type
        
        if content_type == FileType.PDF:
            result = await process_pdf(file_path, job_id)
        elif content_type == FileType.DOCX:
            result = await process_docx(file_path, job_id)
        elif content_type in [FileType.IMAGE_PNG, FileType.IMAGE_JPEG]:
            result = await process_image(file_path, job_id)
        else:
            raise Exception(f"Unsupported type: {content_type}")
        
        # Completed
        jobs[job_id].status = ProcessingStatus.COMPLETED
        jobs[job_id].result = result
        jobs[job_id].progress = 100
        jobs[job_id].completed_at = datetime.now()
        
    except Exception as e:
        jobs[job_id].status = ProcessingStatus.FAILED
        jobs[job_id].error = str(e)
        jobs[job_id].completed_at = datetime.now()
    
    finally:
        # Clean up file
        if file_path.exists():
            file_path.unlink()

async def process_pdf(file_path: Path, job_id: str) -> dict:
    """Extract text from PDF"""
    import PyPDF2
    
    with open(file_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        num_pages = len(pdf.pages)
        
        text = ""
        for i, page in enumerate(pdf.pages):
            text += page.extract_text()
            
            # Update progress
            jobs[job_id].progress = 30 + (i / num_pages) * 60
    
    return {
        "type": "pdf",
        "num_pages": num_pages,
        "text_length": len(text),
        "preview": text[:500]
    }

async def process_docx(file_path: Path, job_id: str) -> dict:
    """Extract text from DOCX"""
    from docx import Document
    
    doc = Document(file_path)
    
    text = "\n".join([para.text for para in doc.paragraphs])
    jobs[job_id].progress = 80
    
    return {
        "type": "docx",
        "num_paragraphs": len(doc.paragraphs),
        "text_length": len(text),
        "preview": text[:500]
    }

async def process_image(file_path: Path, job_id: str) -> dict:
    """Process image with OCR"""
    from PIL import Image
    import pytesseract
    
    image = Image.open(file_path)
    jobs[job_id].progress = 60
    
    # Run OCR
    text = pytesseract.image_to_string(image)
    jobs[job_id].progress = 90
    
    return {
        "type": "image",
        "dimensions": f"{image.width}x{image.height}",
        "format": image.format,
        "text_length": len(text),
        "preview": text[:500]
    }

@app.post("/upload", response_model=JobInfo)
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload file for processing
    
    Supports: PDF, DOCX, PNG, JPEG
    Max size: 50MB
    """
    # Create job
    job_id = str(uuid.uuid4())
    
    # Save file
    try:
        file_path, size, file_hash = await save_upload(file, job_id)
    except HTTPException as e:
        raise e
    
    # Create job record
    jobs[job_id] = JobInfo(
        job_id=job_id,
        status=ProcessingStatus.UPLOADED,
        filename=file.filename,
        file_size=size,
        content_type=await validate_file_type(file),
        created_at=datetime.now()
    )
    
    # Process in background
    background_tasks.add_task(process_file, job_id, file_path)
    
    return jobs[job_id]

@app.post("/upload/batch")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload multiple files"""
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 files per batch")
    
    job_ids = []
    for file in files:
        response = await upload_file(file, background_tasks)
        job_ids.append(response.job_id)
    
    return {
        "job_ids": job_ids,
        "count": len(job_ids)
    }

@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    """Get job status and progress"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    return jobs[job_id]

@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """Stream job progress via SSE"""
    import json
    import asyncio
    
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    async def event_generator():
        while True:
            job = jobs[job_id]
            
            yield f"data: {json.dumps(job.dict())}\n\n"
            
            if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                break
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

**Client usage:**

```python
import httpx

# Upload single file
with open("document.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
    job_id = response.json()["job_id"]

# Stream progress
import sseclient

response = httpx.get(
    f"http://localhost:8000/jobs/{job_id}/stream",
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    data = json.loads(event.data)
    print(f"Progress: {data['progress']:.1f}%, Status: {data['status']}")
    
    if data["status"] in ["completed", "failed"]:
        break
```

---

### Q13: Implement streaming responses for an LLM endpoint that generates text token-by-token. Include: (1) Server-Sent Events, (2) Error handling mid-stream, (3) Client disconnection handling, (4) Rate limiting per user.

**Answer:**

```python
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
import asyncio
import json
from datetime import datetime
import redis.asyncio as redis

app = FastAPI()

# Redis for rate limiting
redis_client = None

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url("redis://localhost:6379")

# Rate limiting
RATE_LIMITS = {
    "free": {"requests_per_minute": 5, "tokens_per_day": 10000},
    "pro": {"requests_per_minute": 60, "tokens_per_day": 1000000},
}

async def check_rate_limit(api_key: str, tier: str):
    """Check rate limits for streaming endpoint"""
    minute_key = f"ratelimit:stream:{api_key}:{datetime.now().strftime('%Y-%m-%d:%H:%M')}"
    day_key = f"ratelimit:tokens:{api_key}:{datetime.now().strftime('%Y-%m-%d')}"
    
    limits = RATE_LIMITS[tier]
    
    # Check requests per minute
    minute_count = await redis_client.incr(minute_key)
    if minute_count == 1:
        await redis_client.expire(minute_key, 60)
    
    if minute_count > limits["requests_per_minute"]:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {limits['requests_per_minute']} requests/minute"
        )
    
    # Check tokens per day
    day_count = int(await redis_client.get(day_key) or 0)
    if day_count >= limits["tokens_per_day"]:
        raise HTTPException(
            status_code=429,
            detail=f"Daily token limit exceeded: {limits['tokens_per_day']} tokens/day"
        )

async def increment_token_count(api_key: str, tokens: int):
    """Increment daily token usage"""
    day_key = f"ratelimit:tokens:{api_key}:{datetime.now().strftime('%Y-%m-%d')}"
    await redis_client.incrby(day_key, tokens)
    await redis_client.expire(day_key, 86400)  # 24 hours

async def generate_llm_tokens(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    """
    Mock LLM token generator
    
    Replace with actual LLM API (OpenAI, Anthropic, etc.)
    """
    # Simulate token-by-token generation
    words = ["This", "is", "a", "generated", "response", "from", "the", "LLM", "model", "."]
    
    for i, word in enumerate(words):
        if i >= max_tokens:
            break
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Yield token
        yield word + " "
        
        # Simulate occasional errors
        if i == 5 and temperature > 0.9:
            raise Exception("Model temperature too high, generation unstable")

@app.post("/generate/stream")
async def generate_stream(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    request: Request = None,
    x_api_key: str = Header(...)
):
    """
    Stream LLM generation token-by-token using Server-Sent Events
    
    Handles:
    - Token streaming
    - Mid-stream errors
    - Client disconnection
    - Rate limiting
    """
    # Check rate limit
    user_tier = "free"  # Get from API key lookup
    await check_rate_limit(x_api_key, user_tier)
    
    async def event_generator():
        token_count = 0
        generated_text = ""
        
        try:
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'prompt': prompt})}\n\n"
            
            # Stream tokens
            async for token in generate_llm_tokens(prompt, max_tokens, temperature):
                # Check if client disconnected
                if await request.is_disconnected():
                    print(f"Client disconnected for {x_api_key}")
                    break
                
                token_count += 1
                generated_text += token
                
                # Send token event
                event_data = {
                    "event": "token",
                    "token": token,
                    "token_count": token_count
                }
                yield f"data: {json.dumps(event_data)}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'event': 'done', 'total_tokens': token_count, 'text': generated_text})}\n\n"
            
            # Update token usage
            await increment_token_count(x_api_key, token_count)
        
        except asyncio.CancelledError:
            # Client disconnected or request cancelled
            yield f"data: {json.dumps({'event': 'cancelled'})}\n\n"
            print(f"Generation cancelled for {x_api_key}")
        
        except Exception as e:
            # Error during generation
            error_data = {
                "event": "error",
                "error": str(e),
                "tokens_generated": token_count
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            print(f"Error during generation: {e}")
        
        finally:
            # Always send final usage stats
            day_key = f"ratelimit:tokens:{x_api_key}:{datetime.now().strftime('%Y-%m-%d')}"
            total_tokens = int(await redis_client.get(day_key) or 0)
            
            usage_data = {
                "event": "usage",
                "tokens_this_request": token_count,
                "tokens_today": total_tokens,
                "limit": RATE_LIMITS[user_tier]["tokens_per_day"]
            }
            yield f"data: {json.dumps(usage_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.post("/generate/batch")
async def generate_batch(
    prompts: list[str],
    max_tokens: int = 100,
    x_api_key: str = Header(...)
):
    """
    Non-streaming batch generation
    
    Use when you don't need token-by-token streaming
    """
    user_tier = "free"
    await check_rate_limit(x_api_key, user_tier)
    
    results = []
    total_tokens = 0
    
    for prompt in prompts:
        generated_text = ""
        token_count = 0
        
        async for token in generate_llm_tokens(prompt, max_tokens):
            generated_text += token
            token_count += 1
        
        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "tokens": token_count
        })
        total_tokens += token_count
    
    # Update token usage
    await increment_token_count(x_api_key, total_tokens)
    
    return {
        "results": results,
        "total_tokens": total_tokens
    }
```

**Client implementation:**

```python
import httpx
import json
import asyncio

async def stream_generation():
    """Client that consumes SSE stream"""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/generate/stream",
            json={"prompt": "Write a story about", "max_tokens": 100},
            headers={"X-API-Key": "user_key"},
            timeout=None  # No timeout for streaming
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return
            
            generated_text = ""
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    event = data.get("event")
                    
                    if event == "start":
                        print(f"Starting generation for: {data['prompt']}")
                    
                    elif event == "token":
                        token = data["token"]
                        generated_text += token
                        print(token, end="", flush=True)
                    
                    elif event == "done":
                        print(f"\n\nCompleted! Total tokens: {data['total_tokens']}")
                    
                    elif event == "error":
                        print(f"\nError: {data['error']}")
                        print(f"Tokens before error: {data['tokens_generated']}")
                    
                    elif event == "usage":
                        print(f"\nUsage: {data['tokens_today']}/{data['limit']} tokens today")

# Run client
asyncio.run(stream_generation())
```

**JavaScript client (browser):**

```javascript
async function streamGeneration(prompt) {
    const response = await fetch('http://localhost:8000/generate/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'user_key'
        },
        body: JSON.stringify({ prompt, max_tokens: 100 })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    let generatedText = '';
    
    while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            
            const data = JSON.parse(line.slice(6));
            
            if (data.event === 'token') {
                generatedText += data.token;
                document.getElementById('output').textContent = generatedText;
            }
            else if (data.event === 'done') {
                console.log(`Completed! Total tokens: ${data.total_tokens}`);
            }
            else if (data.event === 'error') {
                console.error(`Error: ${data.error}`);
            }
        }
    }
}

// Usage
streamGeneration("Write a poem about AI");
```

**Key features implemented:**
1. ✅ Server-Sent Events for streaming
2. ✅ Error handling mid-stream (try/except around generator)
3. ✅ Client disconnection detection (`request.is_disconnected()`)
4. ✅ Rate limiting (requests/minute + tokens/day)
5. ✅ Token counting and usage tracking
6. ✅ Graceful cleanup (finally block)

---

### Q14: Implement authentication using API keys and JWT tokens. Include: (1) API key validation, (2) JWT generation and validation, (3) Role-based access control, (4) Token refresh, (5) Logout/revocation.

**Answer:**

```python
from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
import redis.asyncio as redis

app = FastAPI()

# Configuration
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Redis for token blacklist and API key storage
redis_client = None

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await redis.from_url("redis://localhost:6379")

# User database (in production, use actual database)
users_db = {
    "user1": {
        "username": "user1",
        "email": "user1@example.com",
        "hashed_password": pwd_context.hash("password123"),
        "roles": ["user"],
        "api_key": "sk_test_user1_key"
    },
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "roles": ["user", "admin"],
        "api_key": "sk_test_admin_key"
    }
}

# Models
class User(BaseModel):
    username: str
    email: str
    roles: List[str]

class TokenData(BaseModel):
    username: Optional[str] = None
    roles: List[str] = []

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

# Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(username: str) -> str:
    """Create JWT refresh token (longer expiration)"""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": username,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32)  # Unique token ID
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def is_token_blacklisted(jti: str) -> bool:
    """Check if token is blacklisted"""
    return await redis_client.exists(f"blacklist:{jti}")

async def blacklist_token(jti: str, expires_in: int):
    """Add token to blacklist"""
    await redis_client.setex(f"blacklist:{jti}", expires_in, "1")

# Dependency: Validate API key
async def validate_api_key(x_api_key: Optional[str] = Header(None)) -> User:
    """
    Validate API key from header
    
    Usage: @app.get("/endpoint", dependencies=[Depends(validate_api_key)])
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # Check API key in database
    for username, user_data in users_db.items():
        if user_data["api_key"] == x_api_key:
            return User(
                username=username,
                email=user_data["email"],
                roles=user_data["roles"]
            )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key"
    )

# Dependency: Validate JWT token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Validate JWT token from Authorization header
    
    Usage: @app.get("/endpoint", dependencies=[Depends(get_current_user)])
    """
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        username: str = payload.get("sub")
        roles: List[str] = payload.get("roles", [])
        token_type: str = payload.get("type")
        jti: str = payload.get("jti")
        
        if username is None:
            raise credentials_exception
        
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Check if token is blacklisted
        if jti and await is_token_blacklisted(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        
        token_data = TokenData(username=username, roles=roles)
        
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user_data = users_db.get(token_data.username)
    if user_data is None:
        raise credentials_exception
    
    return User(
        username=token_data.username,
        email=user_data["email"],
        roles=token_data.roles
    )

# Dependency: Role-based access control
class RoleChecker:
    """
    Check if user has required roles
    
    Usage: @app.get("/admin", dependencies=[Depends(RoleChecker(["admin"]))])
    """
    def __init__(self, required_roles: List[str]):
        self.required_roles = required_roles
    
    async def __call__(self, user: User = Depends(get_current_user)):
        for role in self.required_roles:
            if role not in user.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required role: {role}"
                )
        return user

# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login with username/password, get JWT tokens
    """
    # Authenticate user
    user_data = users_db.get(request.username)
    if not user_data or not verify_password(request.password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": request.username, "roles": user_data["roles"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    refresh_token = create_refresh_token(request.username)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token"
    )
    
    try:
        # Decode refresh token
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        jti: str = payload.get("jti")
        
        if username is None or token_type != "refresh":
            raise credentials_exception
        
        # Check if refresh token is blacklisted
        if await is_token_blacklisted(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked"
            )
        
    except JWTError:
        raise credentials_exception
    
    # Get user
    user_data = users_db.get(username)
    if not user_data:
        raise credentials_exception
    
    # Create new access token
    access_token = create_access_token(
        data={"sub": username, "roles": user_data["roles"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # Create new refresh token
    new_refresh_token = create_refresh_token(username)
    
    # Blacklist old refresh token
    exp = payload.get("exp")
    if exp:
        expires_in = exp - int(datetime.utcnow().timestamp())
        await blacklist_token(jti, expires_in)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/auth/logout")
async def logout(user: User = Depends(get_current_user), credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout - blacklist access token
    """
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        exp = payload.get("exp")
        
        if jti and exp:
            expires_in = exp - int(datetime.utcnow().timestamp())
            await blacklist_token(jti, expires_in)
    
    except JWTError:
        pass
    
    return {"message": "Successfully logged out"}

# Protected endpoints
@app.get("/me")
async def get_current_user_info(user: User = Depends(get_current_user)):
    """
    Get current user info (requires JWT authentication)
    """
    return user

@app.get("/predict")
async def predict_with_api_key(
    text: str,
    user: User = Depends(validate_api_key)
):
    """
    ML prediction endpoint (requires API key)
    """
    return {
        "prediction": "positive",
        "user": user.username
    }

@app.get("/predict/jwt")
async def predict_with_jwt(
    text: str,
    user: User = Depends(get_current_user)
):
    """
    ML prediction endpoint (requires JWT token)
    """
    return {
        "prediction": "positive",
        "user": user.username
    }

@app.get("/admin/users")
async def list_users(user: User = Depends(RoleChecker(["admin"]))):
    """
    Admin endpoint - list all users (requires admin role)
    """
    return {
        "users": [
            User(username=username, email=data["email"], roles=data["roles"])
            for username, data in users_db.items()
        ]
    }

@app.post("/admin/users/{username}/api-key")
async def generate_api_key(
    username: str,
    user: User = Depends(RoleChecker(["admin"]))
):
    """
    Generate new API key for user (admin only)
    """
    if username not in users_db:
        raise HTTPException(404, "User not found")
    
    # Generate new API key
    new_api_key = f"sk_live_{secrets.token_urlsafe(32)}"
    users_db[username]["api_key"] = new_api_key
    
    return {
        "username": username,
        "api_key": new_api_key
    }
```

**Client usage examples:**

```python
import httpx

# 1. Login to get tokens
login_response = httpx.post(
    "http://localhost:8000/auth/login",
    json={"username": "user1", "password": "password123"}
)
tokens = login_response.json()
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]

# 2. Use JWT token for authenticated requests
response = httpx.get(
    "http://localhost:8000/predict/jwt?text=hello",
    headers={"Authorization": f"Bearer {access_token}"}
)

# 3. Refresh token when expired
refresh_response = httpx.post(
    "http://localhost:8000/auth/refresh",
    json={"refresh_token": refresh_token}
)
new_tokens = refresh_response.json()

# 4. Use API key (simpler for API clients)
response = httpx.get(
    "http://localhost:8000/predict?text=hello",
    headers={"X-API-Key": "sk_test_user1_key"}
)

# 5. Logout (blacklist token)
httpx.post(
    "http://localhost:8000/auth/logout",
    headers={"Authorization": f"Bearer {access_token}"}
)
```

**Summary:**
- ✅ API key validation (simple, long-lived)
- ✅ JWT token generation and validation (secure, short-lived)
- ✅ Role-based access control (admin vs user)
- ✅ Token refresh (get new access token without re-login)
- ✅ Token revocation/blacklist (logout, security)

---

### Q15: Implement error handling with custom exception classes, detailed error responses, and automatic error logging. Include retry logic for transient failures.

**Answer:**

```python
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import traceback
import sys
from functools import wraps
import asyncio

app = FastAPI()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Custom exception classes
class BaseAPIException(Exception):
    """Base exception for all API errors"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ModelNotFoundException(BaseAPIException):
    """Model not found"""
    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model '{model_id}' not found",
            status_code=404,
            error_code="MODEL_NOT_FOUND",
            details={"model_id": model_id}
        )

class ModelInferenceException(BaseAPIException):
    """Error during model inference"""
    def __init__(self, message: str, model_id: str):
        super().__init__(
            message=f"Inference failed for model '{model_id}': {message}",
            status_code=500,
            error_code="INFERENCE_ERROR",
            details={"model_id": model_id}
        )

class RateLimitException(BaseAPIException):
    """Rate limit exceeded"""
    def __init__(self, limit: int, window: str):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window": window}
        )

class InvalidInputException(BaseAPIException):
    """Invalid input data"""
    def __init__(self, message: str, field: str = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="INVALID_INPUT",
            details={"field": field} if field else {}
        )

class ServiceUnavailableException(BaseAPIException):
    """External service unavailable"""
    def __init__(self, service: str):
        super().__init__(
            message=f"Service '{service}' is currently unavailable",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            details={"service": service}
        )

# Error response model
class ErrorResponse(BaseModel):
    error: str
    error_code: str
    message: str
    details: Dict[str, Any] = {}
    timestamp: str
    path: str
    request_id: Optional[str] = None
    trace_id: Optional[str] = None

# Exception handlers
@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions"""
    error_response = ErrorResponse(
        error=exc.__class__.__name__,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        timestamp=datetime.utcnow().isoformat(),
        path=request.url.path,
        request_id=request.headers.get("X-Request-ID")
    )
    
    # Log error
    logger.error(
        f"API Error: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": request.url.path,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    error_response = ErrorResponse(
        error="ValidationError",
        error_code="VALIDATION_ERROR",
        message="Input validation failed",
        details={"errors": errors},
        timestamp=datetime.utcnow().isoformat(),
        path=request.url.path
    )
    
    logger.warning(
        f"Validation error: {request.url.path}",
        extra={"errors": errors}
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    error_response = ErrorResponse(
        error=exc.__class__.__name__,
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred",
        details={"exception": str(exc)},
        timestamp=datetime.utcnow().isoformat(),
        path=request.url.path
    )
    
    # Log with full traceback
    logger.exception(
        f"Unexpected error: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "traceback": traceback.format_exc()
        }
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )

# Retry decorator for transient failures
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator for handling transient failures
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                
                except (ServiceUnavailableException, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        raise
                
                except Exception as e:
                    # Don't retry on non-transient errors
                    logger.error(f"Non-retriable error: {str(e)}")
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses"""
    start_time = datetime.utcnow()
    
    # Generate request ID if not provided
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "request_id": request_id,
            "client": request.client.host if request.client else None
        }
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Log response
        logger.info(
            f"Response: {response.status_code} - {duration:.3f}s",
            extra={
                "status_code": response.status_code,
                "duration_seconds": duration,
                "request_id": request_id
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.error(
            f"Request failed: {str(e)} - {duration:.3f}s",
            exc_info=True,
            extra={
                "request_id": request_id,
                "duration_seconds": duration
            }
        )
        raise

# Example endpoints with error handling
@app.post("/predict")
@retry_on_failure(max_retries=3, delay=0.5, backoff=2.0)
async def predict(text: str, model_id: str = "default"):
    """
    Prediction endpoint with comprehensive error handling
    """
    # Validate input
    if len(text) < 5:
        raise InvalidInputException(
            "Text too short (minimum 5 characters)",
            field="text"
        )
    
    # Check model exists
    available_models = ["default", "v1", "v2"]
    if model_id not in available_models:
        raise ModelNotFoundException(model_id)
    
    # Simulate external service call (may fail transiently)
    try:
        # Your model inference here
        result = await call_external_service(text, model_id)
        return result
    
    except ConnectionError:
        raise ServiceUnavailableException("model-service")
    
    except Exception as e:
        raise ModelInferenceException(str(e), model_id)

async def call_external_service(text: str, model_id: str) -> dict:
    """Simulate external service that may fail"""
    import random
    
    # Simulate transient failure 30% of the time
    if random.random() < 0.3:
        raise ConnectionError("Temporary network issue")
    
    # Simulate processing
    await asyncio.sleep(0.1)
    
    return {
        "text": text,
        "model_id": model_id,
        "prediction": "positive",
        "confidence": 0.95
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
```

**Example error responses:**

```python
# 1. Validation error (422)
POST /predict
{"text": "hi", "model_id": "default"}

Response:
{
  "error": "ValidationError",
  "error_code": "VALIDATION_ERROR",
  "message": "Input validation failed",
  "details": {
    "errors": [{
      "field": "text",
      "message": "Text too short (minimum 5 characters)",
      "type": "value_error"
    }]
  },
  "timestamp": "2024-02-04T10:30:00Z",
  "path": "/predict",
  "request_id": "abc-123"
}

# 2. Model not found (404)
POST /predict
{"text": "hello world", "model_id": "unknown"}

Response:
{
  "error": "ModelNotFoundException",
  "error_code": "MODEL_NOT_FOUND",
  "message": "Model 'unknown' not found",
  "details": {"model_id": "unknown"},
  "timestamp": "2024-02-04T10:30:00Z",
  "path": "/predict"
}

# 3. Service unavailable (503) - after retries
Response:
{
  "error": "ServiceUnavailableException",
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "Service 'model-service' is currently unavailable",
  "details": {"service": "model-service"},
  "timestamp": "2024-02-04T10:30:00Z",
  "path": "/predict"
}
```

**Summary:**
- ✅ Custom exception hierarchy
- ✅ Structured error responses
- ✅ Automatic error logging
- ✅ Request/response middleware
- ✅ Retry logic for transient failures
- ✅ Request ID tracking

---

### Q16: Implement CORS configuration for a production ML API that needs to be called from multiple web domains. Include: (1) Origin validation, (2) Credential handling, (3) Preflight caching, (4) Security headers.

**Answer:**

```python
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import re

app = FastAPI()

# CORS configuration
ALLOWED_ORIGINS = [
    "https://app.example.com",
    "https://dashboard.example.com",
    "https://mobile.example.com"
]

# Allow localhost for development
ALLOWED_ORIGINS_REGEX = [
    r"https://.*\.example\.com",  # Any subdomain of example.com
    r"http://localhost:\d+",  # Any localhost port
    r"http://127\.0\.0\.1:\d+"  # Any 127.0.0.1 port
]

def is_origin_allowed(origin: str) -> bool:
    """Check if origin is allowed (exact match or regex)"""
    # Exact match
    if origin in ALLOWED_ORIGINS:
        return True
    
    # Regex match
    for pattern in ALLOWED_ORIGINS_REGEX:
        if re.match(pattern, origin):
            return True
    
    return False

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # List of allowed origins
    allow_origin_regex=r"https://.*\.example\.com|http://localhost:\d+",  # Regex pattern
    allow_credentials=True,  # Allow cookies and auth headers
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allowed HTTP methods
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # Allowed headers
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],  # Headers to expose to client
    max_age=3600,  # Cache preflight response for 1 hour
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    # Remove server header (hide FastAPI version)
    if "server" in response.headers:
        del response.headers["server"]
    
    return response

# Custom CORS handling for dynamic validation
@app.middleware("http")
async def custom_cors_handler(request: Request, call_next):
    """
    Custom CORS middleware for advanced origin validation
    
    Use when you need dynamic origin validation (e.g., check database)
    """
    origin = request.headers.get("origin")
    
    if origin and request.method == "OPTIONS":
        # Preflight request
        if is_origin_allowed(origin):
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
                    "Access-Control-Max-Age": "3600"
                }
            )
        else:
            return JSONResponse(
                status_code=403,
                content={"detail": f"Origin '{origin}' not allowed"}
            )
    
    # Actual request
    response = await call_next(request)
    
    if origin and is_origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"  # Important for caching
    
    return response

# Endpoints
@app.get("/predict")
async def predict(
    text: str,
    origin: str = Header(None)
):
    """
    ML prediction endpoint
    
    CORS headers automatically added by middleware
    """
    return {
        "prediction": "positive",
        "confidence": 0.95,
        "origin_allowed": is_origin_allowed(origin) if origin else None
    }

@app.post("/predict/batch")
async def predict_batch(texts: List[str]):
    """Batch prediction with CORS support"""
    return {
        "predictions": [
            {"text": text, "prediction": "positive"}
            for text in texts
        ]
    }

# Admin endpoint - no CORS (backend-only)
@app.get("/admin/stats")
async def admin_stats(request: Request):
    """
    Admin endpoint - reject if called from browser
    """
    origin = request.headers.get("origin")
    
    if origin:
        # Reject if called from browser (has Origin header)
        raise HTTPException(
            status_code=403,
            detail="This endpoint cannot be accessed from browsers"
        )
    
    return {"users": 100, "requests": 1000}
```

**Frontend usage (JavaScript):**

```javascript
// Simple request (no preflight)
fetch('https://api.example.com/predict?text=hello', {
    method: 'GET',
    credentials: 'include',  // Include cookies
    headers: {
        'X-API-Key': 'your-api-key'
    }
})
.then(r => r.json())
.then(data => console.log(data));

// Complex request (triggers preflight)
fetch('https://api.example.com/predict/batch', {
    method: 'POST',
    credentials: 'include',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer token'
    },
    body: JSON.stringify({
        texts: ['hello', 'world']
    })
})
.then(r => r.json())
.then(data => console.log(data));
```

**Preflight request flow:**

```
# Step 1: Browser sends OPTIONS preflight
OPTIONS /predict/batch HTTP/1.1
Origin: https://app.example.com
Access-Control-Request-Method: POST
Access-Control-Request-Headers: Content-Type, Authorization

# Step 2: Server responds with allowed settings
HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Methods: POST
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 3600
Access-Control-Allow-Credentials: true

# Step 3: Browser sends actual request
POST /predict/batch HTTP/1.1
Origin: https://app.example.com
Content-Type: application/json
Authorization: Bearer token

# Step 4: Server responds with data + CORS headers
HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Credentials: true
```

**Security best practices:**

```python
# Production CORS configuration
CORS_CONFIG = {
    # ✅ Whitelist specific origins
    "allow_origins": [
        "https://app.production.com",
        "https://mobile.production.com"
    ],
    
    # ❌ NEVER use "*" with credentials
    # "allow_origins": ["*"],  # DON'T DO THIS
    # "allow_credentials": True,  # CAN'T USE BOTH
    
    # ✅ Only allow necessary methods
    "allow_methods": ["GET", "POST"],  # Not DELETE, PUT
    
    # ✅ Only allow necessary headers
    "allow_headers": ["Content-Type", "Authorization"],
    
    # ✅ Expose only needed headers
    "expose_headers": ["X-Request-ID"],
    
    # ✅ Cache preflight for performance
    "max_age": 3600,
    
    # ✅ Use credentials only when needed
    "allow_credentials": True
}
```

**Testing CORS:**

```bash
# Test preflight
curl -X OPTIONS \
  -H "Origin: https://app.example.com" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  http://localhost:8000/predict/batch

# Expected response headers:
# Access-Control-Allow-Origin: https://app.example.com
# Access-Control-Allow-Methods: POST
# Access-Control-Max-Age: 3600
```

---

### Q17: Implement background tasks for email notifications after model predictions complete. Include task queuing, status tracking, and failure handling with Celery.

**Answer:**

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, EmailStr
from celery import Celery
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import redis

app = FastAPI()

# Celery configuration
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=240,  # 4 minutes soft limit
    task_acks_late=True,  # Acknowledge after task completes
    worker_prefetch_multiplier=1,  # Take one task at a time
    task_reject_on_worker_lost=True
)

# Redis client for status tracking
redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)

# Models
class PredictionRequest(BaseModel):
    text: str
    email: EmailStr
    notify_on_complete: bool = True

class TaskStatus(BaseModel):
    task_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE
    result: Optional[dict] = None
    error: Optional[str] = None

# Celery tasks
@celery_app.task(bind=True, max_retries=3)
def send_email_task(self, to_email: str, subject: str, body: str):
    """
    Celery task to send email
    
    Retries automatically on failure (max 3 times)
    """
    try:
        # SMTP configuration (use environment variables in production)
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        SMTP_USER = "your-email@gmail.com"
        SMTP_PASSWORD = "your-app-password"
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        return {"status": "sent", "to": to_email}
    
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

@celery_app.task(bind=True)
def process_prediction_with_notification(self, text: str, email: str):
    """
    Celery task for prediction + email notification
    
    Long-running task that sends email when complete
    """
    try:
        # Update status
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        # Run model inference (simulated)
        import time
        time.sleep(2)  # Simulate processing
        
        prediction = {
            "text": text,
            "label": "positive",
            "confidence": 0.95
        }
        
        self.update_state(state='PROCESSING', meta={'progress': 80})
        
        # Send email notification
        subject = "ML Prediction Complete"
        body = f"""
        <html>
          <body>
            <h2>Your prediction is ready!</h2>
            <p><strong>Text:</strong> {text}</p>
            <p><strong>Prediction:</strong> {prediction['label']}</p>
            <p><strong>Confidence:</strong> {prediction['confidence']:.2%}</p>
          </body>
        </html>
        """
        
        send_email_task.delay(email, subject, body)
        
        self.update_state(state='PROCESSING', meta={'progress': 100})
        
        return prediction
    
    except Exception as e:
        return {"error": str(e)}

# FastAPI endpoints
@app.post("/predict/async")
async def predict_async(request: PredictionRequest):
    """
    Async prediction with Celery
    
    Returns task_id immediately, processes in background
    """
    # Submit task to Celery
    task = process_prediction_with_notification.delay(
        request.text,
        request.email
    )
    
    return {
        "task_id": task.id,
        "status": "submitted",
        "status_url": f"/tasks/{task.id}"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get Celery task status"""
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0
        }
    elif task.state == 'PROCESSING':
        response = {
            "task_id": task_id,
            "status": "processing",
            "progress": task.info.get('progress', 0)
        }
    elif task.state == 'SUCCESS':
        response = {
            "task_id": task_id,
            "status": "success",
            "result": task.result,
            "progress": 100
        }
    elif task.state == 'FAILURE':
        response = {
            "task_id": task_id,
            "status": "failed",
            "error": str(task.info),
            "progress": 0
        }
    else:
        response = {
            "task_id": task_id,
            "status": task.state.lower()
        }
    
    return response

@app.post("/predict/simple")
async def predict_simple(
    text: str,
    email: EmailStr,
    background_tasks: BackgroundTasks
):
    """
    Simple prediction with FastAPI BackgroundTasks
    
    Use for lightweight tasks that don't need persistence
    """
    # Run prediction immediately
    prediction = {
        "text": text,
        "label": "positive",
        "confidence": 0.95
    }
    
    # Send email in background (after response)
    background_tasks.add_task(
        send_email_notification,
        email,
        prediction
    )
    
    return prediction

def send_email_notification(email: str, prediction: dict):
    """
    Simple background task (not persistent)
    
    If server restarts, task is lost
    """
    subject = "Prediction Complete"
    body = f"Prediction: {prediction['label']} ({prediction['confidence']:.2%})"
    
    # Send email (simplified)
    print(f"Sending email to {email}: {body}")
```

**Running Celery worker:**

```bash
# Start Celery worker
celery -A main.celery_app worker --loglevel=info

# Start Celery with multiple workers
celery -A main.celery_app worker --loglevel=info --concurrency=4

# Monitor tasks (Flower)
celery -A main.celery_app flower
# Open http://localhost:5555
```

**Client usage:**

```python
import httpx
import time

# Submit async task
response = httpx.post(
    "http://localhost:8000/predict/async",
    json={
        "text": "Great product!",
        "email": "user@example.com"
    }
)
task_id = response.json()["task_id"]

# Poll for status
while True:
    status = httpx.get(f"http://localhost:8000/tasks/{task_id}").json()
    print(f"Status: {status['status']}, Progress: {status.get('progress', 0)}%")
    
    if status["status"] in ["success", "failed"]:
        break
    
    time.sleep(1)
```

**Comparison: FastAPI BackgroundTasks vs Celery:**

| Feature | BackgroundTasks | Celery |
|---------|----------------|--------|
| Persistence | ❌ Lost on restart | ✅ Persisted in broker |
| Retries | ❌ Manual | ✅ Automatic |
| Monitoring | ❌ No UI | ✅ Flower UI |
| Distributed | ❌ Same process | ✅ Multiple workers |
| Scheduling | ❌ No | ✅ Celery Beat |
| Task priority | ❌ No | ✅ Yes |
| Best for | Lightweight tasks | Production, long-running |

---

### Q18: Implement request/response logging with sensitive data masking (PII, API keys). Include structured logging and integration with logging systems (ELK, DataDog).

**Answer:**

```python
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import logging
import json
import re
from datetime import datetime
from typing import Any, Dict
import hashlib

app = FastAPI()

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # We'll format as JSON
)
logger = logging.getLogger(__name__)

# Sensitive field patterns
SENSITIVE_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "api_key": r'\b(sk|pk)_[a-zA-Z0-9]{20,}\b',
    "password": r'"password"\s*:\s*"[^"]*"'
}

SENSITIVE_FIELDS = [
    "password", "api_key", "token", "secret", "authorization",
    "credit_card", "ssn", "social_security"
]

def mask_sensitive_data(data: Any) -> Any:
    """
    Recursively mask sensitive data in requests/responses
    """
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            # Check if field name is sensitive
            if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
                if isinstance(value, str) and len(value) > 4:
                    # Show first 4 chars, mask rest
                    masked[key] = value[:4] + "***MASKED***"
                else:
                    masked[key] = "***MASKED***"
            else:
                masked[key] = mask_sensitive_data(value)
        return masked
    
    elif isinstance(data, list):
        return [mask_sensitive_data(item) for item in data]
    
    elif isinstance(data, str):
        # Mask patterns in strings
        masked_str = data
        for pattern_name, pattern in SENSITIVE_PATTERNS.items():
            masked_str = re.sub(
                pattern,
                f"***{pattern_name.upper()}_MASKED***",
                masked_str
            )
        return masked_str
    
    return data

def hash_pii(value: str) -> str:
    """Hash PII for consistent anonymization"""
    return hashlib.sha256(value.encode()).hexdigest()[:16]

class StructuredLogger:
    """Structured JSON logger for ELK/DataDog"""
    
    @staticmethod
    def log_request(
        request: Request,
        request_body: dict,
        request_id: str
    ):
        """Log incoming request"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "body": mask_sensitive_data(request_body),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent")
        }
        
        # Remove sensitive headers
        if "authorization" in log_data["headers"]:
            log_data["headers"]["authorization"] = "***MASKED***"
        if "x-api-key" in log_data["headers"]:
            log_data["headers"]["x-api-key"] = "***MASKED***"
        
        logger.info(json.dumps(log_data))
    
    @staticmethod
    def log_response(
        request: Request,
        response_body: dict,
        status_code: int,
        duration_ms: float,
        request_id: str
    ):
        """Log outgoing response"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "response",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "body": mask_sensitive_data(response_body)
        }
        
        logger.info(json.dumps(log_data))
    
    @staticmethod
    def log_error(
        request: Request,
        error: Exception,
        request_id: str
    ):
        """Log errors"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        logger.error(json.dumps(log_data))

# Middleware for logging
@app.middleware("http")
async def log_requests_responses(request: Request, call_next):
    """Log all requests and responses with masking"""
    import time
    import uuid
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Read request body
    body = await request.body()
    request_body = {}
    if body:
        try:
            request_body = json.loads(body)
        except:
            request_body = {"raw": body.decode()}
    
    # Log request
    start_time = time.time()
    StructuredLogger.log_request(request, request_body, request_id)
    
    # Process request
    try:
        # Create new request with body (FastAPI consumes it)
        async def receive():
            return {"type": "http.request", "body": body}
        
        request._receive = receive
        
        # Get response
        response = await call_next(request)
        
        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Parse response
        try:
            response_data = json.loads(response_body)
        except:
            response_data = {"raw": response_body.decode()}
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        StructuredLogger.log_response(
            request,
            response_data,
            response.status_code,
            duration_ms,
            request_id
        )
        
        # Return response with body
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
    
    except Exception as e:
        # Log error
        StructuredLogger.log_error(request, e, request_id)
        raise

# Example endpoint
@app.post("/user/register")
async def register_user(
    email: str,
    password: str,
    phone: str,
    credit_card: str
):
    """
    Endpoint with sensitive data
    
    Logs will have masked PII
    """
    # Process registration
    return {
        "message": "User registered",
        "email": email,
        "user_id": hash_pii(email)  # Anonymized ID
    }
```

**Logged output (masked):**

```json
{
  "timestamp": "2024-02-04T10:30:00Z",
  "type": "request",
  "request_id": "abc-123",
  "method": "POST",
  "path": "/user/register",
  "body": {
    "email": "***EMAIL_MASKED***",
    "password": "***MASKED***",
    "phone": "***PHONE_MASKED***",
    "credit_card": "***CREDIT_CARD_MASKED***"
  },
  "client_ip": "192.168.1.1"
}

{
  "timestamp": "2024-02-04T10:30:01Z",
  "type": "response",
  "request_id": "abc-123",
  "status_code": 200,
  "duration_ms": 125.5,
  "body": {
    "message": "User registered",
    "email": "***EMAIL_MASKED***",
    "user_id": "a1b2c3d4e5f6"
  }
}
```

**Integration with ELK Stack:**

```python
# Use python-logstash for ELK
import logstash

logger.addHandler(logstash.TCPLogstashHandler(
    host='localhost',
    port=5000,
    version=1
))
```

**Integration with DataDog:**

```python
# Use ddtrace for DataDog
from ddtrace import tracer, patch_all

patch_all()

@app.middleware("http")
async def datadog_middleware(request: Request, call_next):
    with tracer.trace("http.request", service="ml-api"):
        response = await call_next(request)
        return response
```

---

### Q19: Implement health check and readiness probe endpoints for Kubernetes. Include: (1) Database connectivity, (2) Model availability, (3) Dependency checks, (4) Graceful degradation.

**Answer:**

```python
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import httpx
from sqlalchemy import create_engine, text
import torch

app = FastAPI()

# Health status models
class ComponentHealth(BaseModel):
    name: str
    status: str  # healthy, degraded, unhealthy
    message: Optional[str] = None
    response_time_ms: Optional[float] = None
    last_checked: str

class HealthResponse(BaseModel):
    status: str  # healthy, degraded, unhealthy
    version: str
    uptime_seconds: float
    checks: Dict[str, ComponentHealth]

# Track startup time
startup_time = datetime.utcnow()

# Configuration
DB_URL = "postgresql://user:pass@localhost/db"
MODEL_PATH = "models/sentiment.pt"
EXTERNAL_API = "https://api.external.com/health"

async def check_database() -> ComponentHealth:
    """Check database connectivity"""
    start_time = datetime.utcnow()
    
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            
            if row[0] == 1:
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                return ComponentHealth(
                    name="database",
                    status="healthy",
                    message="Connected",
                    response_time_ms=duration,
                    last_checked=datetime.utcnow().isoformat()
                )
    
    except Exception as e:
        return ComponentHealth(
            name="database",
            status="unhealthy",
            message=f"Connection failed: {str(e)}",
            last_checked=datetime.utcnow().isoformat()
        )

async def check_model() -> ComponentHealth:
    """Check if model is loaded and functional"""
    start_time = datetime.utcnow()
    
    try:
        # Check if model file exists
        import os
        if not os.path.exists(MODEL_PATH):
            return ComponentHealth(
                name="model",
                status="unhealthy",
                message="Model file not found",
                last_checked=datetime.utcnow().isoformat()
            )
        
        # Try loading model (cached)
        model = torch.load(MODEL_PATH, map_location='cpu')
        
        # Run test inference
        test_input = torch.randn(1, 128)
        with torch.no_grad():
            output = model(test_input)
        
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ComponentHealth(
            name="model",
            status="healthy",
            message="Model operational",
            response_time_ms=duration,
            last_checked=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        return ComponentHealth(
            name="model",
            status="unhealthy",
            message=f"Model check failed: {str(e)}",
            last_checked=datetime.utcnow().isoformat()
        )

async def check_external_api() -> ComponentHealth:
    """Check external API dependency"""
    start_time = datetime.utcnow()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(EXTERNAL_API, timeout=5.0)
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ComponentHealth(
                    name="external_api",
                    status="healthy",
                    message="API reachable",
                    response_time_ms=duration,
                    last_checked=datetime.utcnow().isoformat()
                )
            else:
                return ComponentHealth(
                    name="external_api",
                    status="degraded",
                    message=f"API returned {response.status_code}",
                    response_time_ms=duration,
                    last_checked=datetime.utcnow().isoformat()
                )
    
    except Exception as e:
        return ComponentHealth(
            name="external_api",
            status="degraded",  # Degraded not unhealthy (can continue without it)
            message=f"API unreachable: {str(e)}",
            last_checked=datetime.utcnow().isoformat()
        )

async def check_memory() -> ComponentHealth:
    """Check memory usage"""
    import psutil
    
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    if memory_percent < 80:
        status = "healthy"
        message = f"Memory usage: {memory_percent:.1f}%"
    elif memory_percent < 90:
        status = "degraded"
        message = f"High memory usage: {memory_percent:.1f}%"
    else:
        status = "unhealthy"
        message = f"Critical memory usage: {memory_percent:.1f}%"
    
    return ComponentHealth(
        name="memory",
        status=status,
        message=message,
        last_checked=datetime.utcnow().isoformat()
    )

async def check_disk() -> ComponentHealth:
    """Check disk space"""
    import psutil
    
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    
    if disk_percent < 80:
        status = "healthy"
    elif disk_percent < 90:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return ComponentHealth(
        name="disk",
        status=status,
        message=f"Disk usage: {disk_percent:.1f}%",
        last_checked=datetime.utcnow().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring
    
    Returns overall health status:
    - healthy: All checks pass
    - degraded: Some non-critical checks fail
    - unhealthy: Critical checks fail
    """
    # Run all checks in parallel
    checks_results = await asyncio.gather(
        check_database(),
        check_model(),
        check_external_api(),
        check_memory(),
        check_disk(),
        return_exceptions=True
    )
    
    # Build checks dict
    checks = {}
    for result in checks_results:
        if isinstance(result, ComponentHealth):
            checks[result.name] = result
        else:
            # Handle exceptions
            checks["unknown"] = ComponentHealth(
                name="unknown",
                status="unhealthy",
                message=str(result),
                last_checked=datetime.utcnow().isoformat()
            )
    
    # Determine overall status
    statuses = [check.status for check in checks.values()]
    
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
        http_status = status.HTTP_200_OK
    elif any(s == "unhealthy" for s in statuses):
        # Critical components unhealthy
        critical_components = ["database", "model"]
        if any(checks[c].status == "unhealthy" for c in critical_components if c in checks):
            overall_status = "unhealthy"
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            overall_status = "degraded"
            http_status = status.HTTP_200_OK
    else:
        overall_status = "degraded"
        http_status = status.HTTP_200_OK
    
    # Calculate uptime
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    
    response = HealthResponse(
        status=overall_status,
        version="1.0.0",
        uptime_seconds=uptime,
        checks=checks
    )
    
    return JSONResponse(
        status_code=http_status,
        content=response.dict()
    )

@app.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe
    
    Simple check: is the service running?
    Returns 200 if alive, otherwise container is restarted
    """
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe
    
    Check if service is ready to accept traffic
    Returns 200 if ready, 503 if not (removed from load balancer)
    """
    # Check critical components only
    db_health = await check_database()
    model_health = await check_model()
    
    if db_health.status == "healthy" and model_health.status == "healthy":
        return {
            "status": "ready",
            "checks": {
                "database": db_health.dict(),
                "model": model_health.dict()
            }
        }
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "checks": {
                    "database": db_health.dict(),
                    "model": model_health.dict()
                }
            }
        )

@app.get("/health/startup")
async def startup_probe():
    """
    Kubernetes startup probe
    
    Check if application has started successfully
    Used for slow-starting apps (model loading)
    """
    # Check if model is loaded
    model_health = await check_model()
    
    if model_health.status == "healthy":
        return {"status": "started"}
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "starting",
                "message": model_health.message
            }
        )
```

**Kubernetes deployment with probes:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: ml-api:latest
        ports:
        - containerPort: 8000
        
        # Startup probe - checks if app started (model loaded)
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 30  # 30 * 5 = 150s max startup time
        
        # Liveness probe - checks if app is alive
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3  # Restart after 3 failures
        
        # Readiness probe - checks if ready for traffic
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          failureThreshold: 3  # Remove from LB after 3 failures
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

---

### Q20: Implement API documentation with custom examples, request/response schemas, and authentication docs. Use FastAPI's automatic OpenAPI generation effectively.

**Answer:**

```python
from fastapi import FastAPI, Header, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Custom OpenAPI configuration
app = FastAPI(
    title="ML Prediction API",
    description="""
    ## Machine Learning Prediction API
    
    This API provides sentiment analysis and text classification services.
    
    ### Authentication
    
    Use API key in header:
    ```
    X-API-Key: your-api-key-here
    ```
    
    ### Rate Limits
    
    - Free tier: 100 requests/day
    - Pro tier: 10,000 requests/day
    
    ### Support
    
    Contact: support@example.com
    """,
    version="1.0.0",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "support@example.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    },
    openapi_tags=[
        {
            "name": "predictions",
            "description": "Endpoints for ML predictions"
        },
        {
            "name": "admin",
            "description": "Admin operations (requires admin role)"
        }
    ]
)

# Models with examples
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text to analyze",
        min_length=5,
        max_length=5000,
        example="This product is amazing! I love it."
    )
    model_version: str = Field(
        default="v1",
        description="Model version to use",
        regex="^v[0-9]+$",
        example="v2"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Include probability scores for all classes",
        example=True
    )
    
    class Config:
        schema_extra = {
            "examples": {
                "positive": {
                    "summary": "Positive review",
                    "value": {
                        "text": "Excellent service! Highly recommended.",
                        "model_version": "v2",
                        "return_probabilities": True
                    }
                },
                "negative": {
                    "summary": "Negative feedback",
                    "value": {
                        "text": "Very disappointed with the quality.",
                        "model_version": "v1",
                        "return_probabilities": False
                    }
                }
            }
        }

class PredictResponse(BaseModel):
    label: Sentiment = Field(..., description="Predicted sentiment", example="positive")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1, example=0.95)
    probabilities: Optional[dict] = Field(
        None,
        description="Probability distribution across all classes",
        example={"positive": 0.95, "negative": 0.03, "neutral": 0.02}
    )
    model_version: str = Field(..., example="v2")
    
    class Config:
        schema_extra = {
            "example": {
                "label": "positive",
                "confidence": 0.95,
                "probabilities": {
                    "positive": 0.95,
                    "negative": 0.03,
                    "neutral": 0.02
                },
                "model_version": "v2"
            }
        }

class BatchRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_items=1,
        max_items=100,
        example=[
            "Great product!",
            "Not satisfied.",
            "It's okay."
        ]
    )

# Documented endpoints
@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["predictions"],
    summary="Predict sentiment for single text",
    description="""
    Analyze sentiment of a single text input.
    
    **Supported languages:** English, Spanish, French
    
    **Response time:** < 100ms (p95)
    
    **Example:**
    ```bash
    curl -X POST "https://api.example.com/predict" \\
      -H "X-API-Key: your-key" \\
      -H "Content-Type: application/json" \\
      -d '{"text": "Great product!", "model_version": "v2"}'
    ```
    """,
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "label": "positive",
                        "confidence": 0.95,
                        "probabilities": {"positive": 0.95, "negative": 0.03, "neutral": 0.02},
                        "model_version": "v2"
                    }
                }
            }
        },
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Text too short (minimum 5 characters)"
                    }
                }
            }
        },
        401: {
            "description": "Missing or invalid API key",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid API key"
                    }
                }
            }
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded: 100 requests/day"
                    }
                }
            }
        }
    }
)
async def predict(
    request: PredictRequest,
    x_api_key: str = Header(..., description="API key for authentication")
):
    """Predict sentiment for single text"""
    # Your prediction logic here
    return PredictResponse(
        label=Sentiment.POSITIVE,
        confidence=0.95,
        probabilities={"positive": 0.95, "negative": 0.03, "neutral": 0.02} if request.return_probabilities else None,
        model_version=request.model_version
    )

@app.post(
    "/predict/batch",
    tags=["predictions"],
    summary="Batch prediction",
    description="Analyze sentiment for multiple texts in one request"
)
async def predict_batch(
    request: BatchRequest,
    x_api_key: str = Header(..., description="API key")
):
    """Batch prediction endpoint"""
    results = [
        PredictResponse(
            label=Sentiment.POSITIVE,
            confidence=0.95,
            model_version="v2"
        )
        for text in request.texts
    ]
    return {"predictions": results}

# Access documentation at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - OpenAPI JSON: http://localhost:8000/openapi.json
```

**Custom OpenAPI schema:**

```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="ML API",
        version="1.0.0",
        description="Custom ML API",
        routes=app.routes,
    )
    
    # Add custom security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Add servers
    openapi_schema["servers"] = [
        {"url": "https://api.production.com", "description": "Production"},
        {"url": "https://api.staging.com", "description": "Staging"},
        {"url": "http://localhost:8000", "description": "Development"}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

**Summary:** The file is complete with comprehensive FastAPI questions covering architecture, implementation, and documentation.

---

