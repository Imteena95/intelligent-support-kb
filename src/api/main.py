from fastapi import FastAPI, HTTPException
import uvicorn
from dotenv import load_dotenv
import os
import logging
from src.core.rag_pipeline import get_rag_pipeline

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FREE Support KB RAG System",
    description="Zero-cost intelligent support knowledge base",
    version="1.0.0"
)

# Initialize RAG pipeline
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    try:
        rag_pipeline = get_rag_pipeline()
        logger.info("✅ RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG pipeline: {e}")

@app.get("/")
async def root():
    return {
        "message": "FREE Intelligent Support KB RAG System",
        "status": "running",
        "mode": "free-only",
        "llm": os.getenv("LOCAL_MODEL", "llama2:7b"),
        "endpoints": ["/query", "/ingest", "/stats", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/stats")
async def stats():
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline.get_stats()

@app.post("/ingest")
async def ingest_documents(documents: list[str], metadata: list[dict] = None):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    result = rag_pipeline.ingest_documents(documents, metadata)
    return result

@app.post("/query")
async def query(question: str, n_results: int = 3):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    result = rag_pipeline.query(question, n_results)
    return result

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8000))
    print(f"🚀 Starting FREE RAG system on http://localhost:{port}")
    print(f"📚 API docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)