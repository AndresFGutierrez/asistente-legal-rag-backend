from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
import os
from rag_pipeline import RAGPipeline
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag = RAGPipeline()

DATASET_PATH = "../dataset"
available_pdfs = glob.glob(os.path.join(DATASET_PATH, "*.pdf"))
logger.info(f"📂 PDFs: {[os.path.basename(p) for p in available_pdfs]}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando...")
    try:
        rag.process_documents(available_pdfs)
        logger.info(f"✅ {len(rag.documents_loaded)} documentos listos")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise
    yield

app = FastAPI(title="Asistente Legal RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    documents_consulted: List[str]

@app.get("/")
async def root():
    return {
        "message": "Asistente Legal RAG",
        "status": "online",
        "documents": rag.documents_loaded
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "documents_count": len(rag.documents_loaded),
        "documents": rag.documents_loaded
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(400, "Pregunta vacía")
        
        logger.info(f"📝 {request.question}")
        result = rag.query(request.question)
        
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"❌ {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
