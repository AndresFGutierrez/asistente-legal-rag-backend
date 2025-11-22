import os
from typing import List, Dict
from pypdf import PdfReader
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import glob
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.documents = []
        
    def load_pdfs(self, pdf_paths: List[str]):
        for pdf_path in pdf_paths[:1]:
            try:
                with open(pdf_path, "rb") as file:
                    reader = PdfReader(file)
                    text = ""
                    for page in reader.pages[:30]:
                        try:
                            text += page.extract_text() + "\n"
                        except:
                            continue
                    
                    self.documents.append({
                        "name": os.path.basename(pdf_path),
                        "content": text[:50000]
                    })
                    logger.info(f"✅ {os.path.basename(pdf_path)}")
            except Exception as e:
                logger.error(f"❌ {e}")
    
    def query(self, question: str) -> Dict:
        context = ""
        for doc in self.documents:
            lines = doc["content"].split("\n")
            relevant = [l for l in lines if any(w.lower() in l.lower() for w in question.split())][:5]
            context += "\n".join(relevant[:3]) + "\n\n"
        
        prompt = f"""Eres un asistente legal colombiano. Responde en español de forma clara y concisa (máximo 3-4 oraciones).

Contexto legal:
{context[:3000]}

Pregunta: {question}

Respuesta:"""
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                },
                timeout=30
            )
            
            answer = response.json()["choices"][0]["message"]["content"]
            
            return {
                "answer": answer,
                "sources": [{"content": context[:200], "source": self.documents[0]["name"]}],
                "documents_consulted": [d["name"] for d in self.documents]
            }
        except Exception as e:
            logger.error(f"❌ {e}")
            raise

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY requerida")

rag = SimpleRAG(openrouter_api_key=OPENROUTER_API_KEY)
DATASET_PATH = "../dataset"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando con OpenRouter...")
    pdfs = glob.glob(os.path.join(DATASET_PATH, "*.pdf"))
    rag.load_pdfs(pdfs)
    logger.info(f"✅ {len(rag.documents)} documentos listos")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    documents_consulted: List[str]

@app.get("/")
async def root():
    return {"message": "Asistente Legal RAG con OpenRouter", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy", "documents_count": len(rag.documents)}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        result = rag.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
