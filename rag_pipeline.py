import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from pypdf import PdfReader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("🆓 Embeddings: HuggingFace (ligero)")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        from transformers import pipeline
        logger.info("🆓 LLM: GPT-2 Local")
        pipe = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=150,
            truncation=True
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.vector_store = None
        self.qa_chain = None
        self.documents_loaded = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                text = ""
                total_pages = len(reader.pages)
                
                logger.info(f"📄 {os.path.basename(pdf_path)} - {total_pages} pág")
                
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except:
                        continue
                
                if not text.strip():
                    raise ValueError("Sin texto")
                
                logger.info(f"✅ {len(text)} chars")
                return text
                
        except Exception as e:
            logger.error(f"❌ {os.path.basename(pdf_path)}: {str(e)}")
            raise
    
    def process_documents(self, pdf_paths: List[str]) -> None:
        all_chunks = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
        )
        
        # SOLO PRIMER PDF (para ahorrar memoria)
        pdf_paths = pdf_paths[:1]
        
        for pdf_path in pdf_paths:
            try:
                if not os.path.exists(pdf_path):
                    continue
                
                text = self.extract_text_from_pdf(pdf_path)
                doc_name = os.path.basename(pdf_path)
                chunks = text_splitter.split_text(text)
                
                for chunk in chunks:
                    all_chunks.append({
                        "content": chunk,
                        "source": doc_name
                    })
                
                self.documents_loaded.append(doc_name)
                logger.info(f"✅ {doc_name}: {len(chunks)} chunks")
                
            except:
                continue
        
        if not all_chunks:
            raise ValueError("Sin documentos")
        
        texts = [c["content"] for c in all_chunks]
        metadatas = [{"source": c["source"]} for c in all_chunks]
        
        logger.info(f"🔄 Indexando {len(texts)} chunks...")
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"✅ {len(texts)} chunks indexados")
        
        prompt_template = """Contexto:
{context}

Pregunta: {question}

Respuesta corta:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 1}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("✅ RAG listo")
    
    def query(self, question: str) -> Dict:
        if not self.qa_chain:
            raise ValueError("No inicializado")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "?")
                })
            
            return {
                "answer": result["result"],
                "sources": sources,
                "documents_consulted": list(set([s["source"] for s in sources]))
            }
        except Exception as e:
            logger.error(f"❌ {str(e)}")
            raise
