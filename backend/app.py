# backend/app.py
import os
import io
import uuid
import logging
import shutil
from typing import Optional
from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Body, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# SQLModel for dataset metadata + logs
from sqlmodel import SQLModel, create_engine, Session, Field, select

# vector / embedding / PDF / OpenAI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
from openai import OpenAI, RateLimitError

# Optional OCR + TTS fallbacks
try:
    import pytesseract
    from pdf2image import convert_from_bytes
except Exception:
    pytesseract = None
    convert_from_bytes = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# HTTP client for Ollama
import requests

# JWT + password context
import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Ollama config (optional)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Fixed admin creds
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")  # plaintext in env for dev

JWT_SECRET = os.getenv("JWT_SECRET", "please_change_me")
JWT_ALGO = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY required in backend/.env")

# ---------------------------------------------------------------------
# Logging & app
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.app")

app = FastAPI(title="PDF Q&A Chatbot (single-file backend)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Check Ollama availability (short, safe check)
# ---------------------------------------------------------------------
def _check_ollama_available(url: str) -> bool:
    if not url:
        return False
    try:
        health = f"{url.rstrip('/')}/api/tags"
        resp = requests.get(health, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

OLLAMA_AVAILABLE = _check_ollama_available(OLLAMA_URL)
if OLLAMA_AVAILABLE:
    logger.info("Ollama detected at %s", OLLAMA_URL)
else:
    logger.info("Ollama not detected (will use OpenAI). Set OLLAMA_URL to enable)")

# ---------------------------------------------------------------------
# DB (SQLModel) for dataset metadata and logs
# ---------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'backend.db')}")
engine = create_engine(DATABASE_URL, echo=False)

class DatasetMeta(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    created_by: Optional[str] = None
    version: int = 1
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class LoggedQuestion(SQLModel, table=True):
    """
    Stores each user question + assistant answer for admin viewing.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: Optional[str] = None
    dataset: Optional[str] = None
    language: Optional[str] = None
    model_used: Optional[str] = None  # Track which model was used
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FeedbackLog(SQLModel, table=True):
    """
    Stores user feedback for answers
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: str
    feedback_type: str  # 'good', 'bad', 'no_response'
    question: Optional[str] = None
    answer: Optional[str] = None
    dataset: Optional[str] = None
    language: Optional[str] = None
    model_used: Optional[str] = None  # Track which model was used
    timestamp: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Create tables
SQLModel.metadata.create_all(engine)

# Manual table alteration for existing databases
try:
    with Session(engine) as session:
        # Try to add model_used column to loggedquestion table if it doesn't exist
        session.execute("""
            PRAGMA table_info(loggedquestion)
        """)
        columns = [row[1] for row in session.execute("PRAGMA table_info(loggedquestion)")]
        
        if 'model_used' not in columns:
            session.execute("ALTER TABLE loggedquestion ADD COLUMN model_used VARCHAR")
            print("Added model_used column to loggedquestion table")
        
        # Try to add model_used column to feedbacklog table if it doesn't exist
        session.execute("""
            PRAGMA table_info(feedbacklog)
        """)
        columns = [row[1] for row in session.execute("PRAGMA table_info(feedbacklog)")]
        
        if 'model_used' not in columns:
            session.execute("ALTER TABLE feedbacklog ADD COLUMN model_used VARCHAR")
            print("Added model_used column to feedbacklog table")
        
        session.commit()
        print("Database schema updated successfully")
except Exception as e:
    print(f"Database schema update completed or not needed: {e}")

# ---------------------------------------------------------------------
# Auth helpers (fixed credentials + JWT)
# ---------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/token")

def create_access_token(sub: str, username: str, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = {"sub": sub, "username": username}
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGO)

def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except Exception:
        return None

def get_current_admin(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    sub = payload.get("sub")
    username = payload.get("username")
    if not sub or username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return type("AdminObj", (), {"username": username})

# ---------------------------------------------------------------------
# OpenAI client, embedder, Chroma - WITH ERROR HANDLING
# ---------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info(f"Embedder loaded: {EMBED_MODEL_NAME}")
except Exception as e:
    logger.exception("Failed to load embedder")
    raise

# Improved ChromaDB initialization with error handling
def initialize_chromadb():
    global chroma_client, collection
    
    # Clean up corrupted database if exists
    if os.path.exists(CHROMA_DIR):
        try:
            # Test if database is healthy
            test_client = chromadb.PersistentClient(path=CHROMA_DIR)
            test_client.heartbeat()
            test_client.list_collections()
            chroma_client = test_client
            logger.info("Existing ChromaDB database is healthy")
        except Exception as e:
            logger.warning(f"Existing ChromaDB database may be corrupted: {e}")
            logger.info("Recreating ChromaDB database...")
            shutil.rmtree(CHROMA_DIR)
            chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    else:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Get or create collection
    try:
        collection = chroma_client.get_or_create_collection(
            name="pdf_docs",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB collection initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize ChromaDB collection")
        raise

try:
    initialize_chromadb()
except Exception as e:
    logger.exception("ChromaDB initialization failed, trying in-memory client as fallback")
    try:
        chroma_client = chromadb.EphemeralClient()
        collection = chroma_client.get_or_create_collection("pdf_docs")
        logger.info("Using in-memory ChromaDB client as fallback")
    except Exception as e:
        logger.exception("All ChromaDB clients failed")
        raise

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        embs = _embedder.encode(texts, normalize_embeddings=True)
        return embs.tolist() if isinstance(embs, np.ndarray) else embs
    except Exception as e:
        logger.exception("Failed to embed texts")
        raise

# ---------------------------------------------------------------------
# Helpers: PDF read, OCR, chunking, prompt builder
# ---------------------------------------------------------------------
def read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        logger.warning(f"Failed to read PDF: {e}")
        return ""
    out: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text and text.strip():
                out.append(text.strip())
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF page: {e}")
            continue
    return "\n".join(out)

def run_ocr_on_pdf(file_bytes: bytes) -> str:
    if pytesseract is None or convert_from_bytes is None:
        raise RuntimeError("OCR dependencies not available")
    parts: List[str] = []
    try:
        images = convert_from_bytes(file_bytes, fmt="jpeg", dpi=200)
        for image in images:
            try:
                text = pytesseract.image_to_string(image)
                if text and text.strip():
                    parts.append(text.strip())
            except Exception as e:
                logger.warning(f"OCR failed for image: {e}")
                continue
        return "\n".join(parts)
    except Exception as e:
        logger.exception("OCR processing failed")
        raise

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = " ".join((text or "").split()).strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    step = max(1, chunk_size - overlap)
    n = len(text)
    chunks: List[str] = []
    for start in range(0, n, step):
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
    return chunks

LANG_NAME = {
    "en": "English", 
    "te": "Telugu", 
    "ta": "Tamil", 
    "hi": "Hindi",
    "mr": "Marathi",
    "kn": "Kannada", 
    "ml": "Malayalam",
    "bn": "Bengali"
}

def build_prompt(question: str, contexts: List[str], lang_code: Optional[str]) -> str:
    language_name = LANG_NAME.get((lang_code or "en").lower(), "English")
    context_block = "\n\n".join([f"[Source {i+1}]\n{c}" for i, c in enumerate(contexts)])
    return (
        "You are a helpful assistant. Use only the provided context to answer. "
        "If the answer isn't in the context, say you don't know. "
        "Reply with ONLY the final answer text — no preface, no citations, no extra lines. "
        f"Respond in {language_name}.\n\n"
        f"Question: {question}\n\nContext:\n{context_block}"
    )

# ---------------------------------------------------------------------
# Ollama helper (improved with better error handling)
# ---------------------------------------------------------------------
def ollama_generate(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> Optional[str]:
    """
    Call local Ollama HTTP API to generate text. Returns None on failure.
    """
    if not OLLAMA_AVAILABLE or not OLLAMA_URL:
        return None
    
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        }
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle Ollama response format
        if isinstance(data, dict):
            # Try different response formats
            if "response" in data:
                return data["response"].strip()
            elif "text" in data:
                return data["text"].strip()
            elif "content" in data:
                return data["content"].strip()
            elif "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
        
        logger.warning(f"Unexpected Ollama response format: {data}")
        return None
        
    except requests.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse Ollama response: {e}")
        return None

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    language: Optional[str] = None
    dataset: Optional[str] = None
    model: Optional[str] = None

class IngestResponse(BaseModel):
    documents_added: int
    chunks_indexed: int

class AskResponse(BaseModel):
    answer: str

class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str  # 'good', 'bad', 'no_response'
    question: Optional[str] = None
    answer: Optional[str] = None
    timestamp: Optional[str] = None

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "PDF Q&A Chatbot API", "status": "healthy"}

@app.get("/models")
def get_available_models():
    """Get available AI models"""
    models = [
        {
            "id": "openai",
            "name": f"OpenAI {OPENAI_MODEL}",
            "provider": "OpenAI",
            "description": "Cloud-based AI model with high accuracy",
            "icon": "🔮",
            "available": True
        }
    ]
    
    if OLLAMA_AVAILABLE:
        models.append({
            "id": "ollama",
            "name": f"Ollama {OLLAMA_MODEL}",
            "provider": "Ollama",
            "description": "Local AI model for privacy and offline use",
            "icon": "🦙",
            "available": True
        })
    
    return {"models": models}

@app.post("/admin/login")
async def admin_login(payload: dict = Body(...)):
    """
    Simple JSON login for frontend convenience.
    POST { "username": "...", "password": "..." }
    Returns { access_token, token_type } on success.
    """
    username = payload.get("username")
    password = payload.get("password")
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(sub=ADMIN_USERNAME, username=ADMIN_USERNAME)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/admin/token")
def admin_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_USERNAME or form_data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(sub=ADMIN_USERNAME, username=ADMIN_USERNAME)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/admin/logs")
def get_logged_questions(
    limit: int = Query(200, gt=0, le=1000), 
    since: Optional[str] = Query(None),
    admin = Depends(get_current_admin)
):
    """
    Return recent logged questions. Query params:
    - limit: max number of rows (default 200, max 1000)
    - since: ISO datetime string to filter from (inclusive)
    """
    with Session(engine) as session:
        q = select(LoggedQuestion).order_by(LoggedQuestion.created_at.desc())
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                q = q.where(LoggedQuestion.created_at >= since_dt)
            except Exception as e:
                logger.warning(f"Invalid 'since' param {since}: {e}")
        rows = session.exec(q.limit(limit)).all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "question": r.question,
                "answer": r.answer,
                "dataset": r.dataset,
                "language": r.language,
                "model_used": r.model_used,
                "created_at": r.created_at.isoformat(),
            })
        return {"logs": out}

@app.get("/admin/files")
async def list_uploaded_files(
    dataset: Optional[str] = Query(None, description="Filter by dataset"),
    admin = Depends(get_current_admin)
):
    """
    List all uploaded PDF files with their chunk counts.
    """
    try:
        # Get all documents
        try:
            if dataset:
                results = collection.get(where={"dataset": dataset})
            else:
                results = collection.get()
        except Exception:
            # Fallback for different ChromaDB versions
            if dataset:
                results = collection.get(where={"dataset": dataset})
            else:
                results = collection.get()
        
        if not results or not results.get("metadatas"):
            return {"files": []}
        
        # Count files and chunks
        file_stats = {}
        metadatas = results["metadatas"]
        
        for metadata in metadatas:
            if isinstance(metadata, dict) and "filename" in metadata:
                filename = metadata["filename"]
                file_dataset = metadata.get("dataset", "default")
                
                if filename not in file_stats:
                    file_stats[filename] = {
                        "filename": filename,
                        "dataset": file_dataset,
                        "chunk_count": 0,
                        "has_ocr": metadata.get("ocr", False),
                        "admin_username": metadata.get("admin_username")
                    }
                file_stats[filename]["chunk_count"] += 1
        
        files_list = list(file_stats.values())
        files_list.sort(key=lambda x: x["filename"])
        
        return {"files": files_list}
        
    except Exception as e:
        logger.exception("Failed to list files")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.delete("/admin/delete/file")
async def delete_file_by_filename(
    filename: str = Query(..., description="Name of the PDF file to delete"),
    dataset: Optional[str] = Query(None, description="Specific dataset to search in"),
    admin = Depends(get_current_admin)
):
    """
    Delete all chunks from a specific PDF file.
    """
    try:
        # Build filter for ChromaDB
        where_filter = {"filename": filename}
        if dataset:
            where_filter["dataset"] = dataset
        
        # Get the documents first to count them
        try:
            results = collection.get(where=where_filter)
        except Exception:
            results = collection.get(where=where_filter)
        
        if not results or not results.get("ids"):
            raise HTTPException(status_code=404, detail=f"No documents found for filename: {filename}")
        
        document_count = len(results["ids"])
        
        # Delete from ChromaDB
        try:
            collection.delete(where=where_filter)
        except Exception:
            collection.delete(where=where_filter)
        
        logger.info(f"Deleted {document_count} chunks from file: {filename}")
        
        return {
            "deleted": True,
            "filename": filename,
            "chunks_removed": document_count,
            "dataset": dataset
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete file {filename}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.delete("/admin/delete/dataset")
async def delete_entire_dataset(
    dataset: str = Query(..., description="Name of the dataset to delete entirely"),
    admin = Depends(get_current_admin)
):
    """
    Delete all documents from a specific dataset.
    """
    try:
        # Get the documents first to count them
        try:
            results = collection.get(where={"dataset": dataset})
        except Exception:
            results = collection.get(where={"dataset": dataset})
        
        if not results or not results.get("ids"):
            raise HTTPException(status_code=404, detail=f"No documents found for dataset: {dataset}")
        
        document_count = len(results["ids"])
        filenames = set()
        if results.get("metadatas"):
            for metadata in results["metadatas"]:
                if isinstance(metadata, dict) and "filename" in metadata:
                    filenames.add(metadata["filename"])
        
        # Delete from ChromaDB
        try:
            collection.delete(where={"dataset": dataset})
        except Exception:
            collection.delete(where={"dataset": dataset})
        
        # Also remove from DatasetMeta table
        with Session(engine) as session:
            datasets = session.exec(select(DatasetMeta).where(DatasetMeta.name == dataset)).all()
            for ds in datasets:
                session.delete(ds)
            session.commit()
        
        logger.info(f"Deleted dataset '{dataset}' with {document_count} chunks from {len(filenames)} files")
        
        return {
            "deleted": True,
            "dataset": dataset,
            "chunks_removed": document_count,
            "files_affected": list(filenames)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete dataset {dataset}")
        raise HTTPException(status_code=500, detail=f"Dataset deletion failed: {str(e)}")

@app.get("/datasets")
def list_datasets():
    try:
        try:
            results = collection.get(include=["metadatas"], limit=10000)
        except Exception:
            results = collection.get()
        metadatas_any = results.get("metadatas") if isinstance(results, dict) else None
        if metadatas_any is None:
            metadatas_any = []
        datasets = set()
        flat_meta_entries: List[Any] = []
        if isinstance(metadatas_any, list):
            for entry in metadatas_any:
                if isinstance(entry, (list, tuple)):
                    for e in entry:
                        flat_meta_entries.append(e)
                else:
                    flat_meta_entries.append(entry)
        for m in flat_meta_entries:
            if m is None:
                continue
            if isinstance(m, dict):
                val = m.get("dataset")
                if isinstance(val, str) and val.strip():
                    datasets.add(val.strip())
            elif isinstance(m, str):
                datasets.add(m.strip())
        return {"datasets": sorted(list(datasets))}
    except Exception as e:
        logger.exception("Failed to list datasets")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: List[UploadFile] = File(...), 
    dataset: Optional[str] = Form(None), 
    ocr: Optional[bool] = Form(False)
):
    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF supported: {f.filename}")
        
        data = await f.read()
        content = read_pdf(data)
        used_ocr = False
        
        if not content.strip() and ocr:
            try:
                content = run_ocr_on_pdf(data)
                used_ocr = True
                logger.info(f"Used OCR for file: {f.filename}")
            except Exception as e:
                logger.warning("OCR failed for %s: %s", f.filename, e)
                content = ""
        
        if not content.strip():
            logger.warning(f"No extractable text found in: {f.filename}")
            continue
        
        chunks = chunk_text(content)
        if not chunks:
            logger.warning(f"No chunks created from: {f.filename}")
            continue
        
        texts.extend(chunks)
        meta = {"filename": f.filename, "ocr": used_ocr}
        if dataset:
            meta["dataset"] = dataset
        metadatas.extend([meta] * len(chunks))
        ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])
    
    if not texts:
        raise HTTPException(status_code=400, detail="No extractable text found in PDFs")
    
    try:
        vectors = embed_texts(texts)
        collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(texts)} chunks from {len(set(m['filename'] for m in metadatas))} files")
    except Exception as e:
        logger.exception("Failed to add documents to ChromaDB")
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {str(e)}")
    
    return IngestResponse(
        documents_added=len(set(m["filename"] for m in metadatas)), 
        chunks_indexed=len(texts)
    )

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Determine which model to use
    use_ollama = False
    model_name = req.model or "openai"
    
    if model_name and "ollama" in model_name.lower():
        use_ollama = True
        if not OLLAMA_AVAILABLE:
            raise HTTPException(status_code=400, detail="Ollama is not available")
    
    # Log the question immediately (we'll update with answer later)
    with Session(engine) as session:
        log_entry = LoggedQuestion(
            question=q,
            dataset=req.dataset,
            language=req.language,
            model_used=model_name
        )
        session.add(log_entry)
        session.commit()
        log_id = log_entry.id
    
    # Embed the question
    try:
        qvecs = embed_texts([q])
        if not qvecs:
            raise Exception("Empty embedding result")
        qvec = qvecs[0]
    except Exception as e:
        logger.exception("Failed to embed query")
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = "Error: Failed to embed query"
                session.commit()
        raise HTTPException(status_code=500, detail="Failed to process query")
    
    # Query ChromaDB
    try:
        if req.dataset:
            results = collection.query(
                query_embeddings=[qvec], 
                n_results=max(1, req.top_k), 
                where={"dataset": req.dataset}
            )
        else:
            results = collection.query(
                query_embeddings=[qvec], 
                n_results=max(1, req.top_k)
            )
    except Exception as e:
        logger.exception("Vector DB query failed")
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = f"Error: Vector DB query failed - {str(e)}"
                session.commit()
        raise HTTPException(status_code=500, detail=str(e))

    # Extract contexts
    contexts: List[str] = []
    try:
        docs = results.get("documents", [[]])
        if isinstance(docs, list) and len(docs) > 0:
            contexts = docs[0] if isinstance(docs[0], list) else list(docs)
    except Exception as e:
        logger.warning("Failed to extract documents from results: %s", e)
        contexts = []

    if not contexts:
        answer_text = "I couldn't find relevant content in the indexed PDFs to answer your question."
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = answer_text
                session.commit()
        return AskResponse(answer=answer_text)

    # Generate answer
    prompt = build_prompt(q, contexts, req.language)
    answer = None
    
    try:
        if use_ollama:
            # Use Ollama
            try:
                system_text = "You are a helpful assistant that answers questions based on the provided context."
                final_prompt = f"{system_text}\n\n{prompt}"
                answer = ollama_generate(final_prompt, temperature=0.2, max_tokens=1024)
                if answer is None:
                    raise RuntimeError("Ollama generation failed or returned no output")
                logger.info(f"Generated answer using Ollama: {OLLAMA_MODEL}")
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}")
                answer = f"I apologize, but I encountered an error while generating the answer. Please try again."
        else:
            # Use OpenAI
            try:
                completion = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1024
                )
                answer = completion.choices[0].message.content.strip()
                logger.info(f"Generated answer using OpenAI: {OPENAI_MODEL}")
            except Exception as e:
                logger.warning(f"OpenAI generation failed: {e}")
                answer = f"I apologize, but I encountered an error while generating the answer. Please try again."

    except Exception as e:
        logger.exception("LLM completion failed")
        answer = f"I apologize, but I encountered an error while generating the answer. Please try again."

    # Update the log entry with the final answer
    with Session(engine) as session:
        log_entry = session.get(LoggedQuestion, log_id)
        if log_entry:
            log_entry.answer = answer
            session.commit()

    return AskResponse(answer=answer)

@app.post("/stt")
async def stt(audio: UploadFile = File(...), lang: Optional[str] = Form(None)):
    allowed = {"audio/webm", "audio/ogg", "audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a", "audio/aac"}
    ct = (audio.content_type or "").lower()
    if ct not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {ct}")
    
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    buf = io.BytesIO(data)
    if "webm" in ct:
        ext = ".webm"
    elif "ogg" in ct:
        ext = ".ogg"
    elif "mpeg" in ct:
        ext = ".mp3"
    elif "wav" in ct:
        ext = ".wav"
    else:
        ext = ".m4a"
    
    buf.name = f"recording{ext}"
    
    try:
        kwargs = {"model": "whisper-1", "file": buf}
        # Add support for new Indian languages in STT
        supported_langs = {"en", "te", "ta", "hi", "mr", "kn", "ml", "bn"}
        if lang and lang.lower() in supported_langs:
            kwargs["language"] = lang.lower()
        
        resp = openai_client.audio.transcriptions.create(**kwargs)
        text = (getattr(resp, "text", None) or "").strip()
        return {"text": text}
    
    except RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI STT quota exceeded")
    except Exception as e:
        logger.exception("STT failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def tts(request: Request):
    """
    Accept JSON { text, language, voice, speed } OR form data.
    Returns audio/mpeg stream (gTTS fallback if OpenAI TTS not available).
    """
    content_type = request.headers.get("content-type", "")
    text = None
    lang = "en"
    voice = None
    speed = 1.0

    use_openai_tts = os.getenv("USE_OPENAI_TTS", "true").lower() in ("1", "true", "yes")

    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text", "")
        lang = body.get("language", body.get("lang", "en"))
        voice = body.get("voice")
        speed = body.get("speed", 1.0)
    else:
        form = await request.form()
        text = form.get("text") or ""
        lang = form.get("lang") or form.get("language") or "en"
        voice = form.get("voice")
        try:
            speed = float(form.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0

    # Validate speed (OpenAI allows 0.25 to 4.0)
    speed = max(0.25, min(4.0, speed))

    txt = (text or "").strip()
    if not txt:
        raise HTTPException(status_code=400, detail="text required")

    # If OpenAI TTS is enabled
    if use_openai_tts:
        try:
            if hasattr(openai_client, "audio") and hasattr(openai_client.audio, "speech"):
                out = openai_client.audio.speech.create(
                    model="tts-1",
                    voice=voice or "alloy",
                    input=txt,
                    speed=speed
                )
                audio_bytes = out.content
                return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
        except Exception as e:
            logger.warning("OpenAI TTS call failed: %s", repr(e))
            # Fall through to gTTS

    # Fallback to gTTS
    if gTTS is None:
        raise HTTPException(status_code=500, detail="No TTS backend available")
    
    try:
        gtts_lang_map = {
            "en": "en", "hi": "hi", "ta": "ta", "te": "te",
            "mr": "mr", "kn": "kn", "ml": "ml", "bn": "bn"
        }
        tts_lang = gtts_lang_map.get(lang, "en")
        tts = gTTS(txt, lang=tts_lang)
        bio = io.BytesIO()
        tts.write_to_fp(bio)
        bio.seek(0)
        return StreamingResponse(bio, media_type="audio/mpeg")
    except Exception as e:
        logger.exception("gTTS failed")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.post("/admin/upload")
async def admin_upload(
    dataset: str = Form(...), 
    files: List[UploadFile] = File(...), 
    admin = Depends(get_current_admin)
):
    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF supported: {f.filename}")
        
        data = await f.read()
        content = read_pdf(data)
        
        # Try OCR if no text found
        if not content.strip():
            try:
                content = run_ocr_on_pdf(data) if pytesseract else ""
            except Exception as e:
                logger.warning(f"OCR failed for {f.filename}: {e}")
                content = ""
        
        if not content.strip():
            logger.warning(f"No extractable text found in: {f.filename}")
            continue
        
        chunks = chunk_text(content)
        if not chunks:
            logger.warning(f"No chunks created from: {f.filename}")
            continue
        
        texts.extend(chunks)
        meta = {
            "filename": f.filename, 
            "admin_username": admin.username, 
            "dataset": dataset
        }
        metadatas.extend([meta] * len(chunks))
        ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])
    
    if not texts:
        raise HTTPException(status_code=400, detail="No extractable text found in PDFs")
    
    try:
        vectors = embed_texts(texts)
        collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)
        logger.info(f"Admin uploaded {len(texts)} chunks to dataset: {dataset}")
    except Exception as e:
        logger.exception("Failed to add documents to ChromaDB")
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {str(e)}")

    # Update dataset metadata table
    with Session(engine) as session:
        prev = session.exec(
            select(DatasetMeta)
            .where(DatasetMeta.name == dataset)
            .order_by(DatasetMeta.version.desc())
        ).first()
        version = 1 if not prev else prev.version + 1
        ds = DatasetMeta(
            name=dataset, 
            created_by=admin.username, 
            version=version, 
            note=f"upload {len(texts)} chunks from {len(files)} files"
        )
        session.add(ds)
        session.commit()

    return {
        "ok": True, 
        "dataset": dataset, 
        "chunks_indexed": len(texts), 
        "dataset_version": version,
        "files_processed": len([f for f in files if f.filename])
    }

# ---------------------------------------------------------------------
# Feedback Routes
# ---------------------------------------------------------------------

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Store user feedback in the database"""
    try:
        timestamp = feedback.timestamp or datetime.utcnow().isoformat()
        
        with Session(engine) as session:
            feedback_entry = FeedbackLog(
                message_id=feedback.message_id,
                feedback_type=feedback.feedback,
                question=feedback.question,
                answer=feedback.answer,
                dataset=None,
                language=None,
                model_used=None,
                timestamp=timestamp
            )
            session.add(feedback_entry)
            session.commit()
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.exception("Failed to record feedback")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/admin/feedback-logs")
async def get_feedback_logs(
    feedback_type: Optional[str] = Query(None, description="Filter by feedback type"),
    limit: int = Query(100, gt=0, le=1000),
    offset: int = Query(0, ge=0),
    admin = Depends(get_current_admin)
):
    """Get feedback logs with filtering options"""
    try:
        with Session(engine) as session:
            query = select(FeedbackLog)
            
            if feedback_type:
                query = query.where(FeedbackLog.feedback_type == feedback_type)
            
            query = query.order_by(FeedbackLog.created_at.desc()).offset(offset).limit(limit)
            feedback_logs = session.exec(query).all()
            
            logs = []
            for log in feedback_logs:
                logs.append({
                    "id": log.id,
                    "message_id": log.message_id,
                    "feedback_type": log.feedback_type,
                    "question": log.question,
                    "answer": log.answer,
                    "dataset": log.dataset,
                    "language": log.language,
                    "model_used": log.model_used,
                    "timestamp": log.timestamp,
                    "created_at": log.created_at.isoformat()
                })
            
            # Get summary counts
            total_count = session.exec(select(FeedbackLog)).all()
            good_count = session.exec(select(FeedbackLog).where(FeedbackLog.feedback_type == "good")).all()
            bad_count = session.exec(select(FeedbackLog).where(FeedbackLog.feedback_type == "bad")).all()
            no_response_count = session.exec(select(FeedbackLog).where(FeedbackLog.feedback_type == "no_response")).all()
            
            summary = {
                "total": len(total_count),
                "good": len(good_count),
                "bad": len(bad_count),
                "no_response": len(no_response_count)
            }
            
            return {
                "feedback_logs": logs,
                "summary": summary,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": len(total_count)
                }
            }
            
    except Exception as e:
        logger.exception("Failed to fetch feedback logs")
        raise HTTPException(status_code=500, detail="Failed to fetch feedback logs")

@app.get("/admin/feedback-stats")
async def get_feedback_stats(
    days: Optional[int] = Query(30, description="Number of days to analyze"),
    admin = Depends(get_current_admin)
):
    """Get feedback statistics and trends"""
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        
        with Session(engine) as session:
            daily_query = session.exec(
                select(FeedbackLog).where(FeedbackLog.created_at >= since_date)
            ).all()
            
            daily_stats = {}
            for log in daily_query:
                date_str = log.created_at.date().isoformat()
                if date_str not in daily_stats:
                    daily_stats[date_str] = {"good": 0, "bad": 0, "no_response": 0, "total": 0}
                
                daily_stats[date_str][log.feedback_type] += 1
                daily_stats[date_str]["total"] += 1
            
            daily_trends = [
                {"date": date, **counts} 
                for date, counts in daily_stats.items()
            ]
            daily_trends.sort(key=lambda x: x["date"])
            
            return {
                "daily_trends": daily_trends,
                "period_days": days
            }
            
    except Exception as e:
        logger.exception("Failed to fetch feedback stats")
        raise HTTPException(status_code=500, detail="Failed to fetch feedback stats")

@app.get("/health")
def health():
    try:
        # Test ChromaDB
        chroma_client.heartbeat()
        chroma_status = "healthy"
        
        # Test database
        with Session(engine) as session:
            session.exec(select(LoggedQuestion).limit(1))
        db_status = "healthy"
        
    except Exception as e:
        chroma_status = f"unhealthy: {str(e)}"
        db_status = "unhealthy"
    
    return {
        "status": "ok", 
        "provider": "openai",
        "chromadb": chroma_status,
        "database": db_status,
        "ollama": "available" if OLLAMA_AVAILABLE else "unavailable"
    }

@app.post("/admin/reset-chromadb")
async def reset_chromadb(admin = Depends(get_current_admin)):
    """
    Reset the ChromaDB database (for emergency recovery)
    """
    try:
        global chroma_client, collection
        
        # Delete existing database
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
            logger.info("Deleted existing ChromaDB directory")
        
        # Reinitialize
        initialize_chromadb()
        
        return {"status": "success", "message": "ChromaDB reset successfully"}
        
    except Exception as e:
        logger.exception("Failed to reset ChromaDB")
        raise HTTPException(status_code=500, detail=f"Failed to reset ChromaDB: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)