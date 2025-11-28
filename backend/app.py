# backend/app.py - Optimized for Render + Vercel Deployment
import os
import io
import uuid
import logging
from typing import Optional, List, Any
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
# Free Tier Optimizations for Render
# ---------------------------------------------------------------------

# Render-specific optimizations
if "RENDER" in os.environ:
    # Use Render's persistent disk for ChromaDB
    CHROMA_DIR = "/opt/render/project/src/chroma_data"
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    # Reduce memory usage
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Use smaller model for free tier
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Free tier limits
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_CHUNKS_PER_FILE = 200
    MAX_FILES_PER_UPLOAD = 2
else:
    # Local development settings
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma")
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB locally
    MAX_CHUNKS_PER_FILE = 500
    MAX_FILES_PER_UPLOAD = 5

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
if not "RENDER" in os.environ:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama config (optional)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Fixed admin creds
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")

JWT_SECRET = os.getenv("JWT_SECRET", "please_change_me_very_secure_key_here")
JWT_ALGO = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY required in environment variables")

# ---------------------------------------------------------------------
# Logging & app
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.app")

app = FastAPI(title="MSME PDF Q&A Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Free Tier Middleware
# ---------------------------------------------------------------------
@app.middleware("http")
async def free_tier_limits(request: Request, call_next):
    # File size limits for free tier
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": f"File size limited to {MAX_FILE_SIZE//1024//1024}MB on free tier"}
        )
    
    response = await call_next(request)
    return response

# ---------------------------------------------------------------------
# Check Ollama availability
# ---------------------------------------------------------------------
def _check_ollama_available(url: str) -> bool:
    if not url:
        return False
    try:
        health_url = f"{url.rstrip('/')}/api/tags"
        resp = requests.get(health_url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            available_models = [model.get("name") for model in models if model.get("name")]
            if OLLAMA_MODEL in available_models:
                logger.info(f"Ollama detected with model '{OLLAMA_MODEL}'")
                return True
            else:
                logger.warning(f"Ollama model '{OLLAMA_MODEL}' not found. Available: {available_models}")
                return False
        return False
    except Exception as e:
        logger.warning(f"Ollama check failed: {e}")
        return False

OLLAMA_AVAILABLE = _check_ollama_available(OLLAMA_URL)
if OLLAMA_AVAILABLE:
    logger.info("Ollama detected at %s with model %s", OLLAMA_URL, OLLAMA_MODEL)
else:
    logger.info("Ollama not detected (will use OpenAI)")

# ---------------------------------------------------------------------
# DB (SQLModel) for dataset metadata and logs
# ---------------------------------------------------------------------
if "RENDER" in os.environ:
    DATABASE_URL = f"sqlite:///{os.path.join('/opt/render/project/src', 'backend.db')}"
else:
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
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: Optional[str] = None
    dataset: Optional[str] = None
    language: Optional[str] = None
    model_used: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FeedbackLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: str
    feedback_type: str
    question: Optional[str] = None
    answer: Optional[str] = None
    dataset: Optional[str] = None
    language: Optional[str] = None
    model_used: Optional[str] = None
    timestamp: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Create tables
SQLModel.metadata.create_all(engine)

# ---------------------------------------------------------------------
# Auth helpers
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
# OpenAI client, embedder, Chroma
# ---------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBED_MODEL_NAME}")
except Exception as e:
    logger.exception("Failed to load embedder")
    raise

try:
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection("pdf_docs")
    logger.info(f"ChromaDB initialized at: {CHROMA_DIR}")
except Exception as e:
    logger.exception("Failed to init Chroma")
    raise

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    embs = _embedder.encode(texts, normalize_embeddings=True)
    return embs.tolist() if isinstance(embs, np.ndarray) else embs

# ---------------------------------------------------------------------
# Helpers: PDF read, OCR, chunking, prompt builder
# ---------------------------------------------------------------------
def read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception:
        return ""
    out: List[str] = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out)

def run_ocr_on_pdf(file_bytes: bytes) -> str:
    if pytesseract is None or convert_from_bytes is None:
        raise RuntimeError("OCR not available")
    parts: List[str] = []
    images = convert_from_bytes(file_bytes, fmt="jpeg")
    for image in images:
        try:
            parts.append(pytesseract.image_to_string(image))
        except Exception:
            continue
    return "\n".join(parts)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:  # Smaller chunks for free tier
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
# Ollama helper
# ---------------------------------------------------------------------
def ollama_generate(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> Optional[str]:
    if not OLLAMA_AVAILABLE or not OLLAMA_URL:
        logger.warning("Ollama not available")
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
        resp = requests.post(url, json=payload, timeout=(10.0, 60.0))
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict) and "response" in data:
            response_text = data["response"].strip()
            if response_text:
                logger.info(f"Ollama generation successful: {len(response_text)} chars")
                return response_text
        
        for field in ["text", "output", "content", "completion"]:
            if field in data and isinstance(data[field], str) and data[field].strip():
                logger.info(f"Ollama generation successful (fallback field {field})")
                return data[field].strip()
                
        logger.warning(f"Unexpected Ollama response format: {data}")
        return None
        
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Ollama connection failed")
        return None
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
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
    feedback: str
    question: Optional[str] = None
    answer: Optional[str] = None
    timestamp: Optional[str] = None

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "MSME PDF Q&A API", "status": "healthy", "free_tier": "active"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "msme-backend" ,"provider": "openai", "free_tier": True}

@app.get("/models")
def get_available_models():
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
    
    ollama_available = OLLAMA_AVAILABLE
    ollama_model_available = False
    
    if ollama_available:
        try:
            url = f"{OLLAMA_URL.rstrip('/')}/api/tags"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                available_models = [model.get("name") for model in data.get("models", []) if model.get("name")]
                ollama_model_available = OLLAMA_MODEL in available_models
        except:
            ollama_model_available = False
    
    if ollama_available and ollama_model_available:
        models.append({
            "id": "ollama",
            "name": f"Ollama {OLLAMA_MODEL}",
            "provider": "Ollama",
            "description": "Local AI model for privacy and offline use",
            "icon": "🦙",
            "available": True
        })
    elif ollama_available:
        models.append({
            "id": "ollama",
            "name": f"Ollama {OLLAMA_MODEL}",
            "provider": "Ollama",
            "description": f"Local AI model - model '{OLLAMA_MODEL}' not found",
            "icon": "🦙",
            "available": False
        })
    else:
        models.append({
            "id": "ollama",
            "name": f"Ollama {OLLAMA_MODEL}",
            "provider": "Ollama",
            "description": "Local AI model (Ollama service not available)",
            "icon": "🦙",
            "available": False
        })
    
    return {"models": models}

@app.post("/admin/login")
async def admin_login(payload: dict = Body(...)):
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
    try:
        try:
            if dataset:
                results = collection.get(where={"dataset": dataset})
            else:
                results = collection.get()
        except TypeError:
            if dataset:
                results = collection.get(filter={"dataset": dataset})
            else:
                results = collection.get()
        
        if not results or not results.get("metadatas"):
            return {"files": []}
        
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
    try:
        where_filter = {"filename": filename}
        if dataset:
            where_filter["dataset"] = dataset
        
        try:
            results = collection.get(where=where_filter)
        except TypeError:
            results = collection.get(filter=where_filter)
        
        if not results or not results.get("ids"):
            raise HTTPException(status_code=404, detail=f"No documents found for filename: {filename}")
        
        document_count = len(results["ids"])
        
        try:
            collection.delete(where=where_filter)
        except TypeError:
            collection.delete(filter=where_filter)
        
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
    try:
        try:
            results = collection.get(where={"dataset": dataset})
        except TypeError:
            results = collection.get(filter={"dataset": dataset})
        
        if not results or not results.get("ids"):
            raise HTTPException(status_code=404, detail=f"No documents found for dataset: {dataset}")
        
        document_count = len(results["ids"])
        filenames = set()
        if results.get("metadatas"):
            for metadata in results["metadatas"]:
                if isinstance(metadata, dict) and "filename" in metadata:
                    filenames.add(metadata["filename"])
        
        try:
            collection.delete(where={"dataset": dataset})
        except TypeError:
            collection.delete(filter={"dataset": dataset})
        
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
        except TypeError:
            results = collection.get()
        metadatas_any = results.get("metadatas") if isinstance(results, dict) else None
        if metadatas_any is None:
            metadatas_any = getattr(collection, "metadatas", []) or []
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
async def ingest(files: List[UploadFile] = File(...), dataset: Optional[str] = Form(None), ocr: Optional[bool] = Form(False)):
    # Free tier: limit files
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=400, detail=f"Free tier limited to {MAX_FILES_PER_UPLOAD} files at once")
    
    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF supported: {f.filename}")
        
        # Check file size
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File {f.filename} exceeds {MAX_FILE_SIZE//1024//1024}MB limit")
        
        content = read_pdf(data)
        used_ocr = False
        
        if not content.strip() and ocr:
            try:
                content = run_ocr_on_pdf(data)
                used_ocr = True
            except Exception as e:
                logger.warning("OCR failed for %s: %s", f.filename, e)
                content = ""
        
        if not content.strip():
            continue
            
        chunks = chunk_text(content)
        
        # Limit chunks for free tier
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            chunks = chunks[:MAX_CHUNKS_PER_FILE]
            
        if not chunks:
            continue
            
        texts.extend(chunks)
        meta = {"filename": f.filename, "ocr": used_ocr}
        if dataset:
            meta["dataset"] = dataset
        metadatas.extend([meta] * len(chunks))
        ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])
    
    if not texts:
        raise HTTPException(status_code=400, detail="No extractable text found in PDFs")
    
    # Limit total chunks for free tier
    if len(texts) > 500:
        texts = texts[:500]
        metadatas = metadatas[:500]
        ids = ids[:500]
    
    vectors = embed_texts(texts)
    collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)
    
    return IngestResponse(
        documents_added=len(set(m["filename"] for m in metadatas)), 
        chunks_indexed=len(texts)
    )

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    use_ollama = False
    model_name = req.model or "openai"
    
    if model_name and "ollama" in model_name.lower():
        use_ollama = True
        if not OLLAMA_AVAILABLE:
            raise HTTPException(status_code=400, detail="Ollama is not available")
    
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
    
    qvecs = embed_texts([q])
    if not qvecs:
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = "Error: Failed to embed query"
                session.commit()
        raise HTTPException(status_code=500, detail="Failed to embed query")
    
    qvec = qvecs[0]
    try:
        if req.dataset:
            results = collection.query(query_embeddings=[qvec], n_results=max(1, req.top_k), where={"dataset": req.dataset})
        else:
            results = collection.query(query_embeddings=[qvec], n_results=max(1, req.top_k))
    except TypeError:
        if req.dataset:
            results = collection.query(query_embeddings=[qvec], n_results=max(1, req.top_k), filter={"dataset": req.dataset})
        else:
            results = collection.query(query_embeddings=[qvec], n_results=max(1, req.top_k))
    except Exception as e:
        logger.exception("Vector DB query failed")
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = f"Error: Vector DB query failed - {str(e)}"
                session.commit()
        raise HTTPException(status_code=500, detail=str(e))

    contexts: List[str] = []
    try:
        docs = results.get("documents", [[]])
        if isinstance(docs, list) and len(docs) > 0:
            contexts = docs[0] if isinstance(docs[0], list) else list(docs)
    except Exception:
        try:
            raw_docs = results.get("documents")
            if isinstance(raw_docs, list):
                contexts = [d for d in raw_docs if isinstance(d, str)]
        except Exception:
            contexts = []

    if not contexts:
        answer_text = "I couldn't find relevant content in the indexed PDFs."
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = answer_text
                session.commit()
        return AskResponse(answer=answer_text)

    prompt = build_prompt(q, contexts, req.language)
    try:
        answer = None
        
        if use_ollama:
            try:
                system_text = "You are a helpful assistant."
                final_prompt = f"{system_text}\n\n{prompt}"
                answer = ollama_generate(final_prompt, temperature=0.2, max_tokens=1024)
                
                if answer is None:
                    logger.warning("Ollama generation failed, falling back to OpenAI")
                    completion = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                    )
                    answer = completion.choices[0].message.content.strip()
                    model_name = "openai (fallback)"
                else:
                    logger.info(f"Generated answer using Ollama: {OLLAMA_MODEL}")
                    
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}, falling back to OpenAI")
                completion = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                answer = completion.choices[0].message.content.strip()
                model_name = "openai (fallback)"
        else:
            try:
                completion = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                answer = completion.choices[0].message.content.strip()
                logger.info(f"Generated answer using OpenAI: {OPENAI_MODEL}")
            except Exception as e:
                logger.warning(f"OpenAI generation failed: {e}")
                answer = f"Error: OpenAI generation failed - {str(e)}"

    except Exception as e:
        logger.exception("LLM completion failed")
        answer = f"Error: Failed to generate answer - {str(e)}"

    with Session(engine) as session:
        log_entry = session.get(LoggedQuestion, log_id)
        if log_entry:
            log_entry.answer = answer
            log_entry.model_used = model_name
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
    content_type = request.headers.get("content-type", "")
    text = None
    lang = "en"
    voice = None
    speed = 1.0

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

    speed = max(0.25, min(4.0, speed))
    txt = (text or "").strip()
    if not txt:
        raise HTTPException(status_code=400, detail="text required")

    use_openai_tts = os.getenv("USE_OPENAI_TTS", "true").lower() in ("1", "true", "yes")

    if use_openai_tts:
        try:
            if hasattr(openai_client, "audio") and hasattr(openai_client.audio, "speech"):
                out = openai_client.audio.speech.create(
                    model="tts-1",
                    voice=voice or "alloy",
                    input=txt,
                    speed=speed
                )
                audio_bytes = getattr(out, "audio", None) or getattr(out, "data", None) or out
                if isinstance(audio_bytes, (bytes, bytearray)):
                    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
        except Exception as e:
            logger.warning("OpenAI TTS call failed: %s", repr(e))
            msg = str(e).lower()
            if "quota" in msg or "insufficient_quota" in msg or "429" in msg or "too many requests" in msg:
                logger.warning("OpenAI TTS quota/429 detected; falling back to gTTS")

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
async def admin_upload(dataset: str = Form(...), files: List[UploadFile] = File(...), admin = Depends(get_current_admin)):
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=400, detail=f"Free tier limited to {MAX_FILES_PER_UPLOAD} files at once")
    
    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF supported: {f.filename}")
        
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File {f.filename} exceeds {MAX_FILE_SIZE//1024//1024}MB limit")
        
        content = read_pdf(data)
        if not content.strip():
            try:
                content = run_ocr_on_pdf(data) if pytesseract else ""
            except Exception:
                content = ""
        if not content.strip():
            continue
            
        chunks = chunk_text(content)
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            chunks = chunks[:MAX_CHUNKS_PER_FILE]
            
        if not chunks:
            continue
            
        texts.extend(chunks)
        meta = {"filename": f.filename, "admin_username": admin.username, "dataset": dataset}
        metadatas.extend([meta] * len(chunks))
        ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])
    
    if not texts:
        raise HTTPException(status_code=400, detail="No extractable text found in PDFs")
    
    if len(texts) > 500:
        texts = texts[:500]
        metadatas = metadatas[:500]
        ids = ids[:500]
    
    vectors = embed_texts(texts)
    collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)

    with Session(engine) as session:
        prev = session.exec(select(DatasetMeta).where(DatasetMeta.name == dataset).order_by(DatasetMeta.version.desc())).first()
        version = 1 if not prev else prev.version + 1
        ds = DatasetMeta(name=dataset, created_by=admin.username, version=version, note=f"upload {len(texts)} chunks")
        session.add(ds)
        session.commit()
        session.refresh(ds)

    return {"ok": True, "dataset": dataset, "chunks_indexed": len(texts), "dataset_version": version}

# ---------------------------------------------------------------------
# Feedback Routes
# ---------------------------------------------------------------------
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
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
            
            total_query = select(FeedbackLog)
            total_count = session.exec(total_query).all()
            good_count = session.exec(total_query.where(FeedbackLog.feedback_type == "good")).all()
            bad_count = session.exec(total_query.where(FeedbackLog.feedback_type == "bad")).all()
            no_response_count = session.exec(total_query.where(FeedbackLog.feedback_type == "no_response")).all()
            
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, access_log=False)