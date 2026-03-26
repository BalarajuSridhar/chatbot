# backend/app.py
import os
import io
import uuid
import logging
import shutil
from typing import Optional, List, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Body, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# SQLModel for dataset metadata + logs
from sqlmodel import SQLModel, create_engine, Session, Field, select

# vector / embedding / PDF
import chromadb
from chromadb.config import Settings
import numpy as np
from pypdf import PdfReader

# Google Gemini API
import google.generativeai as genai

# Optional OCR + TTS fallbacks
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    convert_from_bytes = None
    OCR_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    gTTS = None
    GTTS_AVAILABLE = False

# JWT + password context
import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_data"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_CHUNKS = 1000

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-this-in-production")
JWT_ALGO = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required! Please set it in .env file or environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------
# Logging & app
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend.app")

app = FastAPI(title="PDF Q&A Chatbot with Gemini", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Database setup
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

SQLModel.metadata.create_all(engine)

# ---------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/token", auto_error=False)

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

async def get_current_admin(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    username = payload.get("username")
    if username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"username": username}

# ---------------------------------------------------------------------
# LAZY Embedder — loads only on first use, does NOT block startup
# ---------------------------------------------------------------------
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedder: {EMBED_MODEL_NAME} ...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
        logger.info("✓ Embedder loaded")
    return _embedder

# ---------------------------------------------------------------------
# LAZY ChromaDB — initializes only on first use, does NOT block startup
# ---------------------------------------------------------------------
_chroma_client = None
_collection = None

def get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    os.makedirs(CHROMA_DIR, exist_ok=True)
    try:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        _chroma_client.heartbeat()
        try:
            _collection = _chroma_client.get_collection("pdf_docs")
            logger.info("✓ Using existing ChromaDB collection")
        except Exception:
            _collection = _chroma_client.create_collection(
                name="pdf_docs",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✓ Created new ChromaDB collection")
        logger.info(f"✓ ChromaDB initialized at {CHROMA_DIR}")
    except Exception as e:
        logger.error(f"PersistentClient failed: {e} — falling back to in-memory")
        _chroma_client = chromadb.EphemeralClient()
        _collection = _chroma_client.get_or_create_collection("pdf_docs")
        logger.warning("⚠ Using in-memory ChromaDB (data will not persist)")

    return _collection

# ---------------------------------------------------------------------
# Embed helper
# ---------------------------------------------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        embedder = get_embedder()
        embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embs.tolist() if isinstance(embs, np.ndarray) else embs
    except Exception as e:
        logger.error(f"Failed to embed texts: {e}")
        raise

# ---------------------------------------------------------------------
# PDF processing helpers
# ---------------------------------------------------------------------
def read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text.strip())
            except Exception:
                continue
        return "\n".join(text_parts)
    except Exception:
        return ""

def run_ocr_on_pdf(file_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    parts = []
    try:
        images = convert_from_bytes(file_bytes, fmt="jpeg", dpi=200)
        for image in images:
            try:
                text = pytesseract.image_to_string(image)
                if text and text.strip():
                    parts.append(text.strip())
            except Exception:
                continue
        return "\n".join(parts)
    except Exception:
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = " ".join((text or "").split()).strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
    return chunks

LANG_NAME = {
    "en": "English", "te": "Telugu", "ta": "Tamil", "hi": "Hindi",
    "mr": "Marathi", "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali"
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
# Gemini helper
# ---------------------------------------------------------------------
def gemini_generate(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> Optional[str]:
    try:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.95,
            "top_k": 40,
        }
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=generation_config
        )
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            logger.warning("Gemini returned empty response")
            return None
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return None

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    language: Optional[str] = None
    dataset: Optional[str] = None

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
    return {
        "message": "PDF Q&A Chatbot API with Gemini",
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "ocr": OCR_AVAILABLE,
            "tts": GTTS_AVAILABLE,
            "model": GEMINI_MODEL,
            "provider": "Google Gemini"
        }
    }

@app.get("/health")
async def health():
    chroma_status = "not_initialized"
    db_status = "healthy"
    gemini_status = "not_tested"

    try:
        with Session(engine) as session:
            session.exec(select(LoggedQuestion).limit(1))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Only test chroma if already initialized (don't trigger lazy load here)
    if _chroma_client is not None:
        try:
            _chroma_client.heartbeat()
            chroma_status = "healthy"
        except Exception as e:
            chroma_status = f"unhealthy: {str(e)}"

    return {
        "status": "ok",
        "gemini_model": GEMINI_MODEL,
        "chromadb": chroma_status,
        "database": db_status,
        "ocr": OCR_AVAILABLE,
        "tts": GTTS_AVAILABLE,
        "embedder_loaded": _embedder is not None
    }

@app.get("/models")
async def get_available_models():
    return {
        "models": [
            {
                "id": "gemini",
                "name": f"Google {GEMINI_MODEL}",
                "provider": "Google Gemini",
                "description": "Google's advanced AI model",
                "icon": "🌟",
                "available": True
            }
        ]
    }

@app.post("/admin/login")
async def admin_login(payload: dict = Body(...)):
    username = payload.get("username")
    password = payload.get("password")
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(sub=ADMIN_USERNAME, username=ADMIN_USERNAME)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/admin/token")
async def admin_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_USERNAME or form_data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(sub=ADMIN_USERNAME, username=ADMIN_USERNAME)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/admin/logs")
async def get_logged_questions(
    limit: int = Query(200, gt=0, le=1000),
    since: Optional[str] = Query(None),
    admin=Depends(get_current_admin)
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
        return {"logs": [
            {
                "id": r.id,
                "question": r.question,
                "answer": r.answer,
                "dataset": r.dataset,
                "language": r.language,
                "model_used": r.model_used,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]}

@app.get("/datasets")
async def list_datasets():
    try:
        collection = get_collection()
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        datasets = set()
        for meta in metadatas:
            if isinstance(meta, dict) and "dataset" in meta:
                if meta["dataset"] and meta["dataset"].strip():
                    datasets.add(meta["dataset"].strip())
        return {"datasets": sorted(list(datasets))}
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_pdfs(
    files: List[UploadFile] = File(...),
    dataset: Optional[str] = Form(None),
    ocr: bool = Form(False)
):
    collection = get_collection()
    texts = []
    metadatas = []
    ids = []
    processed_files = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files supported: {f.filename}")

        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: {f.filename}")

        content = read_pdf(data)
        used_ocr = False

        if not content.strip() and ocr:
            try:
                content = run_ocr_on_pdf(data)
                used_ocr = True
            except Exception as e:
                logger.warning(f"OCR failed for {f.filename}: {e}")

        if not content.strip():
            logger.warning(f"⚠ No text found in {f.filename}")
            continue

        chunks = chunk_text(content)
        if not chunks:
            continue

        if len(chunks) > MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]

        meta = {
            "filename": f.filename,
            "ocr": used_ocr,
            "dataset": dataset or "default"
        }

        texts.extend(chunks)
        metadatas.extend([meta] * len(chunks))
        ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])
        processed_files.append(f.filename)
        logger.info(f"✓ Processed {f.filename}: {len(chunks)} chunks")

    if not texts:
        raise HTTPException(status_code=400, detail="No extractable text found in any PDF")

    try:
        vectors = embed_texts(texts)
        collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )
        return {
            "success": True,
            "documents_added": len(processed_files),
            "chunks_indexed": len(texts),
            "files": processed_files,
            "dataset": dataset or "default"
        }
    except Exception as e:
        logger.error(f"Failed to add to ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    with Session(engine) as session:
        log_entry = LoggedQuestion(
            question=question,
            dataset=req.dataset,
            language=req.language,
            model_used=GEMINI_MODEL
        )
        session.add(log_entry)
        session.commit()
        log_id = log_entry.id

    try:
        collection = get_collection()
        qvecs = embed_texts([question])
        if not qvecs:
            raise Exception("Empty embedding result")
        qvec = qvecs[0]

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

        contexts = []
        docs = results.get("documents", [[]])
        if docs and len(docs) > 0:
            contexts = docs[0] if isinstance(docs[0], list) else list(docs)

        if not contexts:
            answer = "I couldn't find relevant content in the indexed PDFs to answer your question."
        else:
            prompt = build_prompt(question, contexts, req.language)
            answer = gemini_generate(prompt, temperature=0.2, max_tokens=1024)
            if not answer:
                answer = "I apologize, but I couldn't generate a proper answer. Please try again."

        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = answer
                session.commit()

        return AskResponse(answer=answer)

    except Exception as e:
        logger.error(f"Ask endpoint error: {e}")
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = f"Error: {str(e)}"
                session.commit()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        with Session(engine) as session:
            feedback_entry = FeedbackLog(
                message_id=feedback.message_id,
                feedback_type=feedback.feedback,
                question=feedback.question,
                answer=feedback.answer,
                timestamp=feedback.timestamp or datetime.utcnow().isoformat()
            )
            session.add(feedback_entry)
            session.commit()
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/admin/feedback-logs")
async def get_feedback_logs(
    feedback_type: Optional[str] = Query(None),
    limit: int = Query(100, gt=0, le=1000),
    offset: int = Query(0, ge=0),
    admin=Depends(get_current_admin)
):
    try:
        with Session(engine) as session:
            query = select(FeedbackLog)
            if feedback_type:
                query = query.where(FeedbackLog.feedback_type == feedback_type)
            query = query.order_by(FeedbackLog.created_at.desc()).offset(offset).limit(limit)
            logs = session.exec(query).all()

            total = len(session.exec(select(FeedbackLog)).all())
            good = len(session.exec(select(FeedbackLog).where(FeedbackLog.feedback_type == "good")).all())
            bad = len(session.exec(select(FeedbackLog).where(FeedbackLog.feedback_type == "bad")).all())
            no_response = len(session.exec(select(FeedbackLog).where(FeedbackLog.feedback_type == "no_response")).all())

            return {
                "feedback_logs": [
                    {
                        "id": log.id,
                        "message_id": log.message_id,
                        "feedback_type": log.feedback_type,
                        "question": log.question,
                        "answer": log.answer,
                        "timestamp": log.timestamp,
                        "created_at": log.created_at.isoformat()
                    }
                    for log in logs
                ],
                "summary": {
                    "total": total,
                    "good": good,
                    "bad": bad,
                    "no_response": no_response
                }
            }
    except Exception as e:
        logger.error(f"Failed to fetch feedback logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch feedback logs")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")