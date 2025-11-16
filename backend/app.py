# backend/app.py
import os
import io
import uuid
import logging
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
    created_at: datetime = Field(default_factory=datetime.utcnow)

SQLModel.metadata.create_all(engine)

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
# OpenAI client, embedder, Chroma
# ---------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    logger.exception("Failed to load embedder")
    raise

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=True))
    collection = chroma_client.get_or_create_collection("pdf_docs")
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
# Schemas
# ---------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    language: Optional[str] = None
    dataset: Optional[str] = None

class IngestResponse(BaseModel):
    documents_added: int
    chunks_indexed: int

class AskResponse(BaseModel):
    answer: str

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

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
                # Continue without the since filter if it's invalid
        rows = session.exec(q.limit(limit)).all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "question": r.question,
                "answer": r.answer,
                "dataset": r.dataset,
                "language": r.language,
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
        except TypeError:
            if dataset:
                results = collection.get(filter={"dataset": dataset})
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
        except TypeError:
            results = collection.get(filter=where_filter)
        
        if not results or not results.get("ids"):
            raise HTTPException(status_code=404, detail=f"No documents found for filename: {filename}")
        
        document_count = len(results["ids"])
        
        # Delete from ChromaDB
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
    """
    Delete all documents from a specific dataset.
    """
    try:
        # Get the documents first to count them
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
        
        # Delete from ChromaDB
        try:
            collection.delete(where={"dataset": dataset})
        except TypeError:
            collection.delete(filter={"dataset": dataset})
        
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
            except Exception as e:
                logger.warning("OCR failed for %s: %s", f.filename, e)
                content = ""
        if not content.strip():
            continue
        chunks = chunk_text(content)
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
    vectors = embed_texts(texts)
    collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)
    return IngestResponse(documents_added=len(set(m["filename"] for m in metadatas)), chunks_indexed=len(texts))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Log the question immediately (we'll update with answer later)
    with Session(engine) as session:
        log_entry = LoggedQuestion(
            question=q,
            dataset=req.dataset,
            language=req.language
        )
        session.add(log_entry)
        session.commit()
        log_id = log_entry.id
    
    qvecs = embed_texts([q])
    if not qvecs:
        # Update log with error
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
        # Update log with error
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
        # Update log with no context found
        with Session(engine) as session:
            log_entry = session.get(LoggedQuestion, log_id)
            if log_entry:
                log_entry.answer = answer_text
                session.commit()
        return AskResponse(answer=answer_text)

    prompt = build_prompt(q, contexts, req.language)
    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        # defensively read response
        try:
            answer = completion.choices[0].message.content.strip()
        except Exception:
            answer = getattr(completion, "text", "") or str(completion)
    except Exception as e:
        logger.exception("LLM completion failed")
        answer = f"Error: Failed to generate answer - {str(e)}"

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
    speed = 1.0  # Default speed (1.0 = normal)

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

    # Try OpenAI TTS (if your OpenAI client supports it)
    try:
        if hasattr(openai_client, "audio") and hasattr(openai_client.audio, "speech"):
            # Use OpenAI TTS with speed control
            out = openai_client.audio.speech.create(
                model="tts-1",  # or "tts-1-hd" for higher quality
                voice=voice or "alloy",
                input=txt,
                speed=speed  # Control speed here (0.25 to 4.0)
            )
            audio_bytes = getattr(out, "audio", None) or getattr(out, "data", None) or out
            if isinstance(audio_bytes, (bytes, bytearray)):
                return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except RateLimitError as e:
        logger.warning(f"OpenAI TTS quota exceeded, falling back to gTTS: {e}")
        # Continue to gTTS fallback
    except Exception as e:
        logger.warning(f"OpenAI TTS failed, falling back to gTTS: {e}")
        # Continue to gTTS fallback

    # Fallback to gTTS - add support for new Indian languages
    if gTTS is None:
        raise HTTPException(status_code=500, detail="No TTS backend available (install gTTS or enable OpenAI TTS)")
    try:
        # gTTS language codes for Indian languages
        gtts_lang_map = {
            "en": "en", "hi": "hi", "ta": "ta", "te": "te",
            "mr": "mr",  # Marathi
            "kn": "kn",  # Kannada  
            "ml": "ml",  # Malayalam
            "bn": "bn"   # Bengali
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
    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF supported: {f.filename}")
        data = await f.read()
        content = read_pdf(data)
        if not content.strip():
            try:
                content = run_ocr_on_pdf(data) if pytesseract else ""
            except Exception:
                content = ""
        if not content.strip():
            continue
        chunks = chunk_text(content)
        if not chunks:
            continue
        texts.extend(chunks)
        meta = {"filename": f.filename, "admin_username": admin.username, "dataset": dataset}
        metadatas.extend([meta] * len(chunks))
        ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])
    if not texts:
        raise HTTPException(status_code=400, detail="No extractable text found in PDFs")
    vectors = embed_texts(texts)
    collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)

    # update dataset metadata table
    with Session(engine) as session:
        prev = session.exec(select(DatasetMeta).where(DatasetMeta.name == dataset).order_by(DatasetMeta.version.desc())).first()
        version = 1 if not prev else prev.version + 1
        ds = DatasetMeta(name=dataset, created_by=admin.username, version=version, note=f"upload {len(texts)} chunks")
        session.add(ds)
        session.commit()
        session.refresh(ds)

    return {"ok": True, "dataset": dataset, "chunks_indexed": len(texts), "dataset_version": version}


@app.get("/health")
def health():
    return {"status": "ok", "provider": "openai"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)