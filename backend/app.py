import os
import io
import uuid
import logging
from typing import List
from openai import RateLimitError 

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
from openai import OpenAI

# -----------------------------------------------------------------------------
# Env & Config
# -----------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

# -----------------------------------------------------------------------------
# App & Middleware
# -----------------------------------------------------------------------------
app = FastAPI(title="PDF Q&A Chatbot (OpenAI + Whisper)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Services: OpenAI, Embeddings, Vector DB
# -----------------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Disable Chroma telemetry noise
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection("pdf_docs")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def read_pdf(file_bytes: bytes) -> str:
    """Extract selectable text from a PDF. (No OCR here.)"""
    reader = PdfReader(io.BytesIO(file_bytes))
    out: List[str] = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            # ignore pages that fail to extract
            pass
    return "\n".join(out)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Robust chunker with safe step to avoid infinite loops.
    Collapses whitespace and advances by (chunk_size - overlap).
    """
    text = " ".join((text or "").split()).strip()
    if not text:
        return []

    if chunk_size <= 0:
        chunk_size = 1000
    if overlap < 0:
        overlap = 0
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

def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = _embedder.encode(texts, normalize_embeddings=True)
    return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join([f"[Source {i+1}]\n{c}" for i, c in enumerate(contexts)])
    return (
        "You are a helpful assistant. Use only the provided context to answer. "
        "If the answer isn't in the context, say you don't know. "
        "Reply with ONLY the final answer text — no preface, no citations, no extra lines.\n\n"
        f"Question: {question}\n\nContext:\n{context_block}"
    )

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = 4

class IngestResponse(BaseModel):
    documents_added: int
    chunks_indexed: int

class AskResponse(BaseModel):
    answer: str

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF supported: {f.filename}")
        data = await f.read()
        content = read_pdf(data)
        if not content.strip():
            # likely scanned PDF with no selectable text (consider OCR later)
            continue
        chunks = chunk_text(content)
        if not chunks:
            continue
        texts.extend(chunks)
        metadatas.extend([{"filename": f.filename}] * len(chunks))
        ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])

    if not texts:
        raise HTTPException(status_code=400, detail="No extractable text found in PDFs")

    vectors = embed_texts(texts)
    collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)

    return IngestResponse(
        documents_added=len(set(m["filename"] for m in metadatas)),
        chunks_indexed=len(texts),
    )

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    qvec = embed_texts([q])[0]
    results = collection.query(query_embeddings=[qvec], n_results=max(1, req.top_k))
    contexts = results.get("documents", [[]])[0] or []

    if not contexts:
        return AskResponse(answer="I couldn't find relevant content in the indexed PDFs.")

    prompt = build_prompt(q, contexts)

    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content.strip()
    return AskResponse(answer=answer)

logger = logging.getLogger("stt")

@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    allowed = {
        "audio/webm", "audio/ogg", "audio/mpeg", "audio/wav",
        "audio/mp4", "audio/x-m4a", "audio/aac"
    }
    ct = (audio.content_type or "").lower()
    if ct not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {ct}")

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    buf = io.BytesIO(data)
    ext = ".webm" if "webm" in ct else ".ogg" if "ogg" in ct else ".mp3" if "mpeg" in ct else ".wav" if "wav" in ct else ".m4a"
    buf.name = f"recording{ext}"

    try:
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=buf)
        text = (getattr(resp, "text", None) or "").strip()
        return {"text": text}
    except RateLimitError as e:
        logger.exception("Whisper quota exceeded")
        # 429 tells the frontend to use browser STT fallback
        raise HTTPException(status_code=429, detail="OpenAI STT quota exceeded")
    except Exception as e:
        logger.exception("Whisper transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {type(e).__name__}: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "provider": "openai"}
