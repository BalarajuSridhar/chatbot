import os
import io
import uuid
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple file-based storage for demo (replace with real DB for production)
import json
import sqlite3

# Create SQLite database
conn = sqlite3.connect('/tmp/chatbot.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        dataset TEXT,
        language TEXT,
        model_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT,
        feedback_type TEXT,
        question TEXT,
        answer TEXT,
        model_used TEXT,
        timestamp TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google.generativeai not installed")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-this")
JWT_ALGO = "HS256"

# Initialize Gemini if available
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured")
else:
    logger.warning("Gemini API not configured")

# Simple in-memory storage for demo (files uploaded to /tmp)
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store chunks in memory (for demo purposes)
chunks_db = []

# Models
class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    language: Optional[str] = "en"
    dataset: Optional[str] = None
    retrieval_model: Optional[str] = None

class AskResponse(BaseModel):
    answer: str

class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str
    question: Optional[str] = None
    answer: Optional[str] = None
    model_used: Optional[str] = None
    timestamp: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MSME Chatbot API...")
    yield
    # Shutdown
    logger.info("Shutting down...")
    conn.close()

app = FastAPI(title="MSME Chatbot API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def generate_response(question: str, context: str = "") -> str:
    """Generate response using Gemini or fallback"""
    if GEMINI_API_KEY and GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            prompt = f"""You are a helpful assistant for MSME (Micro, Small & Medium Enterprises) in India.
            Answer the following question in a helpful, accurate manner.
            Use the context if provided, otherwise use your knowledge about MSME schemes, registration, and policies.
            
            Context: {context if context else 'No specific context provided'}
            
            Question: {question}
            
            Answer:"""
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return get_fallback_response(question)
    else:
        return get_fallback_response(question)

def get_fallback_response(question: str) -> str:
    """Fallback responses when Gemini is not available"""
    q = question.lower()
    if "register" in q or "udyam" in q:
        return "MSME registration can be done through the Udyam Registration portal (https://udyamregistration.gov.in). The process is free, online, and based on self-declaration. You'll need your Aadhaar number and GSTIN (if applicable)."
    elif "loan" in q or "finance" in q:
        return "MSMEs can access various loan schemes including MUDRA loans (up to ₹10 lakhs), CGTMSE coverage (up to ₹2 crores without collateral), and the Emergency Credit Line Guarantee Scheme (ECLGS). Contact your bank's MSME branch for details."
    elif "subsidy" in q or "scheme" in q or "benefit" in q:
        return "Government offers several subsidies for MSMEs: Credit Linked Capital Subsidy Scheme (CLCSS) for technology upgradation (15% subsidy up to ₹15 lakhs), ISO certification reimbursement (up to ₹1 lakh), and GST composition scheme benefits."
    elif "eligibility" in q:
        return "MSME eligibility is based on investment in plant & machinery and annual turnover. Micro: <₹1 crore investment & <₹5 crore turnover. Small: <₹10 crore investment & <₹50 crore turnover. Medium: <₹50 crore investment & <₹250 crore turnover."
    else:
        return "Thank you for your question about MSMEs. For specific details, please visit the official MSME website or contact your local District Industries Centre (DIC). Is there anything specific about registration, loans, or government schemes you'd like to know?"

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1
        if current_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

# API Endpoints
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "MSME Chatbot API is running",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/ask",
            "/datasets",
            "/admin/login",
            "/admin/upload",
            "/admin/files"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "features": {
            "gemini": GEMINI_API_KEY is not None,
            "embeddings": TRANSFORMERS_AVAILABLE
        }
    }

@app.get("/models")
async def get_models():
    return {
        "models": [
            {
                "id": "gemini",
                "name": "Google Gemini",
                "provider": "Google",
                "description": "Google's advanced AI model for document Q&A",
                "icon": "🌟",
                "available": GEMINI_API_KEY is not None
            }
        ]
    }

@app.get("/datasets")
async def list_datasets():
    """List available datasets"""
    datasets = set()
    for chunk in chunks_db:
        if chunk.get("dataset"):
            datasets.add(chunk["dataset"])
    return {"datasets": list(datasets)}

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Log the question
    cursor.execute(
        "INSERT INTO logs (question, dataset, language, model_used) VALUES (?, ?, ?, ?)",
        (question, req.dataset, req.language, "gemini")
    )
    conn.commit()
    log_id = cursor.lastrowid
    
    # Search for relevant chunks
    relevant_chunks = []
    if req.dataset:
        relevant_chunks = [c["text"] for c in chunks_db if c.get("dataset") == req.dataset]
    else:
        relevant_chunks = [c["text"] for c in chunks_db]
    
    # Get context from relevant chunks
    context = "\n\n".join(relevant_chunks[:3]) if relevant_chunks else ""
    
    # Generate answer
    try:
        answer = generate_response(question, context)
        
        # Update log with answer
        cursor.execute(
            "UPDATE logs SET answer = ? WHERE id = ?",
            (answer, log_id)
        )
        conn.commit()
        
        return AskResponse(answer=answer)
    except Exception as e:
        logger.error(f"Ask error: {e}")
        error_answer = f"Error generating response: {str(e)}"
        cursor.execute(
            "UPDATE logs SET answer = ? WHERE id = ?",
            (error_answer, log_id)
        )
        conn.commit()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for an answer"""
    try:
        cursor.execute(
            """INSERT INTO feedback (message_id, feedback_type, question, answer, model_used, timestamp) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (feedback.message_id, feedback.feedback, feedback.question, 
             feedback.answer, feedback.model_used, feedback.timestamp or datetime.utcnow().isoformat())
        )
        conn.commit()
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.post("/admin/login")
async def admin_login(req: LoginRequest):
    """Admin login"""
    if req.username != ADMIN_USERNAME or req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Simple token (in production, use JWT)
    import jwt
    token = jwt.encode(
        {"sub": req.username, "exp": datetime.utcnow() + timedelta(days=1)},
        JWT_SECRET,
        algorithm=JWT_ALGO
    )
    return {"access_token": token, "token_type": "bearer"}

@app.post("/admin/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    dataset: Optional[str] = Form(None)
):
    """Upload PDF files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed = []
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
        
        content = await file.read()
        text = extract_text_from_pdf(content)
        
        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunks_db.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "filename": file.filename,
                    "dataset": dataset or "default",
                    "chunk_index": i
                })
            processed.append(file.filename)
    
    return {
        "success": True,
        "documents_added": len(processed),
        "chunks_indexed": len([c for c in chunks_db if c.get("filename") in processed]),
        "files": processed,
        "dataset": dataset or "default"
    }

@app.get("/admin/files")
async def get_files():
    """Get list of uploaded files"""
    files_dict = {}
    for chunk in chunks_db:
        key = f"{chunk['filename']}_{chunk['dataset']}"
        if key not in files_dict:
            files_dict[key] = {
                "filename": chunk["filename"],
                "dataset": chunk["dataset"],
                "chunk_count": 0,
                "has_ocr": False
            }
        files_dict[key]["chunk_count"] += 1
    
    return {"files": list(files_dict.values())}

@app.delete("/admin/delete/file")
async def delete_file(filename: str):
    """Delete a file and its chunks"""
    global chunks_db
    original_count = len(chunks_db)
    chunks_db = [c for c in chunks_db if c.get("filename") != filename]
    removed = original_count - len(chunks_db)
    
    return {
        "success": True,
        "message": f"Deleted file {filename}",
        "chunks_removed": removed
    }

@app.delete("/admin/delete/dataset")
async def delete_dataset(dataset: str):
    """Delete an entire dataset"""
    global chunks_db
    original_count = len(chunks_db)
    chunks_db = [c for c in chunks_db if c.get("dataset") != dataset]
    removed = original_count - len(chunks_db)
    
    return {
        "success": True,
        "message": f"Deleted dataset {dataset}",
        "chunks_removed": removed
    }

@app.get("/admin/logs")
async def get_logs(limit: int = 100):
    """Get question logs"""
    cursor.execute(
        "SELECT id, question, answer, dataset, language, model_used, created_at FROM logs ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    
    return {"logs": [
        {
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "dataset": row[3],
            "language": row[4],
            "model_used": row[5],
            "created_at": row[6]
        }
        for row in rows
    ]}

@app.get("/admin/feedback-logs")
async def get_feedback_logs(limit: int = 100, offset: int = 0, feedback_type: Optional[str] = None):
    """Get feedback logs"""
    query = "SELECT id, message_id, feedback_type, question, answer, model_used, timestamp, created_at FROM feedback"
    params = []
    
    if feedback_type and feedback_type != "all":
        query += " WHERE feedback_type = ?"
        params.append(feedback_type)
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Get stats
    cursor.execute("SELECT COUNT(*) FROM feedback")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'good'")
    good = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'bad'")
    bad = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'no_response'")
    no_response = cursor.fetchone()[0]
    
    return {
        "feedback_logs": [
            {
                "id": row[0],
                "message_id": row[1],
                "feedback_type": row[2],
                "question": row[3],
                "answer": row[4],
                "model_used": row[5],
                "timestamp": row[6],
                "created_at": row[7]
            }
            for row in rows
        ],
        "summary": {
            "total": total,
            "good": good,
            "bad": bad,
            "no_response": no_response
        }
    }

@app.post("/tts")
async def text_to_speech(text: str, language: str = "en", speed: float = 1.0):
    """Convert text to speech"""
    try:
        from gtts import gTTS
        import tempfile
        
        tts = gTTS(text=text, lang=language[:2], slow=speed < 1.0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            with open(tmp.name, "rb") as f:
                audio_data = f.read()
            os.unlink(tmp.name)
            
        from fastapi.responses import StreamingResponse
        import io
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail="TTS failed")

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...), lang: str = Form("en")):
    """Convert speech to text"""
    try:
        # For now, return empty (implement with a proper STT service)
        return {"text": ""}
    except Exception as e:
        logger.error(f"STT error: {e}")
        return {"text": ""}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)