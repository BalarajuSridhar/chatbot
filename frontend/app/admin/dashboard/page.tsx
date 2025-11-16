"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type AskResponse = { answer: string };
type STTResponse = { text: string };
type MsgRole = "user" | "assistant" | "system";
type Message = { id: string; role: MsgRole; text: string; feedback?: 'good' | 'bad' };
type LangCode = "en" | "te" | "ta" | "hi";

// Logs types
type LogRow = {
  id: number;
  question: string;
  answer?: string;
  dataset?: string | null;
  language?: string | null;
  created_at: string;
  feedback?: 'good' | 'bad';
};

// File management types
type FileInfo = {
  filename: string;
  dataset: string;
  chunk_count: number;
  has_ocr: boolean;
  admin_username?: string;
};

// Minimal Web Speech types so TS doesn't require lib.dom
type SRConstructor = new () => SR;

type SR = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start(): void;
  stop(): void;
  onresult: ((e: SREvent) => void) | null;
  onerror: ((e: Event) => void) | null;
  onend: (() => void) | null;
};

type SREvent = {
  resultIndex: number;
  results: ArrayLike<SRResult>;
};

type SRResult = {
  isFinal: boolean;
  0: { transcript: string };
};

// --- Auth helpers ---
function verifyAdminToken(token: string | null): boolean {
  if (!token) return false;
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return false;
    const payload = JSON.parse(atob(parts[1]));
    const now = Date.now() / 1000;
    if (payload.exp && payload.exp < now) return false;
    return true;
  } catch (e) {
    console.warn("token verify error", e);
    return false;
  }
}

function getAuthHeaders(): HeadersInit {
  const token = typeof window !== "undefined" ? localStorage.getItem("admin_token") : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export default function AdminDashboard() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<FileInfo[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: "system",
      text: "Upload PDFs, click 'Upload & Index', then ask questions. Use the mic to dictate.",
    },
  ]);

  const [language, setLanguage] = useState<LangCode>("en");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [useBrowserSTT, setUseBrowserSTT] = useState(true);
  const [recording, setRecording] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  // Logs state
  const [tab, setTab] = useState<"upload" | "chat" | "logs" | "files">("upload");
  const [logs, setLogs] = useState<LogRow[]>([]);
  const [selectedLog, setSelectedLog] = useState<LogRow | null>(null);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [loadingFiles, setLoadingFiles] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const browserRecRef = useRef<SR | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  const router = useRouter();
  const [adminToken, setAdminToken] = useState<string | null>(() =>
    typeof window !== "undefined" ? localStorage.getItem("admin_token") : null
  );

  // verify token on mount
  useEffect(() => {
    const token = typeof window !== "undefined" ? localStorage.getItem("admin_token") : null;
    if (!verifyAdminToken(token)) {
      console.log("Token invalid or missing — redirecting to login");
      localStorage.removeItem("admin_token");
      setAdminToken(null);
      router.push("/admin/login");
    } else {
      setAdminToken(token);
    }
  }, [router]);

  // fetch datasets once
  useEffect(() => {
    let mounted = true;
    fetch(`${API_BASE}/datasets`)
      .then((r) => r.json())
      .then((j) => {
        if (!mounted) return;
        setDatasets(j.datasets || []);
      })
      .catch(() => setDatasets([]));
    return () => {
      mounted = false;
    };
  }, []);

  // ---------- File handlers ----------
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const dropped = Array.from(e.dataTransfer.files || []).filter(isPDF);
    if (dropped.length) setFiles((prev) => dedupe([...prev, ...dropped]));
  };

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(e.target.files || []).filter(isPDF);
    if (picked.length) setFiles((prev) => dedupe([...prev, ...picked]));
  };

  const removeFile = (name: string) => setFiles((prev) => prev.filter((f) => f.name !== name));
  const clearFiles = () => setFiles([]);

  // ---------- Upload API ----------
  const handleIngest = async () => {
    if (files.length === 0) return toast("Please choose at least one PDF.");
    if (!dataset.trim()) return toast("Please enter a dataset name.");

    const token = localStorage.getItem("admin_token");
    if (!token || !verifyAdminToken(token)) {
      toast("Please login as admin first.");
      router.push("/admin/login");
      return;
    }

    const fd = new FormData();
    fd.append("dataset", dataset);
    files.forEach((f) => fd.append("files", f));

    try {
      setIngesting(true);
      setUploadMessage("");

      const res = await fetch(`${API_BASE}/admin/upload`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: fd,
      });

      if (res.status === 401) {
        toast("Authentication failed. Please login again.");
        localStorage.removeItem("admin_token");
        setAdminToken(null);
        router.push("/admin/login");
        return;
      }

      if (!res.ok) {
        const errorText = await safeText(res);
        setUploadMessage(`Upload failed: ${errorText}`);
        toast("Upload failed");
        return;
      }

      const data = await res.json();
      const message = `Uploaded to ${data.dataset} — ${data.chunks_indexed} chunks indexed`;
      setUploadMessage(message);
      toast(message);

      // Refresh datasets list and files
      fetchDatasets();
      fetchUploadedFiles();
      setFiles([]);
    } catch (err: unknown) {
      const errorMsg = errorMessage(err) || "Failed to upload PDFs";
      setUploadMessage(`Error: ${errorMsg}`);
      toast(errorMsg);
    } finally {
      setIngesting(false);
    }
  };

  // ---------- Ask API ----------
  const handleAsk = async () => {
    const q = question.trim();
    if (!q) return toast("Type or dictate a question first.");
    setMessages((prev) => [...prev, { id: uid(), role: "user", text: q }]);
    setQuestion("");
    try {
      setAsking(true);
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: 4, language, dataset: dataset || undefined }),
      });
      if (!res.ok) throw new Error(await safeText(res));
      const data = (await res.json()) as AskResponse;
      setMessages((prev) => [...prev, { id: uid(), role: "assistant", text: data.answer || "(no answer)" }]);
    } catch (err: unknown) {
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "assistant", text: `Error: ${errorMessage(err) || "Failed to ask"}` },
      ]);
    } finally {
      setAsking(false);
    }
  };

  // ---------- TTS ----------
  async function playTTS(text: string) {
    try {
      if (!text) return;
      const res = await fetch(`${API_BASE}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
      });
      if (!res.ok) throw new Error(await safeText(res));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      a.onended = () => URL.revokeObjectURL(url);
      await a.play();
    } catch (e) {
      toast("TTS failed");
      console.error(e);
    }
  }

  function stopTTS() {
    try {
      // Stop any playing audio
      const audioElements = document.querySelectorAll('audio');
      audioElements.forEach(audio => audio.pause());
    } catch {}
  }

  // ---------- Feedback ----------
  async function handleFeedback(messageId: string, feedback: 'good' | 'bad') {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, feedback } : msg
    ));
    
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message_id: messageId, 
          feedback,
          timestamp: new Date().toISOString()
        }),
      });
      toast(`Feedback ${feedback === 'good' ? '👍' : '👎'} recorded`);
    } catch (e) {
      console.error("Failed to send feedback:", e);
    }
  }

  // ---------- File Management ----------
  async function fetchUploadedFiles() {
    setLoadingFiles(true);
    try {
      const token = localStorage.getItem("admin_token");
      const res = await fetch(`${API_BASE}/admin/files`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (res.status === 401) {
        localStorage.removeItem("admin_token");
        router.push("/admin/login");
        return;
      }
      if (!res.ok) {
        console.error("Failed to fetch files:", await res.text());
        setUploadedFiles([]);
        return;
      }
      const data = await res.json();
      setUploadedFiles(data.files || []);
    } catch (e) {
      console.error("fetchUploadedFiles error:", e);
      setUploadedFiles([]);
    } finally {
      setLoadingFiles(false);
    }
  }

  async function deleteFile(filename: string) {
    if (!confirm(`Are you sure you want to delete "${filename}"? This will remove all associated chunks from the vector database.`)) {
      return;
    }
    
    try {
      const token = localStorage.getItem("admin_token");
      const res = await fetch(`${API_BASE}/admin/delete/file?filename=${encodeURIComponent(filename)}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }
      
      const result = await res.json();
      
      toast(`✅ Deleted "${filename}" (${result.chunks_removed} chunks removed)`);
      fetchUploadedFiles(); // Refresh the list
      fetchDatasets(); // Refresh datasets in case this was the last file in a dataset
    } catch (e: any) {
      toast(`❌ Failed to delete file: ${e.message}`);
      console.error(e);
    }
  }

  async function deleteDataset(datasetName: string) {
    if (!confirm(`Are you sure you want to delete the entire dataset "${datasetName}"? This will remove ALL files and chunks from this dataset.`)) {
      return;
    }
    
    try {
      const token = localStorage.getItem("admin_token");
      const res = await fetch(`${API_BASE}/admin/delete/dataset?dataset=${encodeURIComponent(datasetName)}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }
      
      const result = await res.json();
      
      toast(`✅ Deleted dataset "${datasetName}" (${result.chunks_removed} chunks from ${result.files_affected.length} files)`);
      fetchUploadedFiles(); // Refresh the list
      fetchDatasets(); // Refresh datasets list
    } catch (e: any) {
      toast(`❌ Failed to delete dataset: ${e.message}`);
      console.error(e);
    }
  }

  async function fetchDatasets() {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      if (res.ok) {
        const data = await res.json();
        setDatasets(data.datasets || []);
      }
    } catch (err) {
      console.error("Failed to refresh datasets:", err);
    }
  }

  // ---------- Logs ----------
  async function fetchLogs() {
    setLoadingLogs(true);
    try {
      const token = localStorage.getItem("admin_token");
      const res = await fetch(`${API_BASE}/admin/logs?limit=200`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (res.status === 401) {
        localStorage.removeItem("admin_token");
        router.push("/admin/login");
        return;
      }
      if (!res.ok) {
        console.error("Failed to fetch logs:", await res.text());
        setLogs([]);
        return;
      }
      const j = await res.json();
      setLogs(j.logs || []);
      if ((j.logs || []).length > 0) {
        setSelectedLog((j.logs || [])[0]);
      } else {
        setSelectedLog(null);
      }
    } catch (e) {
      console.error("fetchLogs error:", e);
      setLogs([]);
    } finally {
      setLoadingLogs(false);
    }
  }

  function pickLog(l: LogRow) {
    setSelectedLog(l);
    setTab("logs");
  }

  // ---------- STT helpers ----------
  function sttLocale(code: LangCode) {
    return code === "en" ? "en-US" : code === "hi" ? "hi-IN" : code === "te" ? "te-IN" : code === "ta" ? "ta-IN" : "en-US";
  }

  function getSpeechRecognitionCtor(): SRConstructor | null {
    const w = window as any;
    return w.SpeechRecognition || w.webkitSpeechRecognition || null;
  }

  function startBrowserSTT() {
    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) {
      toast("Browser STT not supported. Falling back to server.");
      setUseBrowserSTT(false);
      startServerRecording();
      return;
    }
    try {
      const rec = new Ctor();
      browserRecRef.current = rec;
      rec.lang = sttLocale(language);
      rec.interimResults = true;
      rec.continuous = true;

      let finalText = "";
      rec.onresult = (e: SREvent) => {
        let interim = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const seg = (e.results[i][0] as { transcript: string }).transcript;
          if ((e.results[i] as SRResult).isFinal) finalText += seg + " ";
          else interim += seg;
        }
        setQuestion(finalText || interim);
      };
      rec.onerror = () => {
        toast("Browser STT error. Falling back to server.");
        try {
          rec.stop();
        } catch {}
        startServerRecording();
      };
      rec.onend = () => {
        setRecording(false);
        const trimmed = finalText.trim();
        if (trimmed) {
          setQuestion((prev) => (prev ? `${prev} ${trimmed}` : trimmed));
          toast("Transcribed (browser)");
        }
      };

      rec.start();
      setRecording(true);
    } catch (e) {
      toast("Browser STT init failed. Falling back to server.");
      startServerRecording();
    }
  }

  async function startServerRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      toast("Mic not available.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const supportsWebm = typeof MediaRecorder !== "undefined" && (MediaRecorder as any).isTypeSupported?.("audio/webm");
      const mr = new MediaRecorder(stream, supportsWebm ? { mimeType: "audio/webm" } : undefined);
      (mr as any).stream = stream;
      chunksRef.current = [];
      mr.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = async () => {
        const type = supportsWebm ? "audio/webm" : "audio/wav";
        const blob = new Blob(chunksRef.current, { type });
        try {
          await sendToSTT(blob);
        } finally {
          try {
            stream.getTracks().forEach((t) => t.stop());
          } catch {}
        }
      };

      mediaRecorderRef.current = mr;
      mr.start();
      setRecording(true);
    } catch (e) {
      toast("Microphone permission denied.");
    }
  }

  function stopBrowserSTT() {
    const rec = browserRecRef.current;
    if (rec) {
      try {
        rec.onresult = null;
        rec.onend = null;
        rec.onerror = null;
        rec.stop();
      } catch {}
      browserRecRef.current = null;
    }
    setRecording(false);
  }

  function stopServerRecording() {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== "inactive") {
      try {
        mr.stop();
      } catch {}
    }
    setRecording(false);
  }

  async function sendToSTT(blob: Blob) {
    const fd = new FormData();
    fd.append("audio", new File([blob], "query.webm", { type: blob.type || "audio/webm" }));
    fd.append("lang", language);
    try {
      const res = await fetch(`${API_BASE}/stt`, { method: "POST", body: fd });
      if (res.status === 429) {
        toast("Server STT quota exceeded. Use Browser STT.");
        setUseBrowserSTT(true);
        return;
      }
      if (!res.ok) throw new Error(await safeText(res));
      const data = (await res.json()) as STTResponse;
      const t = (data?.text || "").trim();
      if (t) {
        setQuestion((prev) => (prev ? `${prev} ${t}` : t));
        toast("Transcribed (server)");
      } else {
        toast("No speech detected.");
      }
    } catch (e) {
      toast(errorMessage(e) || "Transcription failed");
    }
  }

  const disabled = ingesting || asking;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex flex-col overflow-hidden">
      {/* Header - MSME Branded Design */}
      <header className="sticky top-0 z-50 backdrop-blur-lg bg-white/90 border-b border-white/20 shadow-sm">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* MSME Logo */}
            <div className="flex items-center gap-3">
              <img 
                src="/images/msme-logo.png" 
                alt="MSME Logo"
                className="h-12 w-auto"
              />
              <div className="flex flex-col">
                <div className="text-lg font-bold text-gray-900 uppercase tracking-wide leading-tight">
                  Micro, Small & Medium Enterprises
                </div>
                <div className="text-sm text-gray-600 font-medium leading-tight">
                  Ministry of MSME, Government of India
                </div>
              </div>
            </div>
            
            <nav className="ml-6 flex items-center gap-1">
              {[
                { id: "upload", label: "Upload" },
                { id: "chat", label: "Chat" },
                { id: "logs", label: "User Logs" },
                { id: "files", label: "File Management" }
              ].map(({ id, label }) => (
                <button 
                  key={id}
                  onClick={() => setTab(id as any)}
                  className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                    tab === id 
                      ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg' 
                      : 'text-gray-600 hover:bg-white/50 hover:text-gray-900 backdrop-blur-sm'
                  }`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>

          <div className="flex items-center gap-3">
            <button 
              onClick={() => router.push("/")} 
              className="px-4 py-2 rounded-xl border border-blue-300 bg-white text-gray-700 hover:bg-blue-50 transition-all duration-200 text-sm font-medium"
            >
              Main Chat
            </button>
            <button
              onClick={() => {
                localStorage.removeItem("admin_token");
                setAdminToken(null);
                toast("Logged out");
                router.push("/");
              }}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-red-500 to-red-600 text-white hover:from-red-600 hover:to-red-700 transition-all duration-200 text-sm font-medium shadow-lg"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 mx-auto max-w-7xl w-full px-6 py-6">
        {tab === "upload" && (
          <section className="grid gap-6 lg:grid-cols-2">
            {/* Upload Section */}
            <div className="rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-6">
              <h2 className="text-xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent mb-4">
                Upload & Index PDFs
              </h2>
              <p className="text-sm text-gray-600 mb-6">Drop files below or click to select. Only PDFs are accepted.</p>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-3">Dataset Name</label>
                <input
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  placeholder="Enter dataset name..."
                  className="w-full border border-gray-200 rounded-2xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/50 backdrop-blur-sm font-medium"
                />
              </div>

              <div
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }}
                onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); }}
                onDrop={onDrop}
                className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200 ${
                  isDragging 
                    ? "bg-blue-50 border-blue-300" 
                    : "bg-white/50 border-gray-200 hover:border-blue-300"
                }`}>
                <input id="file-input" type="file" accept="application/pdf" multiple onChange={onPick} className="hidden" />
                <label htmlFor="file-input" className="cursor-pointer inline-flex flex-col items-center justify-center gap-3">
                  <CloudIcon />
                  <span className="font-medium text-gray-700">Click to choose PDFs</span>
                  <span className="text-sm text-gray-500">or drag & drop here</span>
                </label>
              </div>

              {files.length > 0 && (
                <div className="mt-6 max-h-48 overflow-auto rounded-2xl border border-gray-200 bg-white/80 backdrop-blur-sm">
                  <ul className="divide-y divide-gray-100">
                    {files.map((f) => (
                      <li key={f.name} className="flex items-center justify-between px-4 py-3 text-sm">
                        <span className="truncate max-w-[70%] font-medium text-gray-700" title={f.name}>{f.name}</span>
                        <button 
                          onClick={() => removeFile(f.name)} 
                          className="text-red-500 hover:text-red-700 text-xs font-medium transition-colors"
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="mt-6 flex items-center gap-3">
                <button
                  onClick={handleIngest}
                  disabled={files.length === 0 || !dataset.trim() || disabled}
                  className="px-6 py-3 rounded-2xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 transition-all duration-200 font-semibold shadow-lg hover:shadow-xl disabled:shadow-none"
                >
                  {ingesting ? (
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Uploading…
                    </div>
                  ) : (
                    "Upload & Index"
                  )}
                </button>
                {files.length > 0 && (
                  <button 
                    onClick={clearFiles} 
                    disabled={disabled}
                    className="px-4 py-3 rounded-2xl border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 font-medium"
                  >
                    Clear All
                  </button>
                )}
              </div>

              {uploadMessage && (
                <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-2xl text-sm text-blue-700">
                  {uploadMessage}
                </div>
              )}
            </div>

            {/* Datasets List */}
            <div className="rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-6">
              <h2 className="text-xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent mb-4">
                Available Datasets
              </h2>
              {datasets.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-8">No datasets yet. Upload PDFs to create datasets.</p>
              ) : (
                <ul className="space-y-3">
                  {datasets.map((ds) => (
                    <li key={ds} className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl border border-blue-100">
                      <span className="font-semibold text-gray-800">{ds}</span>
                      <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-medium">Active</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        {tab === "chat" && (
          <section className="grid gap-6">
            {/* Chat Display Area */}
            <div className="rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-6">
              <h2 className="text-xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent mb-4">
                Admin Chat Interface
              </h2>
              
              <div className="rounded-2xl border border-gray-200 bg-white/50 backdrop-blur-sm p-4 mb-6">
                <div 
                  ref={messagesContainerRef}
                  className="h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-blue-200 scrollbar-track-blue-50 hover:scrollbar-thumb-blue-300"
                >
                  <div className="space-y-4 p-2">
                    {messages.map((m, index) => (
                      <div 
                        key={m.id} 
                        className={`flex ${m.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
                        style={{ animationDelay: `${index * 100}ms` }}
                      >
                        <div
                          className={`group max-w-[85%] px-4 py-3 leading-relaxed whitespace-pre-wrap rounded-2xl shadow-lg backdrop-blur-sm border transform transition-all duration-300 ${
                            m.role === "user" 
                              ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-br-md" 
                              : "bg-white/90 text-gray-800 rounded-bl-md border-gray-100"
                          }`}
                        >
                          <div className="text-sm">{m.text}</div>
                          
                          {/* Assistant Actions with Feedback */}
                          {m.role === "assistant" && (
                            <div className="mt-3 flex flex-wrap items-center gap-2">
                              <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                <button 
                                  onClick={() => playTTS(m.text)} 
                                  className="flex items-center gap-1 px-2 py-1 rounded-full border border-blue-200 bg-white text-gray-700 hover:bg-blue-50 transition-all duration-200 text-xs font-medium"
                                >
                                  <span>🔊</span>
                                  Speak
                                </button>
                                <button 
                                  onClick={() => stopTTS()} 
                                  className="flex items-center gap-1 px-2 py-1 rounded-full border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 text-xs font-medium"
                                >
                                  <span>■</span>
                                  Stop
                                </button>
                                <button 
                                  onClick={() => navigator.clipboard.writeText(m.text)}
                                  className="flex items-center gap-1 px-2 py-1 rounded-full border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 text-xs font-medium"
                                >
                                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                  </svg>
                                  Copy
                                </button>
                              </div>

                              {/* Feedback Buttons */}
                              <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                <div className="h-3 w-px bg-gray-300 mx-2"></div>
                                <span className="text-xs text-gray-500 font-medium mr-1">Helpful?</span>
                                <button
                                  onClick={() => handleFeedback(m.id, 'good')}
                                  className={`p-1.5 rounded-full transition-all duration-200 ${
                                    m.feedback === 'good' 
                                      ? 'bg-green-100 text-green-600 border border-green-200 shadow-sm' 
                                      : 'bg-white text-gray-400 hover:text-green-500 hover:bg-green-50 border border-gray-200 hover:border-green-200'
                                  }`}
                                  title="This answer was helpful"
                                >
                                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                                  </svg>
                                </button>
                                <button
                                  onClick={() => handleFeedback(m.id, 'bad')}
                                  className={`p-1.5 rounded-full transition-all duration-200 ${
                                    m.feedback === 'bad' 
                                      ? 'bg-red-100 text-red-600 border border-red-200 shadow-sm' 
                                      : 'bg-white text-gray-400 hover:text-red-500 hover:bg-red-50 border border-gray-200 hover:border-red-200'
                                  }`}
                                  title="This answer was not helpful"
                                >
                                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    
                    {asking && (
                      <div className="flex justify-start">
                        <div className="max-w-[85%] px-4 py-3 bg-white/90 backdrop-blur-sm rounded-2xl rounded-bl-md shadow-lg border border-gray-100">
                          <div className="flex items-center gap-3 text-gray-600">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                            </div>
                            <div className="text-sm font-medium">Analyzing your question...</div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={bottomRef} />
                  </div>
                </div>
              </div>

              {/* Input Area */}
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-4 text-sm">
                  <label className="flex items-center gap-2 font-medium text-gray-700">
                    <span>Answer language:</span>
                    <select 
                      value={language} 
                      onChange={(e) => setLanguage(e.target.value as LangCode)} 
                      className="border border-gray-200 rounded-xl px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium"
                    >
                      <option value="en">🇺🇸 English</option>
                      <option value="te">🇮🇳 Telugu</option>
                      <option value="ta">🇮🇳 Tamil</option>
                      <option value="hi">🇮🇳 Hindi</option>
                    </select>
                  </label>

                  <label className="flex items-center gap-2 font-medium text-gray-700">
                    <span>Dataset:</span>
                    <select 
                      value={dataset} 
                      onChange={(e) => setDataset(e.target.value)} 
                      className="border border-gray-200 rounded-xl px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium"
                    >
                      <option value="">All Documents</option>
                      {datasets.map((d) => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                  </label>

                  <label className="flex items-center gap-2 font-medium text-gray-700">
                    <input 
                      type="checkbox" 
                      checked={useBrowserSTT} 
                      onChange={(e) => setUseBrowserSTT(e.target.checked)} 
                      className="rounded text-blue-600 focus:ring-blue-500 w-4 h-4"
                    />
                    <span>Browser Speech Recognition</span>
                  </label>
                </div>

                <div className="flex items-center gap-3">
                  <div className="flex-1 relative">
                    <input
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); } }}
                      placeholder="Ask about your documents... (Press Enter to send)"
                      className="w-full border border-gray-200 rounded-2xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/50 backdrop-blur-sm font-medium placeholder-gray-400 pr-32"
                    />
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
                      <button
                        onClick={() => { if (recording) { useBrowserSTT ? stopBrowserSTT() : stopServerRecording(); } else { useBrowserSTT ? startBrowserSTT() : startServerRecording(); } }}
                        title={recording ? "Stop recording" : "Start recording"}
                        className={`p-2.5 rounded-xl border transition-all duration-200 ${
                          recording 
                            ? "border-red-500 bg-red-50 text-red-600 hover:bg-red-100 shadow-sm animate-pulse" 
                            : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50 hover:shadow-sm"
                        }`}
                        disabled={asking}
                      >
                        {recording ? (
                          <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                        ) : (
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                          </svg>
                        )}
                      </button>
                    </div>
                  </div>

                  <button 
                    onClick={handleAsk} 
                    disabled={asking || !question.trim()} 
                    className="px-6 py-3 rounded-2xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 transition-all duration-200 font-semibold shadow-lg hover:shadow-xl disabled:shadow-none"
                  >
                    {asking ? (
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        Thinking…
                      </div>
                    ) : (
                      "Ask Question"
                    )}
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}

        {tab === "logs" && (
          <section className="grid grid-cols-3 gap-6">
            <div className="col-span-1 rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-6 max-h-[70vh] overflow-auto">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-gray-800">Recent User Logs</h3>
                <button 
                  onClick={fetchLogs} 
                  className="px-3 py-1.5 rounded-xl border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 text-sm font-medium"
                >
                  Refresh
                </button>
              </div>

              {loadingLogs ? (
                <div className="flex items-center justify-center py-8">
                  <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                </div>
              ) : logs.length === 0 ? (
                <div className="text-sm text-gray-500 text-center py-8">No logs yet.</div>
              ) : (
                <ul className="space-y-2">
                  {logs.map((l) => (
                    <li 
                      key={l.id} 
                      className={`p-3 rounded-2xl border cursor-pointer transition-all duration-200 hover:shadow-md ${
                        selectedLog?.id === l.id 
                          ? 'bg-blue-50 border-blue-200 shadow-sm' 
                          : 'bg-white/50 border-gray-200 hover:bg-white'
                      }`} 
                      onClick={() => pickLog(l)}
                    >
                      <div className="text-xs text-gray-500 mb-1">{new Date(l.created_at).toLocaleString()}</div>
                      <div className="text-sm font-medium text-gray-800 truncate mb-1">{l.question}</div>
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>{l.dataset ?? "(All datasets)"} • {l.language ?? "en"}</span>
                        {l.feedback && (
                          <span className={`px-1.5 py-0.5 rounded-full text-xs ${
                            l.feedback === 'good' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                          }`}>
                            {l.feedback === 'good' ? '👍' : '👎'}
                          </span>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="col-span-2 rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-6 max-h-[70vh] overflow-auto">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold text-gray-800">Log Detail</h3>
                <div className="text-sm text-gray-500">
                  {selectedLog ? new Date(selectedLog.created_at).toLocaleString() : ""}
                </div>
              </div>

              {!selectedLog ? (
                <div className="text-sm text-gray-500 text-center py-12">Select a log to view details</div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <div className="text-sm font-medium text-gray-700 mb-3">Question</div>
                    <div className="p-4 rounded-2xl bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 whitespace-pre-wrap text-gray-800">
                      {selectedLog.question}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-700 mb-3">Answer</div>
                    <div className="p-4 rounded-2xl bg-white border border-gray-200 whitespace-pre-wrap text-gray-800">
                      {selectedLog.answer ?? "(No answer recorded)"}
                    </div>
                  </div>
                  <div className="flex gap-6 text-sm text-gray-600">
                    <div>
                      <span className="font-medium">Dataset:</span> {selectedLog.dataset ?? "(All)"}
                    </div>
                    <div>
                      <span className="font-medium">Language:</span> {selectedLog.language ?? "en"}
                    </div>
                    {selectedLog.feedback && (
                      <div>
                        <span className="font-medium">Feedback:</span> 
                        <span className={`ml-2 px-2 py-1 rounded-full text-xs ${
                          selectedLog.feedback === 'good' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>
                          {selectedLog.feedback === 'good' ? 'Helpful 👍' : 'Not Helpful 👎'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {tab === "files" && (
          <section className="rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                Manage Uploaded Files
              </h2>
              <div className="flex gap-3">
                <button 
                  onClick={fetchUploadedFiles} 
                  className="px-4 py-2 rounded-xl border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 font-medium text-sm"
                >
                  Refresh
                </button>
              </div>
            </div>

            {loadingFiles ? (
              <div className="flex items-center justify-center py-8">
                <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              </div>
            ) : uploadedFiles.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-8">No files uploaded yet.</p>
            ) : (
              <div className="space-y-4 max-h-96 overflow-auto">
                {uploadedFiles.map((file) => (
                  <div key={`${file.filename}-${file.dataset}`} className="flex items-center justify-between p-4 border border-gray-200 rounded-2xl bg-white/50 backdrop-blur-sm">
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-gray-800 truncate" title={file.filename}>{file.filename}</div>
                      <div className="text-sm text-gray-600 flex flex-wrap gap-3 mt-1">
                        <span>Dataset: <span className="font-medium">{file.dataset}</span></span>
                        <span>•</span>
                        <span>Chunks: <span className="font-medium">{file.chunk_count}</span></span>
                        {file.has_ocr && (
                          <>
                            <span>•</span>
                            <span className="text-amber-600 font-medium">OCR Used</span>
                          </>
                        )}
                        {file.admin_username && (
                          <>
                            <span>•</span>
                            <span>Uploaded by: <span className="font-medium">{file.admin_username}</span></span>
                          </>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => deleteFile(file.filename)}
                      className="ml-4 px-4 py-2 text-sm bg-red-100 text-red-700 rounded-xl hover:bg-red-200 transition-all duration-200 font-medium whitespace-nowrap"
                      title={`Delete ${file.filename}`}
                    >
                      Delete File
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Dataset Management Section */}
            <div className="mt-8 pt-6 border-t border-gray-200">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Dataset Management</h3>
              <div className="space-y-3">
                {datasets.map((datasetName) => {
                  const datasetFiles = uploadedFiles.filter(f => f.dataset === datasetName);
                  const totalChunks = datasetFiles.reduce((sum, file) => sum + file.chunk_count, 0);
                  
                  return (
                    <div key={datasetName} className="flex items-center justify-between p-4 border border-gray-200 rounded-2xl bg-gradient-to-r from-blue-50 to-indigo-50">
                      <div className="flex-1">
                        <div className="font-semibold text-gray-800">{datasetName}</div>
                        <div className="text-sm text-gray-600">
                          {datasetFiles.length} files • {totalChunks} total chunks
                        </div>
                      </div>
                      <button
                        onClick={() => deleteDataset(datasetName)}
                        className="ml-4 px-4 py-2 text-sm bg-red-100 text-red-700 rounded-xl hover:bg-red-200 transition-all duration-200 font-medium"
                        disabled={datasetFiles.length === 0}
                        title={datasetFiles.length === 0 ? "No files in this dataset" : `Delete entire ${datasetName} dataset`}
                      >
                        Delete Dataset
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>
        )}
      </main>

      <Toaster />
    </div>
  );
}

// ---------- UI PRIMITIVES ----------
function CloudIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path 
        d="M7 18H17C19.2091 18 21 16.2091 21 14C21 11.9848 19.5223 10.3184 17.5894 10.0458C16.9942 7.74078 14.9289 6 12.5 6C10.2387 6 8.30734 7.4246 7.65085 9.50138C5.61892 9.68047 4 11.3838 4 13.4375C4 15.6228 5.79086 17.5 8 17.5H7Z" 
        stroke="currentColor" 
        strokeWidth="1.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
        className="text-blue-500"
      />
    </svg>
  );
}

// ---------- Toasts ----------
function Toaster() {
  const [items, setItems] = useState<{ id: string; text: string }[]>([]);
  useEffect(() => {
    const onToast = (e: Event) => {
      const ce = e as CustomEvent<string>;
      const text = ce.detail;
      const id = uid();
      setItems((prev) => [...prev, { id, text }]);
      setTimeout(() => setItems((prev) => prev.filter((i) => i.id !== id)), 3500);
    };
    window.addEventListener("app:toast", onToast as EventListener);
    return () => window.removeEventListener("app:toast", onToast as EventListener);
  }, []);
  return (
    <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50">
      {items.map((i) => (
        <div key={i.id} className="px-4 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg backdrop-blur-sm border border-white/20">
          {i.text}
        </div>
      ))}
    </div>
  );
}

function toast(text: string) {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent<string>("app:toast", { detail: text }));
  }
}

// ---------- Helpers ----------
function uid() {
  const g = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function" ? (crypto as any).randomUUID() : undefined;
  return g || `id_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}
function dedupe(list: File[]) {
  const seen = new Set<string>();
  const out: File[] = [];
  for (const f of list) {
    const key = `${f.name}-${f.size}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(f);
    }
  }
  return out;
}
function isPDF(f: File) {
  return f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
}
async function safeText(res: Response) {
  try {
    return await res.text();
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}
function errorMessage(e: unknown): string | undefined {
  if (typeof e === "string") return e;
  if (e && typeof e === "object" && "message" in e) {
    const m = (e as { message?: unknown }).message;
    if (typeof m === "string") return m;
  }
  return undefined;
}