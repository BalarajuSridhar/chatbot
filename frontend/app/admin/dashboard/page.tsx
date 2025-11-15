"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type AskResponse = { answer: string };
type STTResponse = { text: string };
type MsgRole = "user" | "assistant" | "system";
type Message = { id: string; role: MsgRole; text: string };
type LangCode = "en" | "te" | "ta" | "hi";

// Logs types
type LogRow = {
  id: number;
  question: string;
  answer?: string;
  dataset?: string | null;
  language?: string | null;
  created_at: string;
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
  const [tab, setTab] = useState<"upload" | "chat" | "logs">("upload");
  const [logs, setLogs] = useState<LogRow[]>([]);
  const [selectedLog, setSelectedLog] = useState<LogRow | null>(null);
  const [loadingLogs, setLoadingLogs] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const browserRecRef = useRef<SR | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

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

      // Refresh datasets list
      try {
        const datasetsRes = await fetch(`${API_BASE}/datasets`);
        if (datasetsRes.ok) {
          const datasetsData = await datasetsRes.json();
          setDatasets(datasetsData.datasets || []);
        }
      } catch (err) {
        console.error("Failed to refresh datasets:", err);
      }
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

  // ---------- Logs ----------
  async function fetchLogs() {
    setLoadingLogs(true);
    try {
      const token = localStorage.getItem("admin_token");
      const res = await fetch(`${API_BASE}/admin/logs?limit=200`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (res.status === 401) {
        // token invalid
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
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur bg-white/70 border-b">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-xl bg-slate-900 text-white grid place-items-center font-bold">Q</div>
            <h1 className="text-xl sm:text-2xl font-bold">Admin Dashboard</h1>
            <nav className="ml-6 flex items-center gap-2">
              <button 
                onClick={() => setTab("upload")} 
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  tab === "upload" ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                Upload
              </button>
              <button 
                onClick={() => setTab("chat")} 
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  tab === "chat" ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                Chat
              </button>
              <button 
                onClick={() => { setTab("logs"); fetchLogs(); }} 
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  tab === "logs" ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                User Logs
              </button>
            </nav>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={() => router.push("/")} className="px-3 py-1 rounded border hover:bg-slate-100 text-sm">Main Chat</button>
            <button
              onClick={() => {
                localStorage.removeItem("admin_token");
                setAdminToken(null);
                toast("Logged out");
                router.push("/");
              }}
              className="px-3 py-1 rounded hover:bg-slate-100 text-sm text-rose-600"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-6xl px-4 py-6">
        {tab === "upload" && (
          <section className="grid gap-6 lg:grid-cols-2">
            {/* Upload Section */}
            <div className="rounded-2xl border bg-white/90 shadow-sm p-5">
              <h2 className="text-lg font-semibold">Upload & Index PDFs</h2>
              <p className="text-sm text-slate-600 mb-4">Drop files below or click to select. Only PDFs are accepted.</p>

              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">Dataset Name</label>
                <input
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  placeholder="Enter dataset name..."
                  className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-slate-300"
                />
              </div>

              <div
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }}
                onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); }}
                onDrop={onDrop}
                className={`border-2 border-dashed rounded-2xl p-6 text-center transition ${isDragging ? "bg-slate-100" : "bg-slate-50"}`}>
                <input id="file-input" type="file" accept="application/pdf" multiple onChange={onPick} className="hidden" />
                <label htmlFor="file-input" className="cursor-pointer inline-flex flex-col items-center justify-center gap-2">
                  <CloudIcon />
                  <span className="font-medium">Click to choose PDFs</span>
                  <span className="text-xs text-slate-500">or drag & drop here</span>
                </label>
              </div>

              {files.length > 0 && (
                <div className="mt-4 max-h-48 overflow-auto rounded-xl border bg-white">
                  <ul className="divide-y">
                    {files.map((f) => (
                      <li key={f.name} className="flex items-center justify-between px-3 py-2 text-sm">
                        <span className="truncate max-w-[70%]" title={f.name}>{f.name}</span>
                        <button onClick={() => removeFile(f.name)} className="text-slate-500 hover:text-red-600 text-xs">Remove</button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="mt-4 flex items-center gap-2">
                <button
                  onClick={handleIngest}
                  disabled={files.length === 0 || !dataset.trim() || disabled}
                  className="px-4 py-2 rounded-xl bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {ingesting ? "Uploading…" : "Upload & Index"}
                </button>
                {files.length > 0 && (
                  <button onClick={clearFiles} disabled={disabled} className="px-3 py-2 rounded-xl border hover:bg-slate-50">Clear</button>
                )}
              </div>

              {uploadMessage && (
                <div className="mt-4 p-3 bg-slate-50 rounded-xl text-sm">{uploadMessage}</div>
              )}
            </div>

            {/* Datasets List */}
            <div className="rounded-2xl border bg-white/90 shadow-sm p-5">
              <h2 className="text-lg font-semibold mb-4">Available Datasets</h2>
              {datasets.length === 0 ? (
                <p className="text-sm text-slate-500">No datasets yet. Upload PDFs to create datasets.</p>
              ) : (
                <ul className="space-y-2">
                  {datasets.map((ds) => (
                    <li key={ds} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                      <span className="font-medium text-sm">{ds}</span>
                      <span className="text-xs text-slate-500">Active</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        {tab === "chat" && (
          <section className="rounded-2xl border bg-white/90 shadow-sm p-5">
            <h2 className="text-lg font-semibold mb-4">Ask Questions</h2>
            <div className="space-y-3 max-h-[60vh] overflow-auto pr-1 mb-4">
              {messages.map((m) => (
                <ChatBubble key={m.id} role={m.role} text={m.text} />
              ))}
              {asking && <TypingBubble />}
              <div ref={bottomRef} />
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-4 text-xs text-slate-600">
                <label className="flex items-center gap-2 select-none">
                  <span>Answer language:</span>
                  <select value={language} onChange={(e) => setLanguage(e.target.value as LangCode)} className="border rounded-lg px-2 py-1 text-sm">
                    <option value="en">English</option>
                    <option value="te">Telugu</option>
                    <option value="ta">Tamil</option>
                    <option value="hi">Hindi</option>
                  </select>
                </label>

                <label className="flex items-center gap-2 select-none">
                  <span>Dataset:</span>
                  <select value={dataset} onChange={(e) => setDataset(e.target.value)} className="border rounded-lg px-2 py-1 text-sm">
                    <option value="">(All)</option>
                    {datasets.map((d) => (
                      <option key={d} value={d}>{d}</option>
                    ))}
                  </select>
                </label>

                <label className="flex items-center gap-2 select-none">
                  <input type="checkbox" checked={useBrowserSTT} onChange={(e) => setUseBrowserSTT(e.target.checked)} />
                  <span>Browser STT</span>
                </label>
              </div>

              <div className="flex items-center gap-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); } }}
                  placeholder="Ask about your documents…"
                  className="flex-1 border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-slate-300"
                />

                <button
                  type="button"
                  onClick={() => { if (recording) { useBrowserSTT ? stopBrowserSTT() : stopServerRecording(); } else { useBrowserSTT ? startBrowserSTT() : startServerRecording(); } }}
                  title={recording ? "Stop recording" : "Start recording"}
                  className={`px-3 py-2 rounded-xl border hover:bg-slate-50 ${recording ? "border-red-500 text-red-600" : ""}`}
                  disabled={asking}
                >
                  {recording ? "Stop" : "Mic"}
                </button>

                <button onClick={handleAsk} disabled={asking || !question.trim()} className="px-4 py-2 rounded-xl bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed">
                  {asking ? "Thinking…" : "Ask"}
                </button>
              </div>
            </div>
          </section>
        )}

        {tab === "logs" && (
          <section className="grid grid-cols-3 gap-6">
            <div className="col-span-1 rounded-2xl border bg-white/90 shadow-sm p-4 max-h-[70vh] overflow-auto">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold">Recent User Logs</h3>
                <button onClick={fetchLogs} className="text-xs px-2 py-1 rounded border hover:bg-slate-50">Refresh</button>
              </div>

              {loadingLogs ? (
                <div className="text-sm text-slate-500">Loading…</div>
              ) : logs.length === 0 ? (
                <div className="text-sm text-slate-500">No logs yet.</div>
              ) : (
                <ul className="divide-y">
                  {logs.map((l) => (
                    <li key={l.id} className="px-2 py-3 hover:bg-slate-50 cursor-pointer transition-colors" onClick={() => pickLog(l)}>
                      <div className="text-xs text-slate-500 mb-1">{new Date(l.created_at).toLocaleString()}</div>
                      <div className="text-sm font-medium truncate mb-1">{l.question}</div>
                      <div className="text-xs text-slate-400 truncate">
                        {l.dataset ?? "(All datasets)"} • {l.language ?? "en"}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="col-span-2 rounded-2xl border bg-white/90 shadow-sm p-4 max-h-[70vh] overflow-auto">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Log Detail</h3>
                <div className="text-xs text-slate-500">
                  {selectedLog ? new Date(selectedLog.created_at).toLocaleString() : ""}
                </div>
              </div>

              {!selectedLog ? (
                <div className="text-sm text-slate-500 text-center py-8">Select a log to view details</div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <div className="text-xs text-slate-500 mb-2 font-medium">Question</div>
                    <div className="p-3 rounded-lg bg-slate-50 whitespace-pre-wrap border">{selectedLog.question}</div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500 mb-2 font-medium">Answer</div>
                    <div className="p-3 rounded-lg bg-white border whitespace-pre-wrap">
                      {selectedLog.answer ?? "(No answer recorded)"}
                    </div>
                  </div>
                  <div className="flex gap-4 text-xs text-slate-500">
                    <div>
                      <span className="font-medium">Dataset:</span> {selectedLog.dataset ?? "(All)"}
                    </div>
                    <div>
                      <span className="font-medium">Language:</span> {selectedLog.language ?? "en"}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      <Toaster />
    </div>
  );
}

// ---------- UI PRIMITIVES ----------
function ChatBubble({ role, text }: { role: MsgRole; text: string }) {
  const isUser = role === "user";
  const isSystem = role === "system";
  const align = isUser ? "items-end" : "items-start";
  const bg = isSystem ? "bg-slate-100" : isUser ? "bg-slate-900 text-white" : "bg-slate-50";
  const radius = isUser ? "rounded-t-2xl rounded-bl-2xl" : "rounded-t-2xl rounded-br-2xl";
  return (
    <div className={`flex ${align}`}>
      <div className={`max-w-[90%] sm:max-w-[80%] ${bg} ${radius} px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap`}>{text}</div>
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="flex items-start">
      <div className="bg-slate-50 rounded-t-2xl rounded-br-2xl px-4 py-3 text-sm">
        <span className="inline-flex gap-1">
          <Dot /> <Dot /> <Dot />
        </span>
      </div>
    </div>
  );
}

function Dot() {
  return <span className="w-2 h-2 inline-block rounded-full bg-slate-400 animate-pulse" />;
}

function CloudIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M7 18H17C19.2091 18 21 16.2091 21 14C21 11.9848 19.5223 10.3184 17.5894 10.0458C16.9942 7.74078 14.9289 6 12.5 6C10.2387 6 8.30734 7.4246 7.65085 9.50138C5.61892 9.68047 4 11.3838 4 13.4375C4 15.6228 5.79086 17.5 8 17.5H7Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
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
        <div key={i.id} className="px-4 py-2 rounded-xl bg-slate-900 text-white shadow">{i.text}</div>
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