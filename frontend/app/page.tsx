"use client";

import { useEffect, useMemo, useRef, useState } from "react";

// Backend base URL (set NEXT_PUBLIC_API_BASE in .env.local to override)
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

// --- Types matching your backend (answer-only) ---
type IngestResponse = {
  documents_added: number;
  chunks_indexed: number;
};

type AskResponse = {
  answer: string;
};

type STTResponse = {
  text: string;
};

type MsgRole = "user" | "assistant" | "system";

type Message = {
  id: string;
  role: MsgRole;
  text: string;
};

// ---- Minimal Web Speech API types (to avoid using `any`) ----
// These are small structural types so we compile without extra libs.

type SpeechRecognitionConstructor = new () => SpeechRecognition;

type SpeechRecognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((e: SpeechRecognitionEvent) => void) | null;
  onerror: ((e: Event) => void) | null;
  onend: (() => void) | null;
};

type SpeechRecognitionEvent = {
  resultIndex: number;
  results: SpeechRecognitionResultList;
};

type SpeechRecognitionResultList = ArrayLike<SpeechRecognitionResult>;

type SpeechRecognitionResult = {
  isFinal: boolean;
  0: { transcript: string };
};

export default function Page() {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: "system",
      text: "Upload one or more PDFs, click ‘Upload & Index’, then ask a question. Use the mic to dictate your question.",
    },
  ]);

  // STT state
  const [recording, setRecording] = useState(false);
  const [useBrowserSTT, setUseBrowserSTT] = useState(true);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const browserRecRef = useRef<SpeechRecognition | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

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

  // ---------- API calls ----------
  const handleIngest = async () => {
    if (files.length === 0) return toast("Please choose at least one PDF.");
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    try {
      setIngesting(true);
      const res = await fetch(`${API_BASE}/ingest`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await safeText(res));
      const data = (await res.json()) as IngestResponse;
      toast(`Indexed ${data.documents_added} file(s), ${data.chunks_indexed} chunks.`);
    } catch (err: unknown) {
      toast(errorMessage(err) || "Failed to ingest PDFs");
    } finally {
      setIngesting(false);
    }
  };

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
        body: JSON.stringify({ question: q, top_k: 4 }),
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

  // ---------- STT helpers ----------
  function getSpeechRecognitionCtor(): SpeechRecognitionConstructor | null {
    const w = window as unknown as {
      SpeechRecognition?: SpeechRecognitionConstructor;
      webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
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
      rec.lang = "en-US"; // change if needed
      rec.interimResults = true;
      rec.continuous = true;

      let finalText = "";
      rec.onresult = (e: SpeechRecognitionEvent) => {
        let interim = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const seg = e.results[i][0].transcript;
          if (e.results[i].isFinal) finalText += seg + " ";
          else interim += seg;
        }
        setQuestion(finalText || interim);
      };
      rec.onerror = () => {
        toast("Browser STT error. Falling back to server.");
        rec.stop();
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
    } catch {
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
      const supportsWebm = typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported("audio/webm");
      const mr = new MediaRecorder(stream, supportsWebm ? { mimeType: "audio/webm" } : undefined);

      chunksRef.current = [];
      mr.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = async () => {
        const type = supportsWebm ? "audio/webm" : "audio/wav";
        const blob = new Blob(chunksRef.current, { type });
        await sendToSTT(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorderRef.current = mr;
      mr.start();
      setRecording(true);
    } catch {
      toast("Microphone permission denied.");
    }
  }

  function stopBrowserSTT() {
    const rec = browserRecRef.current;
    if (rec) rec.stop();
    setRecording(false);
  }

  function stopServerRecording() {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    setRecording(false);
  }

  async function sendToSTT(blob: Blob) {
    const fd = new FormData();
    fd.append("audio", new File([blob], "query.webm", { type: blob.type || "audio/webm" }));
    try {
      const res = await fetch(`${API_BASE}/stt`, { method: "POST", body: fd });
      if (res.status === 429) {
        toast("Server STT quota exceeded. Enable Browser STT.");
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
    } catch (e: unknown) {
      toast(errorMessage(e) || "Transcription failed");
    }
  }

  const disabled = ingesting || asking;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur bg-white/70 border-b">
        <div className="mx-auto max-w-5xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Logo />
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight">PDF Q&A Chatbot</h1>
          </div>
          <EnvPill />
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-5xl px-4 py-6 grid gap-6 lg:grid-cols-2">
        {/* Left column: Upload */}
        <section className="lg:sticky lg:top-20 h-fit">
          <Card>
            <h2 className="text-lg font-semibold">1) Upload & Index PDFs</h2>
            <p className="text-sm text-slate-600">Drop files below or click to select. Only PDFs are accepted.</p>

            <div
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDragging(false);
              }}
              onDrop={onDrop}
              className={`mt-4 border-2 border-dashed rounded-2xl p-6 text-center transition ${isDragging ? "bg-slate-100" : "bg-slate-50"}`}
            >
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
                disabled={files.length === 0 || disabled}
                className="px-4 py-2 rounded-xl bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {ingesting ? "Indexing…" : "Upload & Index"}
              </button>
              {files.length > 0 && (
                <button onClick={clearFiles} disabled={disabled} className="px-3 py-2 rounded-xl border hover:bg-slate-50">Clear</button>
              )}
            </div>
          </Card>
        </section>

        {/* Right column: Chat */}
        <section>
          <Card>
            <h2 className="text-lg font-semibold">2) Ask a question</h2>

            <div className="mt-4 space-y-3 max-h-[60vh] overflow-auto pr-1">
              {messages.map((m) => (
                <ChatBubble key={m.id} role={m.role} text={m.text} />
              ))}
              {asking && <TypingBubble />}
              <div ref={bottomRef} />
            </div>

            <div className="mt-4 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleAsk();
                    }
                  }}
                  placeholder="Ask about your PDFs…"
                  className="flex-1 border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-slate-300"
                />
                <button
                  onClick={handleAsk}
                  disabled={asking || !question.trim()}
                  className="px-4 py-2 rounded-xl bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {asking ? "Thinking…" : "Ask"}
                </button>
              </div>

              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-slate-600 select-none">
                  <input
                    type="checkbox"
                    checked={useBrowserSTT}
                    onChange={(e) => setUseBrowserSTT(e.target.checked)}
                  />
                  Browser STT
                </label>

                <button
                  type="button"
                  onClick={() => {
                    if (recording) {
                      useBrowserSTT ? stopBrowserSTT() : stopServerRecording();
                    } else {
                      useBrowserSTT ? startBrowserSTT() : startServerRecording();
                    }
                  }}
                  title={recording ? "Stop recording" : "Start recording"}
                  className={`px-3 py-2 rounded-xl border hover:bg-slate-50 ${recording ? "border-red-500 text-red-600" : ""}`}
                  disabled={asking}
                >
                  {recording ? "Stop" : "Mic"}
                </button>
              </div>
            </div>
          </Card>
        </section>
      </main>

      <Toaster />
    </div>
  );
}

// ---------- UI PRIMITIVES ----------
function Card({ children }: { children: React.ReactNode }) {
  return <div className="rounded-2xl border bg-white/90 shadow-sm p-5">{children}</div>;
}

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

function Logo() {
  return (
    <div className="w-8 h-8 rounded-xl bg-slate-900 text-white grid place-items-center font-bold" aria-label="Logo">Q</div>
  );
}

function EnvPill() {
  const base = useMemo(() => {
    try {
      return new URL(API_BASE).host;
    } catch {
      return API_BASE;
    }
  }, []);
  return (
    <div className="text-xs px-2 py-1 rounded-full border bg-white/70 text-slate-600" title={API_BASE}>
      API: {base}
    </div>
  );
}

// ---------- TOASTS ----------
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
        <div key={i.id} className="px-4 py-2 rounded-xl bg-slate-900 text-white shadow">
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

// ---------- HELPERS ----------
function uid() {
  const g = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function" ? crypto.randomUUID() : undefined;
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

function CloudIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M7 18H17C19.2091 18 21 16.2091 21 14C21 11.9848 19.5223 10.3184 17.5894 10.0458C16.9942 7.74078 14.9289 6 12.5 6C10.2387 6 8.30734 7.4246 7.65085 9.50138C5.61892 9.68047 4 11.3838 4 13.4375C4 15.6228 5.79086 17.5 8 17.5H7Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}
