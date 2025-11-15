// app/page.tsx
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type AskResponse = { answer: string };
type STTResponse = { text: string };
type MsgRole = "user" | "assistant" | "system";
type Message = { id: string; role: MsgRole; text: string };
type LangCode = "en" | "te" | "ta" | "hi";

// --- Minimal Web Speech API types so TS doesn't require lib.dom ---
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

export default function Page() {
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { id: uid(), role: "system", text: "Upload PDFs in Admin, then ask questions about them." },
  ]);
  const [language, setLanguage] = useState<LangCode>("en");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [useBrowserSTT, setUseBrowserSTT] = useState(true);
  const [recording, setRecording] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const browserRecRef = useRef<SR | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [messages, asking]);

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

  // ---------- ASK ----------
  async function handleAsk() {
    const q = question.trim();
    if (!q) return toast("Type a question first.");
    setMessages((p) => [...p, { id: uid(), role: "user", text: q }]);
    setQuestion("");
    setAsking(true);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: 4, language, dataset: dataset || undefined }),
      });
      if (!res.ok) throw new Error(await safeText(res));
      const data = (await res.json()) as AskResponse;
      setMessages((p) => [...p, { id: uid(), role: "assistant", text: data.answer || "(no answer)" }]);
    } catch (e) {
      setMessages((p) => [
        ...p,
        { id: uid(), role: "assistant", text: `Error: ${errorMessage(e) || "Ask failed"}` },
      ]);
    } finally {
      setAsking(false);
    }
  }

  // ---------- TTS ----------
  async function playTTS(text: string) {
    try {
      if (!text) return;
      // Stop any playing
      try {
        audioRef.current?.pause();
        audioRef.current = null;
      } catch {}
      const res = await fetch(`${API_BASE}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
      });
      if (!res.ok) throw new Error(await safeText(res));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      audioRef.current = a;
      a.onended = () => URL.revokeObjectURL(url);
      await a.play();
    } catch (e) {
      toast("TTS failed");
      console.error(e);
    }
  }

  function stopTTS() {
    try {
      audioRef.current?.pause();
      audioRef.current = null;
    } catch {}
  }

  // ---------- STT (Browser + Server fallback) ----------
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
      toast("Browser STT not supported. Falling back to server STT.");
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
        toast("Browser STT error. Falling back to server STT.");
        try {
          rec.stop();
        } catch {}
        startServerRecording();
      };
      rec.onend = () => {
        setRecording(false);
        const trimmed = (rec as any).__final || "".trim(); // not needed; finalText captured above
        // final text already applied in onresult when isFinal fired
        if ((trimmed || "").trim()) toast("Transcribed (browser)");
      };

      rec.start();
      setRecording(true);
    } catch (err) {
      console.warn("Browser STT init failed", err);
      toast("Browser STT init failed. Falling back to server STT.");
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
      const supportsWebm = typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported?.("audio/webm");
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
    } catch (err) {
      console.warn("Server recording failed", err);
      toast("Microphone permission denied or unavailable.");
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

  // Combined toggle for mic
  async function toggleRecording() {
    if (recording) {
      if (useBrowserSTT) stopBrowserSTT();
      else stopServerRecording();
    } else {
      if (useBrowserSTT) startBrowserSTT();
      else startServerRecording();
    }
  }

  // Navigate to admin login
  function goToAdmin() {
    router.push("/admin/login");
  }

  // ---------- UI ----------
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur bg-white/70 border-b">
        <div className="mx-auto max-w-5xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-slate-900 text-white grid place-items-center font-bold">Q</div>
            <h1 className="text-xl sm:text-2xl font-bold">PDF Q&A Chatbot</h1>
            <div className="ml-4 text-sm text-slate-600 hidden sm:block">Ask questions about your uploaded PDFs</div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-xs px-2 py-1 rounded-full border bg-white/70 text-slate-600">
              API:{" "}
              {useMemo(() => {
                try {
                  return new URL(API_BASE).host;
                } catch {
                  return API_BASE;
                }
              }, [])}
            </div>
            
            {/* Admin Login Button */}
            <button
              onClick={goToAdmin}
              className="text-xs px-3 py-1.5 rounded-full border border-slate-300 bg-white/70 text-slate-700 hover:bg-slate-50 transition-colors"
            >
              Admin Login
            </button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-5xl px-4 py-6 grid gap-6 lg:grid-cols-1">
        <section>
          <div className="rounded-2xl border bg-white/90 shadow-sm p-5">
            <h2 className="text-lg font-semibold mb-4">Ask questions about your documents</h2>

            <div className="mb-3 flex items-center gap-3 text-sm text-slate-600">
              <label className="flex items-center gap-2">
                Answer language:
                <select value={language} onChange={(e) => setLanguage(e.target.value as LangCode)} className="border rounded px-2 py-1 ml-1">
                  <option value="en">English</option>
                  <option value="te">Telugu</option>
                  <option value="ta">Tamil</option>
                  <option value="hi">Hindi</option>
                </select>
              </label>

              <label className="flex items-center gap-2">
                Dataset:
                <select value={dataset} onChange={(e) => setDataset(e.target.value)} className="border rounded px-2 py-1 ml-1">
                  <option value="">(All)</option>
                  {datasets.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex items-center gap-2">
                <input type="checkbox" checked={useBrowserSTT} onChange={(e) => setUseBrowserSTT(e.target.checked)} />
                <span>Browser STT</span>
              </label>
            </div>

            <div className="space-y-3 max-h-[54vh] overflow-auto pr-1 mb-4">
              {messages.map((m) => (
                <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"} mb-2`}>
                  <div
                    className={`max-w-[90%] sm:max-w-[80%] px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      m.role === "user" ? "bg-slate-900 text-white rounded-t-2xl rounded-bl-2xl" : "bg-slate-50 rounded-t-2xl rounded-br-2xl"
                    }`}
                  >
                    <div>{m.text}</div>
                    {m.role === "assistant" && (
                      <div className="mt-2 flex gap-2">
                        <button onClick={() => playTTS(m.text)} className="text-xs px-2 py-1 rounded border hover:bg-slate-100">
                          🔊 Speak
                        </button>
                        <button onClick={() => stopTTS()} className="text-xs px-2 py-1 rounded border hover:bg-slate-100">
                          ■ Stop
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {asking && (
                <div className="text-sm text-slate-500">
                  Thinking…
                </div>
              )}
              <div ref={bottomRef} />
            </div>

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
                placeholder="Ask about your documents…"
                className="flex-1 border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-slate-300"
              />

              <button
                onClick={toggleRecording}
                title={recording ? "Stop recording" : "Start recording"}
                className={`px-3 py-2 rounded-xl border hover:bg-slate-50 ${recording ? "border-red-500 text-red-600" : ""}`}
                disabled={asking}
              >
                {recording ? "Stop" : "Mic"}
              </button>

              <button onClick={handleAsk} disabled={asking || !question.trim()} className="px-4 py-2 rounded-xl bg-slate-900 text-white">
                {asking ? "Thinking…" : "Ask"}
              </button>
            </div>
          </div>
        </section>
      </main>

      <Toaster />
    </div>
  );
}

// ---------- Toaster ----------
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

// ---------- Helpers ----------
function uid() {
  const g = typeof crypto !== "undefined" && typeof (crypto as any).randomUUID === "function" ? (crypto as any).randomUUID() : undefined;
  return g || `id_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
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