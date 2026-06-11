"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type AskResponse = { answer: string };
type STTResponse = { text: string };
type MsgRole = "user" | "assistant" | "system";
type Message = { 
  id: string; 
  role: MsgRole; 
  text: string; 
  feedback?: 'good' | 'bad' | 'no_response';
  model?: string;
};
type LangCode = "en" | "te" | "ta" | "hi" | "mr" | "kn" | "ml" | "bn";

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
type BlobEvent = Event & {
  data: Blob;
};

type TTSAudioState = {
  audioUrl: string | null;
  audioElement: HTMLAudioElement | null;
  currentTime: number;
  isPlaying: boolean;
  isPaused: boolean;
  speed: number;
  currentMessageId: string | null;
};

function getLanguageName(code: LangCode): string {
  const names: Record<LangCode, string> = {
    "en": "English",
    "te": "Telugu", 
    "ta": "Tamil",
    "hi": "Hindi",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali"
  };
  return names[code] || code;
}

function getApiHost() {
  try {
    return new URL(API_BASE).host;
  } catch {
    return API_BASE;
  }
}

export default function Page() {
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { id: uid(), role: "system", text: "Welcome to MSME Assistant! Ask me anything about MSME schemes, registration, and policies." },
  ]);
  const [language, setLanguage] = useState<LangCode>("en");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [useBrowserSTT, setUseBrowserSTT] = useState(true);
  const [recording, setRecording] = useState(false);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  
  // TTS state
  const [ttsSpeed, setTtsSpeed] = useState<number>(1.0);
  const [ttsState, setTtsState] = useState<TTSAudioState>({
    audioUrl: null,
    audioElement: null,
    currentTime: 0,
    isPlaying: false,
    isPaused: false,
    speed: 1.0,
    currentMessageId: null
  });

  const [showMobileMenu, setShowMobileMenu] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const browserRecRef = useRef<SR | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  const apiHost = useMemo(() => getApiHost(), []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  useEffect(() => {
    return () => {
      cleanupTTS();
    };
  }, []);

  // Check backend health and fetch datasets
  useEffect(() => {
    let mounted = true;
    
    const checkBackend = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        const res = await fetch(`${API_BASE}/health`, {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (res.ok) {
          setBackendAvailable(true);
          // Fetch datasets
          const datasetsRes = await fetch(`${API_BASE}/datasets`);
          if (datasetsRes.ok) {
            const j = await datasetsRes.json();
            if (mounted) setDatasets(j.datasets || []);
          }
        } else {
          setBackendAvailable(false);
        }
      } catch (error) {
        console.warn("Backend not available:", error);
        setBackendAvailable(false);
      }
    };
    
    checkBackend();
    
    return () => {
      mounted = false;
    };
  }, []);

  // Show backend status message
  useEffect(() => {
    if (backendAvailable === false && messages.length === 1) {
      setMessages(prev => [...prev, {
        id: uid(),
        role: "assistant",
        text: "⚠️ Backend server is not connected. Please make sure the backend is deployed and running. Check your API_BASE configuration.",
        model: "system"
      }]);
    } else if (backendAvailable === true && messages.length === 1 && messages[0].role === "system") {
      // Remove the system message if backend is available
      setMessages([]);
    }
  }, [backendAvailable]);

  // ---------- TTS Functions ----------
  function cleanupTTS() {
    try {
      if (ttsState.audioElement) {
        try {
          ttsState.audioElement.pause();
          ttsState.audioElement.currentTime = 0;
        } catch {}
      }
      if (ttsState.audioUrl) {
        try { URL.revokeObjectURL(ttsState.audioUrl); } catch {}
      }
      setTtsState({
        audioUrl: null,
        audioElement: null,
        currentTime: 0,
        isPlaying: false,
        isPaused: false,
        speed: ttsSpeed,
        currentMessageId: null
      });
    } catch (e) {
      console.error("Error cleaning up TTS:", e);
    }
  }

  async function playTTS(text: string, messageId: string) {
    if (!backendAvailable) {
      // Use browser speech synthesis as fallback
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = language === 'en' ? 'en-US' : 'en-US';
        utterance.rate = ttsSpeed;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
        setTtsState(prev => ({ ...prev, isPlaying: true, currentMessageId: messageId }));
        utterance.onend = () => {
          setTtsState(prev => ({ ...prev, isPlaying: false }));
        };
      } else {
        toast("Speech synthesis not supported");
      }
      return;
    }

    try {
      if (!text) return;

      if (ttsState.isPaused && ttsState.currentMessageId === messageId && ttsState.audioElement) {
        ttsState.audioElement.play();
        setTtsState(prev => ({
          ...prev,
          isPlaying: true,
          isPaused: false
        }));
        return;
      }

      cleanupTTS();

      const res = await fetch(`${API_BASE}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          text, 
          language,
          speed: ttsSpeed 
        }),
      });
      
      if (!res.ok) throw new Error(await safeText(res));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      audio.playbackRate = ttsSpeed;
      
      audio.onended = () => {
        cleanupTTS();
      };

      audio.onpause = () => {
        setTtsState(prev => ({
          ...prev,
          isPlaying: false,
          isPaused: true,
          currentTime: audio.currentTime
        }));
      };

      audio.onplay = () => {
        setTtsState(prev => ({
          ...prev,
          isPlaying: true,
          isPaused: false
        }));
      };

      setTtsState({
        audioUrl: url,
        audioElement: audio,
        currentTime: 0,
        isPlaying: true,
        isPaused: false,
        speed: ttsSpeed,
        currentMessageId: messageId
      });

      await audio.play();
    } catch (e) {
      console.error("TTS failed:", e);
      // Fallback to browser speech synthesis
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = ttsSpeed;
        window.speechSynthesis.speak(utterance);
      } else {
        toast("TTS failed");
      }
      cleanupTTS();
    }
  }

  function pauseResumeTTS() {
    try {
      if (ttsState.audioElement) {
        if (ttsState.isPlaying) {
          ttsState.audioElement.pause();
        } else if (ttsState.isPaused) {
          ttsState.audioElement.play();
        }
        return;
      }
      
      if ('speechSynthesis' in window) {
        if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
          window.speechSynthesis.pause();
          setTtsState(prev => ({ ...prev, isPlaying: false, isPaused: true }));
        } else if (window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
          setTtsState(prev => ({ ...prev, isPlaying: true, isPaused: false }));
        }
      }
    } catch (e) {
      console.error("Error in pauseResumeTTS:", e);
    }
  }

  function stopTTS() {
    cleanupTTS();
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }

  function changeTtsSpeed(speed: number) {
    setTtsSpeed(speed);
    if (ttsState.audioElement) {
      ttsState.audioElement.playbackRate = speed;
      setTtsState(prev => ({ ...prev, speed }));
    }
  }

  // ---------- ASK ----------
  async function handleAsk() {
    const q = question.trim();
    if (!q) return toast("Type a question first.");
    
    if (!backendAvailable) {
      toast("Backend not available. Please check your API configuration.");
      return;
    }
    
    setMessages((p) => [...p, { 
      id: uid(), 
      role: "user", 
      text: q 
    }]);
    
    setQuestion("");
    setAsking(true);
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          question: q, 
          top_k: 4, 
          language, 
          dataset: dataset || undefined,
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!res.ok) {
        const errorText = await safeText(res);
        throw new Error(errorText || `HTTP ${res.status}`);
      }
      
      const data = (await res.json()) as AskResponse;
      
      setMessages((p) => [...p, { 
        id: uid(), 
        role: "assistant", 
        text: data.answer || "I couldn't generate an answer. Please try again.",
        model: "gemini"
      }]);
      
    } catch (e: any) {
      let errorMsg = "Failed to get response";
      if (e.name === 'AbortError') {
        errorMsg = "Request timeout. Please try again.";
      } else if (e.message) {
        errorMsg = e.message;
      }
      
      setMessages((p) => [
        ...p,
        { 
          id: uid(), 
          role: "assistant", 
          text: `Error: ${errorMsg}`,
          model: "error"
        },
      ]);
    } finally {
      setAsking(false);
    }
  }

  // ---------- Feedback ----------
  async function handleFeedback(
    messageId: string, 
    feedback: 'good' | 'bad' | 'no_response'
  ) {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, feedback } : msg
    ));
    
    if (!backendAvailable) {
      toast("Feedback saved locally (backend not connected)");
      return;
    }
    
    try {
      const message = messages.find(m => m.id === messageId);
      await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message_id: messageId, 
          feedback,
          question: message?.role === 'user' ? message.text : undefined,
          answer: message?.role === 'assistant' ? message.text : undefined,
          timestamp: new Date().toISOString()
        }),
      });
      
      const feedbackText = {
        'good': '👍 Helpful',
        'bad': '👎 Not Helpful', 
        'no_response': '⏸️ No Response'
      }[feedback];
      
      toast(`Feedback recorded: ${feedbackText}`);
    } catch (e) {
      console.error("Failed to send feedback:", e);
      toast("Failed to record feedback");
    }
  }

  // ---------- STT Functions ----------
  function sttLocale(code: LangCode): string {
    const supportedMap: Record<LangCode, string> = {
      "en": "en-US",
      "hi": "hi-IN",
      "te": "en-US",
      "ta": "en-US",  
      "mr": "en-US",
      "kn": "en-US",
      "ml": "en-US",
      "bn": "en-US"
    };
    return supportedMap[code];
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

    const isSupported = ["en", "hi"].includes(language);
    if (!isSupported) {
      toast(`${getLanguageName(language)} not supported in browser STT. Using server STT.`);
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
        const trimmed = finalText.trim();
        if (trimmed) {
          setQuestion((prev) => (prev ? `${prev} ${trimmed}` : trimmed));
          toast("Transcribed (browser)");
        }
      };

      rec.start();
      setRecording(true);
    } catch (err) {
      toast("Browser STT init failed. Falling back to server STT.");
      startServerRecording();
    }
  }

  async function startServerRecording() {
    if (!backendAvailable) {
      toast("Server STT requires backend connection. Please use browser STT.");
      setUseBrowserSTT(true);
      startBrowserSTT();
      return;
    }
    
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

  async function toggleRecording() {
    if (recording) {
      if (useBrowserSTT) stopBrowserSTT();
      else stopServerRecording();
    } else {
      if (useBrowserSTT) startBrowserSTT();
      else startServerRecording();
    }
  }

  function goToAdmin() {
    router.push("/admin/login");
  }

  function clearChat() {
    setMessages([]);
    toast("Chat cleared");
  }

  function getModelDisplayName(modelId: string): string {
    return modelId === "gemini" ? "Google Gemini" : "MSME Assistant";
  }

  function getModelIcon(modelId: string): string {
    return "🌟";
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-lg bg-white/95 border-b border-blue-200/50 shadow-sm">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0 min-w-0">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="h-10 sm:h-12 w-10 sm:w-12 bg-gradient-to-r from-blue-600 to-blue-800 rounded-xl flex items-center justify-center text-white font-bold text-sm sm:text-base flex-shrink-0">
                MSME
              </div>
              <div className="flex flex-col min-w-0">
                <div className="text-sm sm:text-lg font-bold text-gray-900 uppercase tracking-tight sm:tracking-wide leading-tight line-clamp-1">
                  Micro, Small & Medium Enterprises
                </div>
                <div className="text-xs sm:text-sm text-gray-600 font-medium leading-tight mt-0.5 hidden xs:block">
                  Ministry of MSME, Government of India
                </div>
                <div className="text-[9px] sm:text-[10px] text-gray-500 leading-tight mt-1 hidden sm:block">
                  UDYAM • ENTREPRENEURSHIP • DEVELOPMENT
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4 lg:gap-6 flex-shrink-0">
            <nav className="hidden sm:flex items-center gap-4 lg:gap-6">
              <button
                onClick={() => router.push("/")}
                className="text-gray-700 hover:text-blue-600 font-medium transition-all duration-200 px-3 py-2 rounded-lg hover:bg-blue-50 border border-transparent hover:border-blue-200 active:scale-95"
              >
                Home
              </button>
              <button
                onClick={() => router.push("/contact")}
                className="text-gray-700 hover:text-blue-600 font-medium transition-all duration-200 px-3 py-2 rounded-lg hover:bg-blue-50 border border-transparent hover:border-blue-200 active:scale-95"
              >
                Contact
              </button>
            </nav>

            <div className="hidden sm:flex items-center gap-2 text-xs px-3 py-2 rounded-full border border-blue-200 bg-white/80 text-gray-600 backdrop-blur-sm whitespace-nowrap">
              <div className={`w-2 h-2 rounded-full animate-pulse ${backendAvailable === true ? 'bg-green-500' : backendAvailable === false ? 'bg-red-500' : 'bg-yellow-500'}`}></div>
              API: {apiHost}
            </div>
            
            <button
              onClick={goToAdmin}
              className="hidden sm:flex px-4 py-2.5 rounded-xl border border-blue-300 bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 transition-all duration-200 font-medium text-sm shadow-md hover:shadow-lg active:scale-95 whitespace-nowrap items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span>Admin Panel</span>
            </button>

            {/* Mobile Menu Button */}
            <div className="sm:hidden relative">
              <button
                onClick={() => setShowMobileMenu(!showMobileMenu)}
                className="p-3 rounded-xl border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 relative shadow-sm hover:shadow-md active:scale-95"
                aria-label="Toggle menu"
              >
                <div className="w-6 h-6 flex flex-col justify-center relative">
                  <span className={`block h-0.5 w-6 bg-current transform transition-all duration-300 ${showMobileMenu ? 'rotate-45 translate-y-2' : ''}`} />
                  <span className={`block h-0.5 w-6 bg-current transition-all duration-300 mt-1.5 ${showMobileMenu ? 'opacity-0' : 'opacity-100'}`} />
                  <span className={`block h-0.5 w-6 bg-current transform transition-all duration-300 mt-1.5 ${showMobileMenu ? '-rotate-45 -translate-y-2' : ''}`} />
                </div>
              </button>

              {showMobileMenu && (
                <>
                  <div 
                    className="fixed inset-0 bg-black bg-opacity-30 z-40 backdrop-blur-sm"
                    onClick={() => setShowMobileMenu(false)}
                  />
                  <div className="absolute top-full right-0 mt-3 w-80 bg-white rounded-2xl shadow-2xl border border-gray-200 backdrop-blur-lg z-50 overflow-hidden">
                    <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4 text-white">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-bold text-lg">MSME Assistant</h3>
                          <p className="text-blue-100 text-sm">AI-Powered Support</p>
                        </div>
                        <div className={`w-2 h-2 rounded-full animate-pulse ${backendAvailable === true ? 'bg-green-400' : 'bg-red-400'}`}></div>
                      </div>
                    </div>

                    <div className="p-4 space-y-3">
                      <button
                        onClick={() => {
                          router.push("/");
                          setShowMobileMenu(false);
                        }}
                        className="w-full text-left px-4 py-3 rounded-xl text-gray-700 hover:bg-blue-50 transition-all duration-200 font-medium flex items-center gap-3 border border-gray-100"
                      >
                        <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                          <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                          </svg>
                        </div>
                        <div>
                          <div className="font-semibold">Home</div>
                          <div className="text-xs text-gray-500">Return to homepage</div>
                        </div>
                      </button>
                      
                      <button
                        onClick={() => {
                          router.push("/contact");
                          setShowMobileMenu(false);
                        }}
                        className="w-full text-left px-4 py-3 rounded-xl text-gray-700 hover:bg-blue-50 transition-all duration-200 font-medium flex items-center gap-3 border border-gray-100"
                      >
                        <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                          <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                          </svg>
                        </div>
                        <div>
                          <div className="font-semibold">Contact</div>
                          <div className="text-xs text-gray-500">Get in touch with us</div>
                        </div>
                      </button>

                      <div className="border-t border-gray-200 my-4"></div>

                      <div className="bg-gray-50 rounded-xl p-4">
                        <div className="text-xs text-gray-600">
                          <div className="font-semibold mb-2">API Status</div>
                          <div className="font-mono break-all">{apiHost}</div>
                          <div className={`mt-2 text-xs ${backendAvailable === true ? 'text-green-600' : 'text-red-600'}`}>
                            {backendAvailable === true ? '✓ Connected' : backendAvailable === false ? '✗ Disconnected' : '⟳ Checking...'}
                          </div>
                        </div>
                      </div>

                      <button
                        onClick={() => {
                          goToAdmin();
                          setShowMobileMenu(false);
                        }}
                        className="w-full px-4 py-4 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 transition-all duration-200 font-semibold flex items-center justify-center gap-3 shadow-lg mt-2"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span className="text-base">Admin Panel</span>
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col mx-auto max-w-6xl w-full px-4 sm:px-6 py-4 sm:py-6 gap-4 sm:gap-6 overflow-hidden">
        {/* Chat Display Area */}
        <div className="flex-1 relative rounded-3xl overflow-hidden shadow-2xl bg-gradient-to-br from-white via-blue-50/30 to-indigo-100/20 border border-white/50">
          <div 
            ref={messagesContainerRef}
            className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-blue-200 scrollbar-track-blue-50 hover:scrollbar-thumb-blue-300"
          >
            <div className="min-h-full p-4 sm:p-8 flex flex-col justify-end">
              <div className="space-y-4 sm:space-y-6 max-w-4xl mx-auto w-full">
                {messages.filter(m => m.role !== 'system').map((m, index) => (
                  <div 
                    key={m.id} 
                    className={`flex ${m.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
                  >
                    <div
                      className={`group max-w-[90%] sm:max-w-[85%] px-4 sm:px-6 py-3 sm:py-5 leading-relaxed whitespace-pre-wrap rounded-3xl shadow-lg backdrop-blur-sm border transform transition-all duration-300 hover:scale-[1.02] ${
                        m.role === "user" 
                          ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-br-md shadow-blue-500/25 border-blue-500/20" 
                          : "bg-white/90 text-gray-800 rounded-bl-md shadow-gray-200/50 border-gray-100/50"
                      }`}
                    >
                      <div className={`flex items-center gap-2 mb-2 sm:mb-3 ${m.role === "user" ? "text-blue-100" : "text-gray-500"}`}>
                        {m.role === "assistant" ? (
                          <>
                            <div className="w-5 h-5 sm:w-6 sm:h-6 rounded-full bg-gradient-to-r from-teal-400 to-blue-500 flex items-center justify-center">
                              <span className="text-xs text-white">🌟</span>
                            </div>
                            <span className="text-xs font-semibold">
                              {getModelDisplayName(m.model || "gemini")}
                            </span>
                          </>
                        ) : (
                          <>
                            <div className="w-5 h-5 sm:w-6 sm:h-6 rounded-full bg-white/20 flex items-center justify-center">
                              <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                              </svg>
                            </div>
                            <span className="text-xs font-semibold">You</span>
                          </>
                        )}
                      </div>

                      <div className="text-sm sm:text-[15px] leading-6 sm:leading-7 font-medium">{m.text}</div>
                      
                      {m.role === "assistant" && m.model !== "error" && (
                        <div className="mt-3 sm:mt-4 flex flex-wrap items-center gap-2 sm:gap-3">
                          <div className="flex gap-1 sm:gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <button 
                              onClick={() => playTTS(m.text, m.id)} 
                              className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full border border-blue-200 bg-white text-gray-700 hover:bg-blue-50 transition-all duration-200 text-xs font-medium shadow-sm hover:shadow-md active:scale-95"
                            >
                              <span>🔊</span>
                              <span className="hidden sm:inline">Speak</span>
                            </button>
                            <button 
                              onClick={pauseResumeTTS}
                              className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 text-xs font-medium shadow-sm hover:shadow-md active:scale-95"
                            >
                              <span>{ttsState.isPlaying ? '⏸️' : '▶️'}</span>
                              <span className="hidden sm:inline">{ttsState.isPlaying ? 'Pause' : 'Resume'}</span>
                            </button>
                            <button 
                              onClick={stopTTS}
                              className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 text-xs font-medium shadow-sm hover:shadow-md active:scale-95"
                            >
                              <span>⏹️</span>
                              <span className="hidden sm:inline">Stop</span>
                            </button>
                            <button 
                              onClick={() => navigator.clipboard.writeText(m.text)}
                              className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 text-xs font-medium shadow-sm hover:shadow-md active:scale-95"
                            >
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                              </svg>
                              <span className="hidden sm:inline">Copy</span>
                            </button>
                          </div>

                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <div className="h-4 w-px bg-gray-300 mx-1 sm:mx-2"></div>
                            <span className="text-xs text-gray-500 font-medium mr-1 hidden sm:inline">Speed:</span>
                            <select 
                              onChange={(e) => changeTtsSpeed(parseFloat(e.target.value))}
                              className="text-xs border border-gray-200 rounded-lg px-2 py-1 bg-white focus:outline-none focus:ring-1 focus:ring-blue-500 hover:border-gray-300 transition-colors"
                              value={ttsSpeed}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <option value="0.5">0.5x</option>
                              <option value="0.75">0.75x</option>
                              <option value="1.0">1.0x</option>
                              <option value="1.25">1.25x</option>
                              <option value="1.5">1.5x</option>
                              <option value="2.0">2.0x</option>
                            </select>
                          </div>

                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <div className="h-4 w-px bg-gray-300 mx-1 sm:mx-2"></div>
                            <span className="text-xs text-gray-500 font-medium mr-1 hidden sm:inline">Helpful?</span>
                            <button
                              onClick={() => handleFeedback(m.id, 'good')}
                              className={`p-1.5 sm:p-2 rounded-full transition-all duration-200 hover:scale-110 ${
                                m.feedback === 'good' 
                                  ? 'bg-green-100 text-green-600 border border-green-200 shadow-sm' 
                                  : 'bg-white text-gray-400 hover:text-green-500 hover:bg-green-50 border border-gray-200 hover:border-green-200'
                              }`}
                              title="Helpful"
                            >
                              <svg className="w-3 h-3 sm:w-4 sm:h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                              </svg>
                            </button>
                            <button
                              onClick={() => handleFeedback(m.id, 'bad')}
                              className={`p-1.5 sm:p-2 rounded-full transition-all duration-200 hover:scale-110 ${
                                m.feedback === 'bad' 
                                  ? 'bg-red-100 text-red-600 border border-red-200 shadow-sm' 
                                  : 'bg-white text-gray-400 hover:text-red-500 hover:bg-red-50 border border-gray-200 hover:border-red-200'
                              }`}
                              title="Not Helpful"
                            >
                              <svg className="w-3 h-3 sm:w-4 sm:h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                              </svg>
                            </button>
                            <button
                              onClick={() => handleFeedback(m.id, 'no_response')}
                              className={`p-1.5 sm:p-2 rounded-full transition-all duration-200 hover:scale-110 ${
                                m.feedback === 'no_response' 
                                  ? 'bg-yellow-100 text-yellow-600 border border-yellow-200 shadow-sm' 
                                  : 'bg-white text-gray-400 hover:text-yellow-500 hover:bg-yellow-50 border border-gray-200 hover:border-yellow-200'
                              }`}
                              title="No Response"
                            >
                              <svg className="w-3 h-3 sm:w-4 sm:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
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
                    <div className="max-w-[90%] sm:max-w-[85%] px-4 sm:px-6 py-3 sm:py-5 bg-white/90 backdrop-blur-sm rounded-3xl rounded-bl-md shadow-lg border border-gray-100/50">
                      <div className="flex items-center gap-3 sm:gap-4 text-gray-600">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                          <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
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
        </div>

        {/* Control Panel */}
        <div className="rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-4 sm:p-6">
          <div className="flex flex-col gap-4 sm:gap-6">
            <div className="flex flex-wrap items-center justify-between gap-3 sm:gap-4">
              <div className="flex flex-wrap items-center gap-3 sm:gap-4 text-sm">
                <label className="flex items-center gap-2 font-medium text-gray-700">
                  <svg className="w-4 h-4 text-gray-500 hidden sm:block" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                  </svg>
                  Language:
                  <select 
                    value={language} 
                    onChange={(e) => setLanguage(e.target.value as LangCode)} 
                    className="border border-gray-200 rounded-xl px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium text-sm hover:border-gray-300 transition-colors"
                  >
                    <option value="en">🇺🇸 English</option>
                    <option value="te">🇮🇳 Telugu</option>
                    <option value="ta">🇮🇳 Tamil</option>
                    <option value="hi">🇮🇳 Hindi</option>
                    <option value="mr">🇮🇳 Marathi</option>
                    <option value="kn">🇮🇳 Kannada</option>
                    <option value="ml">🇮🇳 Malayalam</option>
                    <option value="bn">🇮🇳 Bengali</option>
                  </select>
                </label>

                <label className="flex items-center gap-2 font-medium text-gray-700">
                  <svg className="w-4 h-4 text-gray-500 hidden sm:block" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  Dataset:
                  <select 
                    value={dataset} 
                    onChange={(e) => setDataset(e.target.value)} 
                    className="border border-gray-200 rounded-xl px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium text-sm hover:border-gray-300 transition-colors"
                    disabled={!backendAvailable}
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
                    className="rounded text-blue-600 focus:ring-blue-500 w-4 h-4 hover:scale-110 transition-transform"
                  />
                  <span className="hidden sm:inline">Browser Speech Recognition</span>
                  <span className="sm:hidden">Browser STT</span>
                  <span className="text-xs text-gray-500 hidden sm:inline">(EN, HI only)</span>
                </label>
              </div>

              <button
                onClick={clearChat}
                className="px-4 py-2 rounded-xl border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 font-medium text-sm flex items-center gap-2 hover:shadow-md active:scale-95"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                <span className="hidden sm:inline">Clear Chat</span>
                <span className="sm:hidden">Clear</span>
              </button>
            </div>

            <div className="flex items-center gap-3 sm:gap-4">
              <div className="flex-1 relative">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleAsk();
                    }
                  }}
                  placeholder="Ask about MSME schemes, Udyam registration, government policies... (Press Enter)"
                  className="w-full border border-gray-200 rounded-2xl px-4 sm:px-6 py-3 sm:py-4 text-base sm:text-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/50 backdrop-blur-sm font-medium placeholder-gray-400 pr-24 sm:pr-32 hover:border-gray-300 transition-colors"
                  disabled={!backendAvailable}
                />
                <div className="absolute right-2 sm:right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-1 sm:gap-2">
                  <button
                    onClick={toggleRecording}
                    title={recording ? "Stop recording" : "Start recording"}
                    className={`p-2.5 sm:p-3.5 rounded-xl border transition-all duration-200 hover:shadow-md active:scale-95 ${
                      recording 
                        ? "border-red-500 bg-red-50 text-red-600 hover:bg-red-100 shadow-sm animate-pulse" 
                        : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50 hover:border-gray-400"
                    }`}
                    disabled={asking}
                  >
                    {recording ? (
                      <div className="w-4 h-4 sm:w-5 sm:h-5 bg-red-500 rounded-full"></div>
                    ) : (
                      <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>

              <button 
                onClick={handleAsk} 
                disabled={asking || !question.trim() || !backendAvailable} 
                className="px-6 sm:px-8 py-3 sm:py-4 rounded-2xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 transition-all duration-200 font-semibold text-base sm:text-lg shadow-lg hover:shadow-xl disabled:shadow-none transform hover:scale-105 disabled:scale-100 whitespace-nowrap active:scale-95"
              >
                {asking ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span className="hidden sm:inline">Asking...</span>
                  </div>
                ) : (
                  <span className="hidden sm:inline">Ask MSME Assistant</span>
                )}
              </button>
            </div>
            
            {!backendAvailable && (
              <div className="text-center text-sm text-red-600 bg-red-50 rounded-xl p-3">
                ⚠️ Backend not connected. Please check your API configuration and ensure the backend server is running.
              </div>
            )}
          </div>
        </div>
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
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `id_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
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