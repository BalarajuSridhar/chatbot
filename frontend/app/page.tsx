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
  model?: string; // Add model info to messages
};
type LangCode = "en" | "te" | "ta" | "hi" | "mr" | "kn" | "ml" | "bn";

// AI Model types
type AIModel = {
  id: string;
  name: string;
  provider: string;
  description: string;
  icon: string;
  version: string;
  isPopular?: boolean;
  available?: boolean;
};

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

// BlobEvent type for TypeScript
type BlobEvent = Event & {
  data: Blob;
};

// Available AI Models
const AI_MODELS: AIModel[] = [
  // OpenAI GPT Models
  {
    id: "openai",
    name: "GPT-4o Mini", 
    provider: "OpenAI",
    description: "Fast and cost-effective GPT-4 model",
    icon: "🤖",
    version: "Latest",
    isPopular: true,
    available: true
  },
  {
    id: "gpt-4-turbo",
    name: "GPT-4 Turbo",
    provider: "OpenAI",
    description: "Advanced reasoning and problem solving",
    icon: "⚡", 
    version: "Latest",
    available: true
  },
  
  // Ollama Models
  {
    id: "ollama",
    name: "Llama 3.1",
    provider: "Ollama",
    description: "Local AI model for privacy and offline use",
    icon: "🦙",
    version: "8B",
    available: false // Will be updated dynamically
  }
];

// Helper function to get language name
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

// Move API host calculation outside component to avoid hook issues
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
    { id: uid(), role: "system", text: "Upload PDFs in Admin, then ask questions about them." },
  ]);
  const [language, setLanguage] = useState<LangCode>("en");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [useBrowserSTT, setUseBrowserSTT] = useState(true);
  const [recording, setRecording] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.5);
  const [selectedModel, setSelectedModel] = useState<AIModel>(AI_MODELS[0]);
  const [availableModels, setAvailableModels] = useState<AIModel[]>(AI_MODELS);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [showModelConfirmation, setShowModelConfirmation] = useState(false);
  const [pendingModel, setPendingModel] = useState<AIModel | null>(null);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const browserRecRef = useRef<SR | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  // Calculate API host once at component level
  const apiHost = useMemo(() => getApiHost(), []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  // Fetch datasets and available models
  useEffect(() => {
    let mounted = true;
    
    // Fetch datasets
    fetch(`${API_BASE}/datasets`)
      .then((r) => r.json())
      .then((j) => {
        if (!mounted) return;
        setDatasets(j.datasets || []);
      })
      .catch(() => setDatasets([]));

    // Fetch available models from backend
    fetchAvailableModels();
    
    return () => {
      mounted = false;
    };
  }, []);

  // Fetch available models from backend
const fetchAvailableModels = async () => {
  setLoadingModels(true);
  try {
    console.log('Fetching models from:', `${API_BASE}/models`);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
    
    const res = await fetch(`${API_BASE}/models`, {
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!res.ok) {
      console.warn(`Models endpoint returned ${res.status}`);
      throw new Error(`Failed to fetch models: ${res.status}`);
    }
    
    const data = await res.json();
    console.log('Received models data:', data);
    
    const backendModels = data.models || [];
    
    // Update our models with availability from backend
    const updatedModels = AI_MODELS.map(model => {
      const backendModel = backendModels.find((m: any) => m.id === model.id);
      return {
        ...model,
        available: backendModel ? backendModel.available : model.id === "openai", // Default OpenAI to available
        name: backendModel?.name || model.name,
        description: backendModel?.description || model.description
      };
    });
    
    setAvailableModels(updatedModels);
    
    // Set default model to first available one
    const firstAvailable = updatedModels.find(m => m.available) || updatedModels[0];
    setSelectedModel(firstAvailable);
    
  } catch (error) {
    console.warn("Failed to fetch models, using defaults:", error);
    
    // Fallback to default models
    const fallbackModels = AI_MODELS.map(model => ({
      ...model,
      available: model.id === "openai" // Only enable OpenAI by default
    }));
    
    setAvailableModels(fallbackModels);
    setSelectedModel(fallbackModels[0]);
    
    // Only show toast for network errors, not for aborted requests
    if ((error as any)?.name !== 'AbortError') {
      toast("Using default models - backend unavailable");
    }
  } finally {
    setLoadingModels(false);
  }
};

  // ---------- AI Model Selection ----------
  const handleModelSelect = (model: AIModel) => {
    if (!model.available) {
      toast(`${model.name} is currently unavailable`);
      return;
    }
    
    if (model.id === selectedModel.id) {
      setShowModelSelector(false);
      return;
    }
    
    setPendingModel(model);
    setShowModelConfirmation(true);
    setShowModelSelector(false);
  };

  const confirmModelChange = () => {
    if (pendingModel) {
      setSelectedModel(pendingModel);
      toast(`Switched to ${pendingModel.name}`);
      
      // Add model info to the next assistant message
      setMessages(prev => prev.map(msg => 
        msg.role === 'assistant' && !msg.model 
          ? { ...msg, model: pendingModel.id }
          : msg
      ));
    }
    setShowModelConfirmation(false);
    setPendingModel(null);
  };

  const cancelModelChange = () => {
    setShowModelConfirmation(false);
    setPendingModel(null);
  };

  // ---------- ASK ----------
  async function handleAsk() {
    const q = question.trim();
    if (!q) return toast("Type a question first.");
    
    // Add user message with current model context
    setMessages((p) => [...p, { 
      id: uid(), 
      role: "user", 
      text: q 
    }]);
    
    setQuestion("");
    setAsking(true);
    
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          question: q, 
          top_k: 4, 
          language, 
          dataset: dataset || undefined,
          model: selectedModel.id // Send selected model to backend
        }),
      });
      
      if (!res.ok) throw new Error(await safeText(res));
      const data = (await res.json()) as AskResponse;
      
      // Add assistant message with model info
      setMessages((p) => [...p, { 
        id: uid(), 
        role: "assistant", 
        text: data.answer || "(no answer)",
        model: selectedModel.id // Store model info with response
      }]);
      
    } catch (e) {
      setMessages((p) => [
        ...p,
        { 
          id: uid(), 
          role: "assistant", 
          text: `Error: ${errorMessage(e) || "Ask failed"}`,
          model: selectedModel.id
        },
      ]);
    } finally {
      setAsking(false);
    }
  }

  // ---------- TTS with Speed Control ----------
  async function playTTS(text: string, speed: number = playbackSpeed) {
    try {
      if (!text) return;
      stopTTS();
      
      const res = await fetch(`${API_BASE}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          text, 
          language,
          speed: speed
        }),
      });
      
      if (!res.ok) throw new Error(await safeText(res));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      
      audio.playbackRate = speed;
      audioRef.current = audio;
      audio.onended = () => URL.revokeObjectURL(url);
      await audio.play();
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

  // ---------- Feedback ----------
  async function handleFeedback(
    messageId: string, 
    feedback: 'good' | 'bad' | 'no_response', 
    question?: string, 
    answer?: string
  ) {
    // Update UI immediately
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, feedback } : msg
    ));
    
    try {
      const message = messages.find(m => m.id === messageId);
      await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message_id: messageId, 
          feedback,
          question: question || messages.find(m => m.id === messageId)?.text,
          answer: answer || messages.find(m => m.id === messageId && m.role === 'assistant')?.text,
          model_used: message?.model || selectedModel.id, // Include model info in feedback
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
    // Browser Speech Recognition has limited language support
    const supportedMap: Record<LangCode, string> = {
      "en": "en-US",
      "hi": "hi-IN",    // Hindi - supported in Chrome
      "te": "en-US",    // Telugu - not widely supported
      "ta": "en-US",    // Tamil - not widely supported  
      "mr": "en-US",    // Marathi - not widely supported
      "kn": "en-US",    // Kannada - not widely supported
      "ml": "en-US",    // Malayalam - not widely supported
      "bn": "en-US"     // Bengali - not widely supported
    };
    
    const locale = supportedMap[code];
    
    // Show warning for unsupported languages
    if (locale === "en-US" && code !== "en") {
      console.warn(`Browser STT doesn't support ${code}, falling back to English`);
    }
    
    return locale;
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

    // Check if the language is supported by browser STT
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

  // Clear conversation
  function clearChat() {
    setMessages([{ id: uid(), role: "system", text: "Upload PDFs in Admin, then ask questions about them." }]);
  }

  // Refresh models
  function refreshModels() {
    fetchAvailableModels();
    toast("Refreshing available models...");
  }

  // Get model display name
  function getModelDisplayName(modelId: string): string {
    const model = availableModels.find(m => m.id === modelId);
    return model ? model.name : modelId;
  }

  // Get model icon
  function getModelIcon(modelId: string): string {
    const model = availableModels.find(m => m.id === modelId);
    return model ? model.icon : "🤖";
  }

  // ---------- UI ----------
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex flex-col overflow-hidden">
      {/* Header - MSME Branded Design - Updated for Mobile */}
      <header className="sticky top-0 z-50 backdrop-blur-lg bg-white/95 border-b border-blue-200/50 shadow-sm">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
          {/* Logo and Brand Section */}
          <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0 min-w-0">
            {/* MSME Logo Container */}
            <div className="flex items-center gap-2 sm:gap-3">
              <img 
                src="/images/msme-logo.png" 
                alt="MSME Logo"
                className="h-10 sm:h-12 w-auto flex-shrink-0"
              />
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

          {/* Navigation and Controls Section */}
          <div className="flex items-center gap-2 sm:gap-4 lg:gap-6 flex-shrink-0">
            {/* Navigation Links - Hidden on small mobile, visible from sm up */}
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

              {/* AI Model Selector - Now placed next to Contact button */}
              <div className="relative">
                <button
                  onClick={() => setShowModelSelector(!showModelSelector)}
                  className="px-4 py-2.5 rounded-xl border border-teal-200 bg-gradient-to-r from-teal-50 to-blue-50 text-teal-800 font-semibold text-sm shadow-sm hover:shadow-md transition-all duration-200 flex items-center gap-2 active:scale-95 hover:from-teal-100 hover:to-blue-100"
                  disabled={loadingModels}
                >
                  {loadingModels ? (
                    <div className="w-4 h-4 border-2 border-teal-600 border-t-transparent rounded-full animate-spin"></div>
                  ) : (
                    <>
                      <span className="text-lg">{selectedModel.icon}</span>
                      <span className="hidden lg:inline">{selectedModel.name}</span>
                      <span className="lg:hidden">Model</span>
                    </>
                  )}
                  <svg className={`w-4 h-4 transition-transform ${showModelSelector ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {/* Model Selector Dropdown */}
                {showModelSelector && (
                  <div className="absolute top-full right-0 mt-2 w-72 sm:w-80 bg-white rounded-2xl shadow-2xl border border-gray-200 backdrop-blur-lg z-50 max-h-96 overflow-y-auto">
                    <div className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-bold text-gray-900 text-lg">Select AI Model</h3>
                        <div className="flex items-center gap-2">
                          <button 
                            onClick={refreshModels}
                            className="p-1.5 hover:bg-gray-100 rounded-xl transition-colors"
                            title="Refresh models"
                          >
                            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          </button>
                          <button 
                            onClick={() => setShowModelSelector(false)}
                            className="p-1.5 hover:bg-gray-100 rounded-xl transition-colors"
                          >
                            <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        {availableModels.map((model) => (
                          <button
                            key={model.id}
                            onClick={() => handleModelSelect(model)}
                            disabled={!model.available}
                            className={`w-full p-4 rounded-xl border-2 transition-all duration-200 text-left hover:shadow-lg group ${
                              selectedModel.id === model.id
                                ? 'border-blue-500 bg-blue-50 shadow-md scale-[1.02]'
                                : model.available
                                ? 'border-gray-200 bg-white hover:border-blue-300 hover:scale-[1.01]'
                                : 'border-gray-100 bg-gray-50 opacity-60 cursor-not-allowed'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                <span className="text-2xl group-hover:scale-110 transition-transform">{model.icon}</span>
                                <div>
                                  <div className="font-semibold text-gray-900 text-sm">
                                    {model.name}
                                    {!model.available && (
                                      <span className="ml-2 text-xs text-gray-500">(Unavailable)</span>
                                    )}
                                  </div>
                                  <div className="text-xs text-gray-500">
                                    {model.provider} • {model.version}
                                  </div>
                                </div>
                              </div>
                              {model.isPopular && (
                                <span className="px-2.5 py-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white text-xs rounded-full font-medium shadow-sm">
                                  Popular
                                </span>
                              )}
                            </div>
                            <div className="text-xs text-gray-600 mt-2 leading-relaxed">
                              {model.description}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </nav>

            {/* API Indicator - Hidden on very small screens */}
            <div className="hidden sm:flex items-center gap-2 text-xs px-3 py-2 rounded-full border border-blue-200 bg-white/80 text-gray-600 backdrop-blur-sm whitespace-nowrap">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              API: {apiHost}
            </div>
            
            {/* Admin Panel Button - Visible on desktop */}
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

            {/* Mobile Menu Button - Shows on small screens */}
            <div className="sm:hidden relative">
              <button
                onClick={() => setShowMobileMenu(!showMobileMenu)}
                className="p-3 rounded-xl border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 relative z-60 shadow-sm hover:shadow-md active:scale-95"
                aria-label="Toggle menu"
              >
                <div className="w-6 h-6 flex flex-col justify-center relative">
                  <span className={`block h-0.5 w-6 bg-current transform transition-all duration-300 ${showMobileMenu ? 'rotate-45 translate-y-2' : ''}`} />
                  <span className={`block h-0.5 w-6 bg-current transition-all duration-300 mt-1.5 ${showMobileMenu ? 'opacity-0' : 'opacity-100'}`} />
                  <span className={`block h-0.5 w-6 bg-current transform transition-all duration-300 mt-1.5 ${showMobileMenu ? '-rotate-45 -translate-y-2' : ''}`} />
                </div>
              </button>

              {/* Backdrop for closing menu when clicking outside */}
              {showMobileMenu && (
                <div 
                  className="fixed inset-0 bg-black bg-opacity-30 z-40 backdrop-blur-sm"
                  onClick={() => setShowMobileMenu(false)}
                />
              )}

              {/* Mobile Menu Dropdown */}
              {showMobileMenu && (
                <div className="absolute top-full right-0 mt-3 w-80 bg-white rounded-2xl shadow-2xl border border-gray-200 backdrop-blur-lg z-50 animate-scale-in overflow-hidden">
                  {/* Menu Header */}
                  <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4 text-white">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <div>
                          <h3 className="font-bold text-lg">MSME Assistant</h3>
                          <p className="text-blue-100 text-sm">AI-Powered Support</p>
                        </div>
                      </div>
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    </div>
                  </div>

                  <div className="max-h-[70vh] overflow-y-auto">
                    <div className="p-4 space-y-3">
                      {/* Navigation Links */}
                      <div className="space-y-2">
                        <button
                          onClick={() => {
                            router.push("/");
                            setShowMobileMenu(false);
                          }}
                          className="w-full text-left px-4 py-3.5 rounded-xl text-gray-700 hover:bg-blue-50 hover:text-blue-600 transition-all duration-200 font-medium flex items-center gap-3 group border border-gray-100 hover:border-blue-200 active:scale-95"
                        >
                          <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center group-hover:bg-blue-200 transition-colors">
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
                          className="w-full text-left px-4 py-3.5 rounded-xl text-gray-700 hover:bg-blue-50 hover:text-blue-600 transition-all duration-200 font-medium flex items-center gap-3 group border border-gray-100 hover:border-blue-200 active:scale-95"
                        >
                          <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <div>
                            <div className="font-semibold">Contact</div>
                            <div className="text-xs text-gray-500">Get in touch with us</div>
                          </div>
                        </button>
                      </div>

                      {/* Divider */}
                      <div className="border-t border-gray-200 my-4"></div>

                      {/* AI Model Selection */}
                      <div className="space-y-3">
                        <div className="flex items-center gap-2 px-1">
                          <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center">
                            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                          </div>
                          <div>
                            <div className="font-semibold text-gray-900 text-sm">AI Model</div>
                            <div className="text-xs text-gray-500">Choose your assistant</div>
                          </div>
                        </div>
                        
                        <select 
                          value={selectedModel.id}
                          onChange={(e) => {
                            const model = availableModels.find(m => m.id === e.target.value);
                            if (model && model.available) {
                              setSelectedModel(model);
                              toast(`Switched to ${model.name}`);
                            } else if (model) {
                              toast(`${model.name} is currently unavailable`);
                            }
                            setShowMobileMenu(false);
                          }}
                          className="w-full border-2 border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white font-medium"
                        >
                          {availableModels.map((model) => (
                            <option 
                              key={model.id} 
                              value={model.id}
                              disabled={!model.available}
                            >
                              {model.icon} {model.name} {!model.available && '(Unavailable)'}
                            </option>
                          ))}
                        </select>
                      </div>

                      {/* API Info */}
                      <div className="bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl p-4 border border-gray-200">
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center">
                            <svg className="w-3 h-3 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                          <div className="text-xs font-semibold text-gray-700">API STATUS</div>
                        </div>
                        <div className="text-xs text-gray-600 bg-white rounded-lg px-3 py-2 font-mono border border-gray-300">
                          {apiHost}
                        </div>
                      </div>

                      {/* Admin Panel Button - Mobile */}
                      <button
                        onClick={() => {
                          goToAdmin();
                          setShowMobileMenu(false);
                        }}
                        className="w-full px-4 py-4 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 transition-all duration-200 font-semibold flex items-center justify-center gap-3 shadow-lg hover:shadow-xl active:scale-95 mt-2"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span className="text-base">Admin Panel</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Model Change Confirmation Popup */}
      {showModelConfirmation && pendingModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6 animate-scale-in border border-gray-200">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-100 to-purple-100 flex items-center justify-center">
                <span className="text-2xl">{pendingModel.icon}</span>
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg">Switch AI Model?</h3>
                <p className="text-sm text-gray-600">You're about to change the AI model</p>
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl p-4 mb-6 border border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{pendingModel.icon}</span>
                  <div>
                    <div className="font-semibold text-gray-900">{pendingModel.name}</div>
                    <div className="text-xs text-gray-500">{pendingModel.provider}</div>
                  </div>
                </div>
                <div className="text-xs bg-blue-100 text-blue-700 px-2.5 py-1 rounded-full font-medium">
                  {pendingModel.version}
                </div>
              </div>
              <div className="text-sm text-gray-600 mt-2 leading-relaxed">
                {pendingModel.description}
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={cancelModelChange}
                className="flex-1 px-4 py-3.5 rounded-xl border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-all duration-200 font-medium hover:border-gray-400 active:scale-95"
              >
                Cancel
              </button>
              <button
                onClick={confirmModelChange}
                className="flex-1 px-4 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 transition-all duration-200 font-medium shadow-lg hover:shadow-xl active:scale-95"
              >
                Switch Model
              </button>
            </div>
          </div>
        </div>
      )}

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
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div
                      className={`group max-w-[90%] sm:max-w-[85%] px-4 sm:px-6 py-3 sm:py-5 leading-relaxed whitespace-pre-wrap rounded-3xl shadow-lg backdrop-blur-sm border transform transition-all duration-300 hover:scale-[1.02] ${
                        m.role === "user" 
                          ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-br-md shadow-blue-500/25 border-blue-500/20" 
                          : "bg-white/90 text-gray-800 rounded-bl-md shadow-gray-200/50 border-gray-100/50"
                      }`}
                    >
                      {/* Message Header */}
                      <div className={`flex items-center gap-2 mb-2 sm:mb-3 ${m.role === "user" ? "text-blue-100" : "text-gray-500"}`}>
                        {m.role === "assistant" ? (
                          <>
                            <div className="w-5 h-5 sm:w-6 sm:h-6 rounded-full bg-gradient-to-r from-teal-400 to-blue-500 flex items-center justify-center">
                              <span className="text-xs text-white">{getModelIcon(m.model || selectedModel.id)}</span>
                            </div>
                            <span className="text-xs font-semibold">
                              {getModelDisplayName(m.model || selectedModel.id)}
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

                      {/* Message Content */}
                      <div className="text-sm sm:text-[15px] leading-6 sm:leading-7 font-medium">{m.text}</div>
                      
                      {/* Assistant Actions */}
                      {m.role === "assistant" && (
                        <div className="mt-3 sm:mt-4 flex flex-wrap items-center gap-2 sm:gap-3">
                          {/* Action Buttons */}
                          <div className="flex gap-1 sm:gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <button 
                              onClick={() => playTTS(m.text, playbackSpeed)} 
                              className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full border border-blue-200 bg-white text-gray-700 hover:bg-blue-50 transition-all duration-200 text-xs font-medium shadow-sm hover:shadow-md active:scale-95"
                            >
                              <span>🔊</span>
                              <span className="hidden sm:inline">Speak</span>
                            </button>
                            <button 
                              onClick={() => stopTTS()} 
                              className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 transition-all duration-200 text-xs font-medium shadow-sm hover:shadow-md active:scale-95"
                            >
                              <span>■</span>
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

                          {/* Speed Control */}
                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <div className="h-4 w-px bg-gray-300 mx-1 sm:mx-2"></div>
                            <span className="text-xs text-gray-500 font-medium mr-1 hidden sm:inline">Speed:</span>
                            <select 
                              onChange={(e) => {
                                const newSpeed = parseFloat(e.target.value);
                                setPlaybackSpeed(newSpeed);
                                if (audioRef.current) {
                                  audioRef.current.playbackRate = newSpeed;
                                }
                              }}
                              className="text-xs border border-gray-200 rounded-lg px-2 py-1 bg-white focus:outline-none focus:ring-1 focus:ring-blue-500 hover:border-gray-300 transition-colors"
                              value={playbackSpeed}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <option value="0.75">0.75x</option>
                              <option value="1.0">1.0x</option>
                              <option value="1.25">1.25x</option>
                              <option value="1.5">1.5x</option>
                              <option value="1.75">1.75x</option>
                              <option value="2.0">2.0x</option>
                            </select>
                          </div>

                          {/* Feedback Buttons */}
                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <div className="h-4 w-px bg-gray-300 mx-1 sm:mx-2"></div>
                            <span className="text-xs text-gray-500 font-medium mr-1 hidden sm:inline">Helpful?</span>
                            <button
                              onClick={() => handleFeedback(
                                m.id, 
                                'good', 
                                messages.find(msg => msg.role === 'user' && messages.indexOf(msg) < messages.indexOf(m))?.text,
                                m.text
                              )}
                              className={`p-1.5 sm:p-2 rounded-full transition-all duration-200 hover:scale-110 ${
                                m.feedback === 'good' 
                                  ? 'bg-green-100 text-green-600 border border-green-200 shadow-sm' 
                                  : 'bg-white text-gray-400 hover:text-green-500 hover:bg-green-50 border border-gray-200 hover:border-green-200'
                              }`}
                              title="This answer was helpful"
                            >
                              <svg className="w-3 h-3 sm:w-4 sm:h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                              </svg>
                            </button>
                            <button
                              onClick={() => handleFeedback(
                                m.id, 
                                'bad', 
                                messages.find(msg => msg.role === 'user' && messages.indexOf(msg) < messages.indexOf(m))?.text,
                                m.text
                              )}
                              className={`p-1.5 sm:p-2 rounded-full transition-all duration-200 hover:scale-110 ${
                                m.feedback === 'bad' 
                                  ? 'bg-red-100 text-red-600 border border-red-200 shadow-sm' 
                                  : 'bg-white text-gray-400 hover:text-red-500 hover:bg-red-50 border border-gray-200 hover:border-red-200'
                              }`}
                              title="This answer was not helpful"
                            >
                              <svg className="w-3 h-3 sm:w-4 sm:h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                              </svg>
                            </button>
                            <button
                              onClick={() => handleFeedback(
                                m.id, 
                                'no_response', 
                                messages.find(msg => msg.role === 'user' && messages.indexOf(msg) < messages.indexOf(m))?.text,
                                m.text
                              )}
                              className={`p-1.5 sm:p-2 rounded-full transition-all duration-200 hover:scale-110 ${
                                m.feedback === 'no_response' 
                                  ? 'bg-yellow-100 text-yellow-600 border border-yellow-200 shadow-sm' 
                                  : 'bg-white text-gray-400 hover:text-yellow-500 hover:bg-yellow-50 border border-gray-200 hover:border-yellow-200'
                              }`}
                              title="No response needed"
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
                        <div className="text-sm font-medium">Analyzing with {selectedModel.name}...</div>
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
            {/* Settings Row */}
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
                  >
                    <option value="">All Documents</option>
                    {datasets.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
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
                  <span className="text-xs text-gray-500 hidden sm:inline">(Only EN, HI supported)</span>
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

            {/* Input Row */}
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
                  placeholder="Ask anything about MSME documents, Udyam registration, or government schemes... (Press Enter to send)"
                  className="w-full border border-gray-200 rounded-2xl px-4 sm:px-6 py-3 sm:py-4 text-base sm:text-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/50 backdrop-blur-sm font-medium placeholder-gray-400 pr-24 sm:pr-32 hover:border-gray-300 transition-colors"
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
                disabled={asking || !question.trim()} 
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
        <div key={i.id} className="px-4 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg backdrop-blur-sm border border-white/20 animate-scale-in">
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