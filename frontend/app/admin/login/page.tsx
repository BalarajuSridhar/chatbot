// app/admin/login/page.tsx
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiHost, setApiHost] = useState("");
  const [mounted, setMounted] = useState(false);
  const router = useRouter();

  // Handle client-side only rendering
  useEffect(() => {
    setMounted(true);
    
    // Calculate API host on client side only
    try {
      setApiHost(new URL(API_BASE).host);
    } catch {
      setApiHost(API_BASE);
    }
  }, []);

  // Check if already logged in
  useEffect(() => {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("admin_token");
      if (token) {
        // Verify token is valid (simple check)
        try {
          const parts = token.split(".");
          if (parts.length === 3) {
            router.push("/admin/dashboard");
          }
        } catch (e) {
          // Invalid token, clear it
          localStorage.removeItem("admin_token");
        }
      }
    }
  }, [router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("Please fill in all fields");
      return;
    }
    
    setLoading(true);
    setError("");
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const res = await fetch(`${API_BASE}/admin/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!res.ok) {
        let errorMsg = "Login failed";
        try {
          const errorData = await res.json();
          errorMsg = errorData.detail || errorData.message || errorMsg;
        } catch {
          const text = await res.text();
          if (text) errorMsg = text;
        }
        setError(errorMsg);
        return;
      }
      
      const j = await res.json();
      if (j.access_token) {
        localStorage.setItem("admin_token", j.access_token);
        // Small delay to ensure storage is written
        setTimeout(() => {
          router.push("/admin/dashboard");
        }, 100);
      } else {
        setError("No token returned from server");
      }
    } catch (e: any) {
      if (e.name === 'AbortError') {
        setError("Request timeout. Please check if the backend server is running.");
      } else if (e.name === 'TypeError' && e.message.includes('fetch')) {
        setError(`Cannot connect to backend at ${API_BASE}. Please ensure the backend server is running.`);
      } else {
        setError(e.message || "Network error. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  // Don't render until mounted to avoid hydration issues
  if (!mounted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Header */}
        <div className="text-center">
          <div className="mx-auto flex justify-center mb-6">
            <div className="flex items-center gap-3">
              {/* MSME Logo - using regular img tag to avoid Next.js Image issues */}
              <div className="h-14 w-14 bg-gradient-to-r from-blue-600 to-blue-800 rounded-xl flex items-center justify-center text-white font-bold text-xl">
                MSME
              </div>
              <div className="flex flex-col text-left">
                <div className="text-lg font-bold text-gray-900 uppercase tracking-wide leading-tight">
                  Micro, Small & Medium Enterprises
                </div>
                <div className="text-sm text-gray-600 font-medium leading-tight">
                  Ministry of MSME, Government of India
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-8">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              Admin Portal
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              Access the administration dashboard
            </p>
          </div>
        </div>

        {/* Login Form */}
        <form onSubmit={submit} className="mt-8 space-y-6" noValidate>
          <div className="rounded-3xl backdrop-blur-lg bg-white/80 border border-white/50 shadow-2xl p-8">
            {error && (
              <div className="mb-6 rounded-xl bg-red-50 border border-red-200 p-4">
                <div className="text-sm text-red-700 text-center">{error}</div>
              </div>
            )}
            
            <div className="space-y-5">
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                  Username
                </label>
                <div className="relative">
                  <input
                    id="username"
                    name="username"
                    type="text"
                    autoComplete="off"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="Enter your username"
                    className="w-full border border-gray-200 rounded-2xl px-6 py-4 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/50 backdrop-blur-sm font-medium placeholder-gray-400 transition-colors hover:border-gray-300"
                    disabled={loading}
                  />
                  <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                </div>
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type="password"
                    autoComplete="off"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    className="w-full border border-gray-200 rounded-2xl px-6 py-4 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/50 backdrop-blur-sm font-medium placeholder-gray-400 transition-colors hover:border-gray-300"
                    disabled={loading}
                  />
                  <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                    </svg>
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full px-8 py-4 rounded-2xl bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 transition-all duration-200 font-semibold text-lg shadow-lg hover:shadow-xl disabled:shadow-none transform hover:scale-105 disabled:scale-100 active:scale-95"
              >
                {loading ? (
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Signing In...
                  </div>
                ) : (
                  "Sign In to Admin Panel"
                )}
              </button>
            </div>
          </div>

          {/* Back to Chat Button */}
          <div className="text-center">
            <button
              type="button"
              onClick={() => router.push("/")}
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors duration-200 font-medium flex items-center justify-center gap-2 mx-auto"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to MSME Assistant
            </button>
          </div>
        </form>

        {/* API Info */}
        <div className="text-center">
          <div className="text-xs text-gray-500 inline-flex items-center gap-2 px-3 py-2 rounded-full border border-blue-200/50 bg-white/80 backdrop-blur-sm">
            <div className={`w-2 h-2 rounded-full animate-pulse ${apiHost ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
            API: {apiHost || "Detecting..."}
          </div>
        </div>

        {/* Help Text */}
        <div className="text-center text-xs text-gray-400">
          <p>Default credentials: admin / admin123 (configure in backend .env)</p>
        </div>
      </div>
    </div>
  );
}