// app/admin/login/page.tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function LoginPage(){
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading,setLoading] = useState(false);
  const router = useRouter();

  async function submit(e: React.FormEvent){
    e.preventDefault();
    if(!username.trim()||!password.trim()){ alert("Fill fields"); return }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/admin/login`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ username, password }),
      });
      if(!res.ok){ const t = await res.text(); alert(t || "Login failed"); return }
      const j = await res.json();
      if(j.access_token){
        localStorage.setItem("admin_token", j.access_token);
        router.push("/admin/dashboard");
      } else {
        alert("No token returned");
      }
    } catch (e) {
      alert("Network error");
    } finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-white">
      <div className="w-full max-w-md bg-white border border-black rounded-lg p-8 shadow-[0_4px_12px_rgba(0,0,0,0.1)]">
        <h2 className="text-2xl font-bold mb-6 text-black text-center">Admin Login</h2>
        <form onSubmit={submit} className="space-y-5">
          <div>
            <input 
              value={username} 
              onChange={(e)=>setUsername(e.target.value)} 
              placeholder="Username" 
              className="w-full border border-black rounded px-4 py-3 bg-white text-black placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
            />
          </div>
          <div>
            <input 
              value={password} 
              onChange={(e)=>setPassword(e.target.value)} 
              placeholder="Password" 
              type="password" 
              className="w-full border border-black rounded px-4 py-3 bg-white text-black placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
            />
          </div>
          <button 
            type="submit" 
            disabled={loading} 
            className="w-full bg-black text-white py-3 rounded font-medium hover:bg-gray-900 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Signing In..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  )
}