const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:80000";

export async function apiFetch(path: string, opts: RequestInit ={}) {
    const headers = new Headers(opts.headers || {});
    
}