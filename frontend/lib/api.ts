// lib/api.ts or wherever you keep your API functions

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"; // Fixed port

export async function apiFetch(path: string, opts: RequestInit = {}) {
    const headers = new Headers(opts.headers || {});
    
    // Add default headers if not present
    if (!headers.has("Content-Type") && !(opts.body instanceof FormData)) {
        headers.set("Content-Type", "application/json");
    }
    
    // Add authorization token if available
    if (typeof window !== "undefined") {
        const token = localStorage.getItem("admin_token");
        if (token && !headers.has("Authorization")) {
            headers.set("Authorization", `Bearer ${token}`);
        }
    }
    
    const config: RequestInit = {
        ...opts,
        headers,
    };
    
    try {
        const response = await fetch(`${API_BASE}${path}`, config);
        
        // Handle unauthorized responses
        if (response.status === 401) {
            // Clear invalid token
            if (typeof window !== "undefined") {
                localStorage.removeItem("admin_token");
                // Redirect to login if on admin page
                if (window.location.pathname.includes("/admin")) {
                    window.location.href = "/admin/login";
                }
            }
            throw new Error("Unauthorized. Please login again.");
        }
        
        return response;
    } catch (error) {
        console.error(`API fetch error for ${path}:`, error);
        throw error;
    }
}

// Helper methods for common HTTP verbs
export const api = {
    get: (path: string, opts?: RequestInit) => 
        apiFetch(path, { ...opts, method: "GET" }),
    
    post: (path: string, data?: any, opts?: RequestInit) => 
        apiFetch(path, {
            ...opts,
            method: "POST",
            body: data instanceof FormData ? data : JSON.stringify(data),
        }),
    
    put: (path: string, data?: any, opts?: RequestInit) => 
        apiFetch(path, {
            ...opts,
            method: "PUT",
            body: data instanceof FormData ? data : JSON.stringify(data),
        }),
    
    delete: (path: string, opts?: RequestInit) => 
        apiFetch(path, { ...opts, method: "DELETE" }),
    
    patch: (path: string, data?: any, opts?: RequestInit) => 
        apiFetch(path, {
            ...opts,
            method: "PATCH",
            body: data instanceof FormData ? data : JSON.stringify(data),
        }),
};