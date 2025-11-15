// utils/auth.ts
const ADMIN_CREDENTIALS = {
  username: "admin",
  password: "admin123", // Change this to your desired strong password
  email: "admin@pdfchat.com"
};

export function verifyAdminToken(token: string | null): boolean {
  if (!token) return false;
  
  try {
    const decoded = JSON.parse(atob(token));
    return decoded.username === ADMIN_CREDENTIALS.username;
  } catch {
    return false;
  }
}

export function getAdminCredentials() {
  return ADMIN_CREDENTIALS;
}