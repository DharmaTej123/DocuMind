import { Navigate } from "react-router-dom";

export function RequireAuth({ children }: { children: React.ReactElement }) {
  const token = localStorage.getItem("documind_token");
  // Demo mode: allow entry without a token so the UI is reviewable before
  // Step 4/5 backend auth is live. Tighten this once /auth is implemented.
  if (!token && !import.meta.env.DEV) {
    return <Navigate to="/login" replace />;
  }
  return children;
}
