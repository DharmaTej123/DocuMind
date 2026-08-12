const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem("documind_token");
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

export const api = {
  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  register: (email: string, password: string) =>
    request<{ id: string }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  upload: (file: File) => {
  const token = localStorage.getItem("documind_token");
  const formData = new FormData();
  formData.append("file", file);
  return fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  }).then((res) => {
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json() as Promise<DocumentItem>;
  });
},

  listDocuments: () => request<DocumentItem[]>("/documents/"),
  ask: (question: string) =>
    request<AskResponse>("/query/ask", {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
  history: () => request<AskResponse[]>("/query/history"),

  deleteDocument: (id: string) =>
  request<void>(`/documents/${id}`, { method: "DELETE" }),
};

export interface DocumentItem {
  id: string;
  filename: string;
  doc_type: string;
  created_at: string;
}

export interface TraceStage {
  stage: "interpret" | "retrieve" | "rerank" | "synthesize";
  summary: string;
  detail: string;
}

export interface Citation {
  doc: string;
  section: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: Citation[];
  trace: TraceStage[];
}
