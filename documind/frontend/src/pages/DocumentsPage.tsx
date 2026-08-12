import { useState } from "react";
import { FileText, Trash2 } from "lucide-react";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/card";
import type { DocumentItem} from "@/lib/api";
import { api } from "@/lib/api";
import { useEffect } from "react";
// Demo seed data — replaced by api.listDocuments() once Step 3/4 ingestion
// (upload -> parse -> chunk -> embed) is wired to this page.
// const DEMO_DOCS: DocumentItem[] = [
//   { id: "1", filename: "tutorial_background-tasks.md", doc_type: "markdown", created_at: "2026-07-24" },
//   { id: "2", filename: "tutorial_security_oauth2-jwt.md", doc_type: "markdown", created_at: "2026-07-24" },
//   { id: "3", filename: "advanced_custom-response.md", doc_type: "markdown", created_at: "2026-07-25" },
// ];


// export function DocumentsPage() {
//   const [docs, setDocs] = useState<DocumentItem[]>(DEMO_DOCS);

//   function handleFiles(files: FileList) {
//     const added: DocumentItem[] = Array.from(files).map((f, i) => ({
//       id: `pending-${Date.now()}-${i}`,
//       filename: f.name,
//       doc_type: f.name.split(".").pop() ?? "file",
//       created_at: new Date().toISOString().slice(0, 10),
//     }));
//     setDocs((prev) => [...added, ...prev]);
//   }

//   function handleDelete(id: string) {
//     setDocs((prev) => prev.filter((d) => d.id !== id));
//   }
export function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);

  useEffect(() => {
    api.listDocuments().then(setDocs).catch(() => {});
  }, []);

  async function handleFiles(files: FileList) {
    for (const file of Array.from(files)) {
      try {
        const uploaded = await api.upload(file);
        setDocs((prev) => [uploaded, ...prev]);
      } catch (err) {
        console.error("Upload failed:", err);
      }
    }
  }

  function handleDelete(id: string) {
    api.deleteDocument(id).catch(() => {});
    setDocs((prev) => prev.filter((d) => d.id !== id));
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-6">
      <header className="mb-4">
        <h1 className="font-display text-2xl font-semibold text-ink">Documents</h1>
        <p className="text-sm text-muted">
          Upload technical docs to add them to your workspace's retrieval index.
        </p>
      </header>

      <UploadDropzone onFiles={handleFiles} />

      <div className="mt-5 space-y-2">
        {docs.map((doc) => (
          <Card key={doc.id} className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-3">
              <FileText className="h-4 w-4 text-muted" />
              <div>
                <p className="text-sm text-ink">{doc.filename}</p>
                <p className="font-mono text-[11px] text-muted">
                  {doc.created_at}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge tone="muted">{doc.doc_type}</Badge>
              <button
                onClick={() => handleDelete(doc.id)}
                className="text-muted hover:text-danger transition-colors"
                aria-label={`Delete ${doc.filename}`}
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
