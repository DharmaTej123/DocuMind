import { FileText } from "lucide-react";
import type { Citation } from "@/lib/api";

export function CitationChip({ citation }: { citation: Citation }) {
  return (
    <button
      className="inline-flex items-center gap-1 rounded-sm border border-border bg-surface px-2 py-0.5 font-mono text-[11px] text-ink hover:border-signal hover:text-signal transition-colors"
      title={`${citation.doc} — ${citation.section}`}
    >
      <FileText className="h-3 w-3" />
      {citation.doc}
      <span className="text-muted">§{citation.section}</span>
    </button>
  );
}
