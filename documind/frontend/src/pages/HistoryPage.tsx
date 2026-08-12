import { useState } from "react";
import { AnswerCard } from "@/components/AnswerCard";
import type { AskResponse } from "@/lib/api";

const DEMO_HISTORY: AskResponse[] = [
  {
    question: "What's the difference between a query parameter and a path parameter?",
    answer:
      "A path parameter is part of the URL path itself (e.g. /items/{item_id}) and is always required. A query parameter comes after ? in the URL (e.g. /items?skip=0) and can have a default value, making it optional.",
    citations: [
      { doc: "tutorial_path-params.md", section: "1" },
      { doc: "tutorial_query-params.md", section: "1" },
    ],
    trace: [
      { stage: "interpret", summary: "Comparison question spanning two docs.", detail: "decomposed_into=2 sub-queries" },
      { stage: "retrieve", summary: "Pulled chunks from both path-params and query-params docs.", detail: "top_k=6" },
      { stage: "rerank", summary: "Kept top 2 chunks per doc.", detail: "score_top1=0.91" },
      { stage: "synthesize", summary: "Merged both into a single comparative answer.", detail: "model=gpt-4o-mini" },
    ],
  },
];

export function HistoryPage() {
  const [items] = useState<AskResponse[]>(DEMO_HISTORY);

  return (
    <div className="mx-auto max-w-3xl px-6 py-6">
      <header className="mb-4">
        <h1 className="font-display text-2xl font-semibold text-ink">History</h1>
        <p className="text-sm text-muted">Past questions asked in this workspace.</p>
      </header>

      <div className="space-y-3">
        {items.length === 0 && (
          <div className="rounded-lg border border-dashed border-border px-6 py-10 text-center">
            <p className="text-sm text-muted">No history yet.</p>
          </div>
        )}
        {items.map((item, i) => (
          <AnswerCard key={i} item={item} />
        ))}
      </div>
    </div>
  );
}
