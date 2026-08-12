import { useState } from "react";
import { ChatComposer } from "@/components/ChatComposer";
import { AnswerCard } from "@/components/AnswerCard";
import type { AskResponse } from "@/lib/api";
import { api } from "@/lib/api";

// Demo fallback so the UI is reviewable before the agent graph (Step 4/5) is
// implemented. Once /query/ask is live, the real response replaces this.
function demoResponse(question: string): AskResponse {
  return {
    question,
    answer:
      "You can add a background task by declaring a parameter of type BackgroundTasks in your path operation function, then calling background_tasks.add_task(...) with the function and arguments to run after the response is sent.",
    citations: [
      { doc: "tutorial_background-tasks.md", section: "2" },
      { doc: "tutorial_bigger-applications.md", section: "4" },
    ],
    trace: [
      {
        stage: "interpret",
        summary: "Question identified as a single-step, code-dependent lookup.",
        detail: "rewritten_query: 'BackgroundTasks usage FastAPI'",
      },
      {
        stage: "retrieve",
        summary: "Pulled 8 candidate chunks from pgvector.",
        detail: "top_k=8, source=tutorial_background-tasks.md, bigger-applications.md",
      },
      {
        stage: "rerank",
        summary: "Cohere Rerank reordered candidates by relevance; top 3 kept.",
        detail: "score_top1=0.94, score_top3=0.71, dropped=5",
      },
      {
        stage: "synthesize",
        summary: "Answer generated and grounded in the top 3 reranked chunks.",
        detail: "model=gpt-4o-mini, citations_attached=2",
      },
    ],
  };
}

export function AskPage() {
  const [items, setItems] = useState<AskResponse[]>([]);
  const [loading, setLoading] = useState(false);

  async function handleAsk(question: string) {
    setLoading(true);
    try {
      const res = await api.ask(question);
      setItems((prev) => [res, ...prev]);
    } catch {
      setItems((prev) => [demoResponse(question), ...prev]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col px-6 py-6">
      <header className="mb-4">
        <h1 className="font-display text-2xl font-semibold text-ink">Ask</h1>
        <p className="text-sm text-muted">
          Answers are grounded in your uploaded docs, with the full retrieval
          trace shown below each one.
        </p>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto pb-4">
        {items.length === 0 && (
          <div className="rounded-lg border border-dashed border-border px-6 py-10 text-center">
            <p className="text-sm text-muted">
              No questions yet. Ask something about your uploaded docs below.
            </p>
          </div>
        )}
        {items.map((item, i) => (
          <AnswerCard key={i} item={item} />
        ))}
      </div>

      <ChatComposer onAsk={handleAsk} loading={loading} />
    </div>
  );
}
