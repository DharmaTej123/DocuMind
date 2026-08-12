import { useState } from "react";
import { cn } from "@/lib/utils";
import type { TraceStage } from "@/lib/api";

const STAGE_LABEL: Record<TraceStage["stage"], string> = {
  interpret: "interpret_query",
  retrieve: "retrieve",
  rerank: "rerank",
  synthesize: "synthesize",
};

const STAGE_COLOR: Record<TraceStage["stage"], string> = {
  interpret: "bg-muted",
  retrieve: "bg-signal",
  rerank: "bg-attention",
  synthesize: "bg-ink",
};

/**
 * Signature element: renders the actual LangGraph node sequence that produced
 * an answer. Each node is expandable to show what happened at that stage —
 * this is the pipeline itself, not a decorative loading animation.
 */
export function AgentTrace({ trace }: { trace: TraceStage[] }) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <div className="mt-3 border-t border-border pt-3">
      <div className="flex items-center gap-1.5">
        {trace.map((stage, i) => (
          <div key={stage.stage} className="flex items-center gap-1.5">
            <button
              onClick={() => setOpenIndex(openIndex === i ? null : i)}
              className={cn(
                "group flex items-center gap-1.5 rounded-sm px-2 py-1 font-mono text-[11px] transition-colors",
                openIndex === i ? "bg-ink text-white" : "text-muted hover:bg-paper"
              )}
              aria-expanded={openIndex === i}
            >
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full",
                  openIndex === i ? "bg-white" : STAGE_COLOR[stage.stage]
                )}
              />
              {STAGE_LABEL[stage.stage]}
            </button>
            {i < trace.length - 1 && (
              <span className="h-px w-3 bg-border" aria-hidden />
            )}
          </div>
        ))}
      </div>

      {openIndex !== null && (
        <div className="mt-2 rounded-md bg-paper px-3 py-2">
          <p className="font-mono text-[11px] uppercase tracking-wide text-muted">
            {STAGE_LABEL[trace[openIndex].stage]}
          </p>
          <p className="mt-1 text-sm text-ink">{trace[openIndex].summary}</p>
          <p className="mt-1 font-mono text-xs text-muted">
            {trace[openIndex].detail}
          </p>
        </div>
      )}
    </div>
  );
}
