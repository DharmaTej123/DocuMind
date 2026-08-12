import { Card } from "@/components/ui/card";
import { CitationChip } from "@/components/CitationChip";
import { AgentTrace } from "@/components/AgentTrace";
import type { AskResponse } from "@/lib/api";

export function AnswerCard({ item }: { item: AskResponse }) {
  return (
    <Card className="p-4">
      <p className="font-display text-[15px] font-medium text-ink">
        {item.question}
      </p>
      <p className="mt-2 text-sm leading-relaxed text-ink/90">{item.answer}</p>

      {item.citations.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {item.citations.map((c, i) => (
            <CitationChip key={i} citation={c} />
          ))}
        </div>
      )}

      <AgentTrace trace={item.trace} />
    </Card>
  );
}
