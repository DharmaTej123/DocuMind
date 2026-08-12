import { useState, type FormEvent } from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ChatComposer({
  onAsk,
  loading,
}: {
  onAsk: (question: string) => void;
  loading?: boolean;
}) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!value.trim() || loading) return;
    onAsk(value.trim());
    setValue("");
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-end gap-2 rounded-lg border border-border bg-surface p-2 shadow-panel"
    >
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
          }
        }}
        rows={1}
        placeholder="Ask about your uploaded docs — e.g. how do I add a background task?"
        className="max-h-32 flex-1 resize-none bg-transparent px-2 py-2 text-sm text-ink placeholder:text-muted focus:outline-none"
      />
      <Button
        type="submit"
        size="sm"
        variant="signal"
        disabled={loading || !value.trim()}
        aria-label="Send question"
      >
        <ArrowUp className="h-4 w-4" />
      </Button>
    </form>
  );
}
