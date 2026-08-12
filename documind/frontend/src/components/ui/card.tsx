import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-surface shadow-panel",
        className
      )}
      {...props}
    />
  );
}

export function Badge({
  className,
  tone = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & {
  tone?: "default" | "signal" | "attention" | "muted";
}) {
  const tones: Record<string, string> = {
    default: "bg-paper text-ink",
    signal: "bg-signal-soft text-signal",
    attention: "bg-attention-soft text-attention",
    muted: "bg-paper text-muted",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm px-2 py-0.5 text-xs font-mono",
        tones[tone],
        className
      )}
      {...props}
    />
  );
}
