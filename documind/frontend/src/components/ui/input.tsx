import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "flex h-10 w-full rounded-md border border-border bg-surface px-3 text-sm text-ink placeholder:text-muted",
      "focus:outline-none focus-visible:ring-2 focus-visible:ring-signal focus-visible:ring-offset-1",
      className
    )}
    {...props}
  />
));
Input.displayName = "Input";
