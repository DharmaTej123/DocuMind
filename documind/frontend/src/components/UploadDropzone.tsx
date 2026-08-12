import { useRef, useState, type DragEvent } from "react";
import { UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

export function UploadDropzone({
  onFiles,
}: {
  onFiles: (files: FileList) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors",
        dragging ? "border-signal bg-signal-soft" : "border-border bg-surface"
      )}
    >
      <UploadCloud className={cn("h-6 w-6", dragging ? "text-signal" : "text-muted")} />
      <p className="text-sm text-ink">
        Drop PDF, Markdown, or HTML files here
      </p>
      <p className="font-mono text-xs text-muted">or click to browse</p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.md,.html"
        className="hidden"
        onChange={(e) => e.target.files && onFiles(e.target.files)}
      />
    </div>
  );
}
