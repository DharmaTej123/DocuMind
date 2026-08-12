import { NavLink } from "react-router-dom";
import { MessageSquareText, FileStack, History, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Ask", icon: MessageSquareText },
  { to: "/documents", label: "Documents", icon: FileStack },
  { to: "/history", label: "History", icon: History },
];

export function Sidebar({ onLogout }: { onLogout: () => void }) {
  return (
    <aside className="flex h-screen w-56 flex-col bg-ink px-3 py-4 text-white/80">
      <div className="px-2 pb-6">
        <p className="font-display text-lg font-semibold text-white">
          DocuMind
        </p>
        <p className="mt-0.5 font-mono text-[11px] text-white/40">
          workspace · engineering docs
        </p>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-ink-600 text-white"
                  : "text-white/60 hover:bg-ink-700 hover:text-white"
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <button
        onClick={onLogout}
        className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-white/50 hover:bg-ink-700 hover:text-white transition-colors"
      >
        <LogOut className="h-4 w-4" />
        Log out
      </button>
    </aside>
  );
}
