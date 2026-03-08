"use client";

import { useAppState, Phase } from "@/lib/store";
import { cn } from "@/lib/utils";
import {
  Upload,
  MessageSquare,
  Target,
  Loader2,
  BarChart3,
  Check,
  Sun,
  Moon,
} from "lucide-react";

const phases: { id: Phase; label: string; icon: typeof Upload }[] = [
  { id: "import", label: "Import Data", icon: Upload },
  { id: "variables", label: "Data Ingestion", icon: MessageSquare },
  { id: "intent", label: "Study Intent", icon: Target },
  { id: "analysis", label: "Analysis", icon: Loader2 },
  { id: "results", label: "Results", icon: BarChart3 },
];

export function Sidebar() {
  const { phase, setPhase, completedPhases, theme, toggleTheme } = useAppState();

  const canNavigate = (p: Phase) => {
    return completedPhases.includes(p) || p === phase;
  };

  return (
    <aside className="w-64 border-r border-border bg-sidebar flex flex-col h-screen">
      <div className="px-6 py-4 border-b border-border flex items-center h-[57px]">
        <h1 className="text-lg font-semibold text-foreground tracking-tight">
          Multiverse
        </h1>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-3 mb-3">
          Workflow
        </p>
        {phases.map((p, i) => {
          const isActive = phase === p.id;
          const isCompleted = completedPhases.includes(p.id);
          const isLocked = !canNavigate(p.id);
          const Icon = p.icon;

          return (
            <button
              key={p.id}
              onClick={() => canNavigate(p.id) && setPhase(p.id)}
              disabled={isLocked}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all",
                isActive &&
                  "bg-accent text-sidebar-active font-medium",
                isCompleted &&
                  !isActive &&
                  "text-sidebar-foreground hover:text-sidebar-active hover:bg-accent/50 cursor-pointer",
                isLocked &&
                  "text-muted-foreground/40 cursor-not-allowed"
              )}
            >
              <div className="relative">
                {isCompleted && !isActive ? (
                  <div className="w-5 h-5 rounded-full bg-success/20 flex items-center justify-center">
                    <Check className="w-3 h-3 text-success" />
                  </div>
                ) : (
                  <Icon
                    className={cn(
                      "w-5 h-5",
                      isActive && p.id === "analysis" && "animate-spin"
                    )}
                  />
                )}
              </div>
              <span>{p.label}</span>
              {isActive && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-foreground" />
              )}
            </button>
          );
        })}
      </nav>
      <div className="p-4 border-t border-border space-y-3">
        <button
          onClick={toggleTheme}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
        >
          {theme === "dark" ? (
            <Sun className="w-4 h-4" />
          ) : (
            <Moon className="w-4 h-4" />
          )}
          <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>
        </button>
        <p className="text-xs text-muted-foreground px-3">
          Hackathon POC v0.1
        </p>
      </div>
    </aside>
  );
}
