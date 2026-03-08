"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useAppState, ChatMessage, ColumnProfile, DataModification } from "@/lib/store";
import { sendChatMessage, updateVariable, revertModification } from "@/lib/api";
import {
  Send,
  Loader2,
  ArrowRight,
  AlertTriangle,
  ChevronRight,
  X,
  ArrowLeftRight,
  Type,
  Sparkles,
  User,
  Filter,
} from "lucide-react";
import { Markdown } from "@/components/markdown";

const TYPE_OPTIONS = ["continuous", "binary", "count/ordinal", "categorical"];
const PANEL_MIN = 240;
const PANEL_MAX = 600;
const PANEL_DEFAULT = 360;
const CHAR_PX = 8;
const VARIABLE_TABLE_MIN_WIDTH = 460;

function clampPanelWidth(width: number): number {
  return Math.min(PANEL_MAX, Math.max(PANEL_MIN, Math.round(width)));
}

function estimatePanelWidth(
  dataProfile: { column_profiles: ColumnProfile[] } | null,
  modifications: DataModification[]
): number {
  if (!dataProfile) return PANEL_DEFAULT;

  // Fixed columns (type/dist/missing/skew + paddings/borders) in the variable table.
  const fixedVariableColumns = 230;
  const longestVarName = dataProfile.column_profiles.reduce(
    (maxLen, col) => Math.max(maxLen, (col.name || "").length),
    0
  );
  const nameColumnWidth = Math.max(90, longestVarName * CHAR_PX + 28);
  const variableTabWidth = Math.max(
    VARIABLE_TABLE_MIN_WIDTH,
    fixedVariableColumns + nameColumnWidth
  );

  // Modifications tab can have longer labels (rename/retype/transform descriptions).
  const longestModificationLabel = modifications.reduce((maxLen, mod) => {
    const label =
      mod.type === "transform"
        ? mod.description || mod.from || ""
        : `${mod.from || ""} -> ${mod.to || ""}`;
    return Math.max(maxLen, label.length);
  }, 0);
  const modificationsTabWidth = longestModificationLabel
    ? 70 + longestModificationLabel * CHAR_PX
    : PANEL_MIN;

  return clampPanelWidth(Math.max(variableTabWidth, modificationsTabWidth));
}

function DistributionBar({ profile }: { profile: ColumnProfile }) {
  const bins = profile.histogram;
  if (!bins || bins.length === 0) return <div className="w-16 h-4" />;
  const max = Math.max(...bins);
  if (max === 0) return <div className="w-16 h-4" />;
  return (
    <div className="flex items-end gap-px h-4 w-16">
      {bins.map((count, i) => (
        <div
          key={i}
          className="bg-foreground/50 rounded-[1px] flex-1 min-w-[2px]"
          style={{ height: `${(count / max) * 100}%` }}
        />
      ))}
    </div>
  );
}

function VariableRow({
  profile,
  sessionId,
  onUpdate,
}: {
  profile: ColumnProfile;
  sessionId: string;
  onUpdate: (name: string, updates: Partial<ColumnProfile>) => void;
}) {
  const [editingName, setEditingName] = useState(false);
  const [editName, setEditName] = useState(profile.name);

  const commitName = () => {
    const trimmed = editName.trim();
    if (trimmed && trimmed !== profile.name) {
      onUpdate(profile.name, { name: trimmed });
      updateVariable(sessionId, profile.name, { name: trimmed }).catch(() => {});
    } else {
      setEditName(profile.name);
    }
    setEditingName(false);
  };

  return (
    <tr className="border-b border-border last:border-0 hover:bg-accent/30 transition-colors text-xs">
      <td className="px-3 py-2">
        {editingName ? (
          <input
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={commitName}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitName();
              if (e.key === "Escape") {
                setEditName(profile.name);
                setEditingName(false);
              }
            }}
            className="bg-background border border-border rounded px-2 py-1 text-xs w-full focus:outline-none focus:ring-1 focus:ring-ring"
            autoFocus
          />
        ) : (
          <span
            className="font-medium cursor-text hover:text-foreground/70 transition-colors"
            onClick={() => setEditingName(true)}
            title="Click to rename"
          >
            {profile.name}
          </span>
        )}
      </td>
      <td className="px-3 py-2">
        <span className="text-muted-foreground">
          {profile.distribution || profile.dtype}
        </span>
      </td>
      <td className="px-3 py-2 text-center">
        <DistributionBar profile={profile} />
      </td>
      <td className="px-3 py-2 text-right text-muted-foreground">
        {profile.missing_pct}%
      </td>
      <td className="px-3 py-2 text-center">
        {profile.is_skewed && (
          <AlertTriangle className="w-3 h-3 text-warning inline" />
        )}
      </td>
    </tr>
  );
}

function ModificationRow({
  mod,
  onRevert,
}: {
  mod: DataModification;
  onRevert: (mod: DataModification) => void;
}) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-border last:border-0 hover:bg-accent/30 transition-colors text-xs group">
      <div className="flex-shrink-0">
        {mod.type === "rename" ? (
          <ArrowLeftRight className="w-3 h-3 text-info" />
        ) : mod.type === "transform" ? (
          <Filter className="w-3 h-3 text-success" />
        ) : (
          <Type className="w-3 h-3 text-warning" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        {mod.type === "transform" ? (
          <>
            <p className="font-medium leading-snug break-words">
              {mod.description || mod.from}
            </p>
            <p className="text-[10px] text-muted-foreground mt-0.5">
              → {mod.to}
            </p>
          </>
        ) : (
          <p className="font-medium leading-snug break-words">
            {mod.type === "rename" ? "Renamed" : "Retyped"}{" "}
            <span className="text-muted-foreground">{mod.from}</span>
            {" → "}
            <span>{mod.to}</span>
          </p>
        )}
        <p className="text-[10px] text-muted-foreground flex items-center gap-1 mt-0.5">
          {mod.source === "ai" ? (
            <Sparkles className="w-2.5 h-2.5" />
          ) : (
            <User className="w-2.5 h-2.5" />
          )}
          {mod.source === "ai" ? "AI suggested" : "Manual"}
        </p>
      </div>
      <button
        onClick={() => onRevert(mod)}
        className="p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors opacity-0 group-hover:opacity-100 flex-shrink-0"
        title="Revert this change"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
}

type PanelTab = "variables" | "modifications";

export function ChatPhase() {
  const {
    sessionId,
    chatMessages,
    setChatMessages,
    dataProfile,
    setDataProfile,
    setPhase,
    completedPhases,
    setCompletedPhases,
    setColumns,
    modifications,
    setModifications,
    addModification,
  } = useAppState();
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [panelWidth, setPanelWidth] = useState(PANEL_DEFAULT);
  const [panelTab, setPanelTab] = useState<PanelTab>("variables");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const hasManuallyResized = useRef(false);
  const lastAutoWidth = useRef<number | null>(null);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // Drag-to-resize logic
  const onMouseDownHandle = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      dragStartX.current = e.clientX;
      dragStartWidth.current = panelWidth;

      const onMove = (ev: MouseEvent) => {
        if (!dragging.current) return;
        hasManuallyResized.current = true;
        const delta = dragStartX.current - ev.clientX; // dragging left increases width
        const next = Math.min(
          PANEL_MAX,
          Math.max(PANEL_MIN, dragStartWidth.current + delta)
        );
        setPanelWidth(next);
      };
      const onUp = () => {
        dragging.current = false;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [panelWidth]
  );

  useEffect(() => {
    if (!dataProfile || hasManuallyResized.current) return;
    const autoWidth = estimatePanelWidth(dataProfile, modifications);
    if (autoWidth !== lastAutoWidth.current) {
      setPanelWidth(autoWidth);
      lastAutoWidth.current = autoWidth;
    }
  }, [dataProfile, modifications]);

  const handleSend = async () => {
    if (!input.trim() || !sessionId || loading) return;
    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    setChatMessages([...chatMessages, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChatMessage(sessionId, userMsg.content);
      setChatMessages([
        ...chatMessages,
        userMsg,
        { role: "assistant", content: data.response },
      ]);
      if (data.profile_updated && data.profile) {
        setDataProfile(data.profile);
        setColumns(data.profile.column_profiles.map((c: ColumnProfile) => c.name));
      }
      if (data.modifications) {
        for (const mod of data.modifications) {
          addModification(mod as DataModification);
        }
      }
    } catch {
      setChatMessages([
        ...chatMessages,
        userMsg,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleProceed = () => {
    setCompletedPhases([...completedPhases, "variables"]);
    setPhase("intent");
  };

  const handleVariableUpdate = (
    originalName: string,
    updates: Partial<ColumnProfile>
  ) => {
    if (!dataProfile) return;
    const updatedProfiles = dataProfile.column_profiles.map((col) =>
      col.name === originalName ? { ...col, ...updates } : col
    );
    const newProfile = { ...dataProfile, column_profiles: updatedProfiles };
    setDataProfile(newProfile);
    setColumns(updatedProfiles.map((c) => c.name));

    if (updates.name && updates.name !== originalName) {
      addModification({
        type: "rename",
        variable: updates.name,
        from: originalName,
        to: updates.name,
        source: "user",
        timestamp: Date.now() / 1000,
      });
    }
  };

  const handleRevertModification = async (mod: DataModification) => {
    if (!sessionId) return;
    try {
      const data = await revertModification(sessionId, mod.timestamp);
      if (data.profile) {
        setDataProfile(data.profile);
        setColumns(data.profile.column_profiles.map((c: ColumnProfile) => c.name));
      }
      if (data.modifications) {
        setModifications(data.modifications as DataModification[]);
      } else {
        setModifications([]);
      }
    } catch {
      // silently fail
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="border-b border-border px-6 flex items-center justify-between h-[57px] flex-shrink-0">
        <div>
          <h2 className="text-sm font-semibold">Data Ingestion</h2>
          <p className="text-xs text-muted-foreground">
            Review your data with the AI assistant
          </p>
        </div>
        <button
          onClick={handleProceed}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-foreground text-background text-sm font-medium hover:bg-foreground/90 transition-colors"
        >
          Continue
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto py-6">
            <div className="max-w-2xl mx-auto px-6 space-y-6">
              {chatMessages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {msg.role === "user" ? (
                    <div className="max-w-[75%] bg-accent rounded-2xl px-4 py-2.5 text-sm leading-relaxed">
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  ) : (
                    <div className="w-full text-sm leading-relaxed text-foreground">
                      <Markdown content={msg.content} />
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="p-4 flex-shrink-0">
            <div className="max-w-2xl mx-auto">
              <div className="flex items-center gap-2 bg-accent border border-border rounded-2xl px-4 py-2 focus-within:ring-1 focus-within:ring-ring transition-shadow">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) =>
                    e.key === "Enter" && !e.shiftKey && handleSend()
                  }
                  placeholder="Ask about your variables, distributions, or data quality..."
                  className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none py-1"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || loading}
                  className="p-1.5 rounded-lg bg-foreground text-background hover:bg-foreground/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Variable panel */}
        {dataProfile && (
          <>
            {/* Resize handle — only shown when expanded */}
            {!collapsed && (
              <div
                onMouseDown={onMouseDownHandle}
                className="w-1.5 flex-shrink-0 bg-border hover:bg-foreground/25 transition-colors cursor-col-resize group relative flex items-center justify-center"
              >
                <div className="w-4 h-8 rounded-full bg-border group-hover:bg-foreground/20 flex items-center justify-center">
                  <div className="w-0.5 h-4 rounded-full bg-muted-foreground/40" />
                </div>
              </div>
            )}

            {/* Panel itself */}
            <div
              className="flex flex-col border-l border-border overflow-hidden transition-[width] duration-200"
              style={{ width: collapsed ? 0 : panelWidth, flexShrink: 0 }}
            >
              {!collapsed && (
                <>
                  {/* Panel header with tabs */}
                  <div className="border-b border-border flex-shrink-0">
                    <div className="flex items-center justify-between px-4 pt-3 pb-0">
                      <p className="text-xs text-muted-foreground">
                        {dataProfile.rows.toLocaleString()} rows &middot;{" "}
                        {dataProfile.columns} cols &middot;{" "}
                        {dataProfile.missing_total_pct}% missing
                      </p>
                      <button
                        onClick={() => setCollapsed(true)}
                        className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
                        title="Collapse panel"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="flex px-4 mt-2 gap-1">
                      <button
                        onClick={() => setPanelTab("variables")}
                        className={`px-3 py-1.5 text-[11px] font-medium rounded-t transition-colors border-b-2 ${
                          panelTab === "variables"
                            ? "border-foreground text-foreground"
                            : "border-transparent text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        Variables
                      </button>
                      <button
                        onClick={() => setPanelTab("modifications")}
                        className={`px-3 py-1.5 text-[11px] font-medium rounded-t transition-colors border-b-2 flex items-center gap-1.5 ${
                          panelTab === "modifications"
                            ? "border-foreground text-foreground"
                            : "border-transparent text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        Modifications
                        {modifications.length > 0 && (
                          <span className="bg-foreground text-background text-[9px] font-bold rounded-full w-4 h-4 flex items-center justify-center">
                            {modifications.length}
                          </span>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Tab content */}
                  <div className="flex-1 overflow-y-auto">
                    {panelTab === "variables" ? (
                      <div className="overflow-x-auto">
                        <table className="w-full min-w-[460px]">
                          <thead>
                            <tr className="border-b border-border bg-accent/30 text-[10px] uppercase tracking-wider text-muted-foreground">
                              <th className="text-left px-3 py-2 font-medium">Name</th>
                              <th className="text-left px-3 py-2 font-medium">Type</th>
                              <th className="text-center px-3 py-2 font-medium">Dist</th>
                              <th className="text-right px-3 py-2 font-medium">Miss</th>
                              <th className="text-center px-3 py-2 font-medium">Skew</th>
                            </tr>
                          </thead>
                          <tbody>
                            {dataProfile.column_profiles.map((col) => (
                              <VariableRow
                                key={col.name}
                                profile={col}
                                sessionId={sessionId!}
                                onUpdate={handleVariableUpdate}
                              />
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : modifications.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full text-center px-6 py-12">
                        <Sparkles className="w-5 h-5 text-muted-foreground/40 mb-2" />
                        <p className="text-xs text-muted-foreground">
                          No modifications yet
                        </p>
                        <p className="text-[10px] text-muted-foreground/60 mt-1">
                          Changes to variables and data will appear here
                        </p>
                      </div>
                    ) : (
                      <div>
                        {modifications.map((mod) => (
                          <ModificationRow
                            key={mod.timestamp}
                            mod={mod}
                            onRevert={handleRevertModification}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>

            {/* Collapsed tab */}
            {collapsed && (
              <button
                onClick={() => setCollapsed(false)}
                className="flex-shrink-0 w-7 border-l border-border bg-accent/30 hover:bg-accent transition-colors flex flex-col items-center justify-center gap-2 text-muted-foreground hover:text-foreground"
                title="Expand variables panel"
              >
                <ChevronRight className="w-3.5 h-3.5 rotate-180" />
                <span
                  className="text-[9px] font-medium uppercase tracking-widest"
                  style={{ writingMode: "vertical-rl" }}
                >
                  Variables
                </span>
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
