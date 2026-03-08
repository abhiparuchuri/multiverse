"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useAppState, ChatMessage, ColumnProfile } from "@/lib/store";
import { sendChatMessage, updateVariable } from "@/lib/api";
import {
  Send,
  Loader2,
  ArrowRight,
  Pencil,
  Check,
  X,
  AlertTriangle,
  ChevronRight,
} from "lucide-react";
import { Markdown } from "@/components/markdown";

const TYPE_OPTIONS = ["continuous", "binary", "count/ordinal", "categorical"];
const PANEL_MIN = 240;
const PANEL_MAX = 600;
const PANEL_DEFAULT = 360;

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
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState(profile.name);
  const [editType, setEditType] = useState(profile.distribution || "continuous");

  const handleSave = () => {
    const updates: Partial<ColumnProfile> = {};
    if (editName !== profile.name) updates.name = editName;
    if (editType !== profile.distribution) updates.distribution = editType;
    if (Object.keys(updates).length > 0) {
      onUpdate(profile.name, updates);
      updateVariable(sessionId, profile.name, updates).catch(() => {});
    }
    setEditing(false);
  };

  const handleCancel = () => {
    setEditName(profile.name);
    setEditType(profile.distribution || "continuous");
    setEditing(false);
  };

  return (
    <tr className="border-b border-border last:border-0 hover:bg-accent/30 transition-colors text-xs">
      <td className="px-3 py-2">
        {editing ? (
          <input
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            className="bg-background border border-border rounded px-2 py-1 text-xs w-full focus:outline-none focus:ring-1 focus:ring-ring"
            autoFocus
          />
        ) : (
          <span className="font-medium">{profile.name}</span>
        )}
      </td>
      <td className="px-3 py-2">
        {editing ? (
          <select
            value={editType}
            onChange={(e) => setEditType(e.target.value)}
            className="bg-background border border-border rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          >
            {TYPE_OPTIONS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        ) : (
          <span className="text-muted-foreground">
            {profile.distribution || profile.dtype}
          </span>
        )}
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
      <td className="px-3 py-2 text-right">
        {editing ? (
          <div className="flex gap-1 justify-end">
            <button
              onClick={handleSave}
              className="p-1 rounded hover:bg-accent text-success"
            >
              <Check className="w-3 h-3" />
            </button>
            <button
              onClick={handleCancel}
              className="p-1 rounded hover:bg-accent text-muted-foreground"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ) : (
          <button
            onClick={() => setEditing(true)}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          >
            <Pencil className="w-3 h-3" />
          </button>
        )}
      </td>
    </tr>
  );
}

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
  } = useAppState();
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [panelWidth, setPanelWidth] = useState(PANEL_DEFAULT);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
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
                  {/* Panel header */}
                  <div className="px-4 py-3 border-b border-border flex items-center justify-between flex-shrink-0">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Variables
                      </p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {dataProfile.rows.toLocaleString()} rows &middot;{" "}
                        {dataProfile.columns} cols &middot;{" "}
                        {dataProfile.missing_total_pct}% missing
                      </p>
                    </div>
                    <button
                      onClick={() => setCollapsed(true)}
                      className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
                      title="Collapse panel"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Table */}
                  <div className="flex-1 overflow-y-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-border bg-accent/30 text-[10px] uppercase tracking-wider text-muted-foreground">
                          <th className="text-left px-3 py-2 font-medium">Name</th>
                          <th className="text-left px-3 py-2 font-medium">Type</th>
                          <th className="text-center px-3 py-2 font-medium">Dist</th>
                          <th className="text-right px-3 py-2 font-medium">Miss</th>
                          <th className="text-center px-3 py-2 font-medium">Skew</th>
                          <th className="text-right px-3 py-2 font-medium">Edit</th>
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
