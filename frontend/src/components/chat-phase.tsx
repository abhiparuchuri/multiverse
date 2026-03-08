"use client";

import { useState, useRef, useEffect } from "react";
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
} from "lucide-react";
import { Markdown } from "@/components/markdown";

const TYPE_OPTIONS = ["continuous", "binary", "count/ordinal", "categorical"];

function DistributionBar({ profile }: { profile: ColumnProfile }) {
  const bins = profile.histogram;
  if (!bins || bins.length === 0) {
    return <div className="w-16 h-4" />;
  }

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
          <span className="text-muted-foreground">{profile.distribution || profile.dtype}</span>
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
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

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
      // Auto-update variable table if the LLM made edits
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
    <div className="flex-1 flex flex-col h-full">
      <div className="border-b border-border px-6 flex items-center justify-between h-[57px]">
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

      <div className="flex-1 flex overflow-hidden">
        {/* Chat panel */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {chatMessages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-foreground text-background"
                      : "bg-accent text-foreground"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <Markdown content={msg.content} />
                  ) : (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-accent rounded-xl px-4 py-3">
                  <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-border p-4">
            <div className="flex gap-3">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) =>
                  e.key === "Enter" && !e.shiftKey && handleSend()
                }
                placeholder="Ask about your variables, distributions, or data quality..."
                className="flex-1 bg-accent rounded-lg px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="px-4 py-3 rounded-lg bg-foreground text-background hover:bg-foreground/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Variable table panel */}
        {dataProfile && (
          <div className="w-[380px] border-l border-border flex flex-col overflow-hidden">
            <div className="px-4 py-3 border-b border-border">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Variables
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {dataProfile.rows.toLocaleString()} rows &middot;{" "}
                {dataProfile.columns} columns &middot;{" "}
                {dataProfile.missing_total_pct}% missing
              </p>
            </div>
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
          </div>
        )}
      </div>
    </div>
  );
}
