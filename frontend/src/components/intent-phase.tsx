"use client";

import { useState, useRef, useEffect } from "react";
import { useAppState, ChatMessage, ColumnProfile, DataModification } from "@/lib/store";
import { sendIntentChatStream, IntentDraft } from "@/lib/api";
import { Send, Loader2, Lock, AlertTriangle, ChevronRight, Sparkles, ArrowLeftRight, Type, Filter, User, Target, FlaskConical, Shield, X, Plus } from "lucide-react";
import { Markdown } from "@/components/markdown";

type PanelTab = "variables" | "modifications";
type DesignTab = "outcome" | "predictors" | "covariates";

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

function ReadOnlyVariableRow({ profile }: { profile: ColumnProfile }) {
  return (
    <tr className="border-b border-border last:border-0 hover:bg-accent/30 transition-colors text-xs">
      <td className="px-3 py-2 font-medium">{profile.name}</td>
      <td className="px-3 py-2 text-muted-foreground">{profile.distribution || profile.dtype}</td>
      <td className="px-3 py-2 text-center">
        <DistributionBar profile={profile} />
      </td>
      <td className="px-3 py-2 text-right text-muted-foreground">{profile.missing_pct}%</td>
      <td className="px-3 py-2 text-center">
        {profile.is_skewed && <AlertTriangle className="w-3 h-3 text-warning inline" />}
      </td>
    </tr>
  );
}

function ReadOnlyModificationRow({ mod }: { mod: DataModification }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-border last:border-0 hover:bg-accent/30 transition-colors text-xs">
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
            <p className="font-medium leading-snug break-words">{mod.description || mod.from}</p>
            <p className="text-[10px] text-muted-foreground mt-0.5">→ {mod.to}</p>
          </>
        ) : (
          <p className="font-medium leading-snug break-words">
            {mod.type === "rename" ? "Renamed" : "Retyped"}{" "}
            <span className="text-muted-foreground">{mod.from}</span> {" → "} <span>{mod.to}</span>
          </p>
        )}
        <p className="text-[10px] text-muted-foreground flex items-center gap-1 mt-0.5">
          {mod.source === "ai" ? <Sparkles className="w-2.5 h-2.5" /> : <User className="w-2.5 h-2.5" />}
          {mod.source === "ai" ? "AI suggested" : "Manual"}
        </p>
      </div>
    </div>
  );
}

function VariableChip({
  name,
  onRemove,
}: {
  name: string;
  onRemove?: () => void;
}) {
  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] bg-foreground text-background font-medium">
      {name}
      {onRemove && (
        <button
          onClick={onRemove}
          className="hover:bg-background/20 rounded-full p-0.5 transition-colors"
        >
          <X className="w-2.5 h-2.5" />
        </button>
      )}
    </span>
  );
}

function AddVariableDropdown({
  options,
  onAdd,
}: {
  options: string[];
  onAdd: (name: string) => void;
}) {
  const [open, setOpen] = useState(false);

  if (options.length === 0) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-[11px] border border-dashed border-border text-muted-foreground hover:text-foreground hover:border-foreground/40 transition-colors"
      >
        <Plus className="w-2.5 h-2.5" />
        Add
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 z-10 bg-popover border border-border rounded-lg shadow-lg py-1 max-h-40 overflow-y-auto min-w-[140px]">
          {options.map((name) => (
            <button
              key={name}
              onClick={() => {
                onAdd(name);
                setOpen(false);
              }}
              className="w-full text-left px-3 py-1.5 text-[11px] hover:bg-accent transition-colors"
            >
              {name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function StudyDesignPanel({
  intentDraft,
  designTab,
  setDesignTab,
  allColumns,
  onUpdateDraft,
  onCollapse,
}: {
  intentDraft: IntentDraft | null;
  designTab: DesignTab;
  setDesignTab: (tab: DesignTab) => void;
  allColumns: string[];
  onUpdateDraft: (draft: IntentDraft) => void;
  onCollapse: () => void;
}) {
  const outcome = intentDraft?.outcome_variable || "";
  const predictors = intentDraft?.predictors || [];
  const covariates = intentDraft?.confounders || [];

  const usedColumns = new Set([outcome, ...predictors, ...covariates].filter(Boolean));
  const availableForOutcome = allColumns.filter((c) => c !== outcome && !predictors.includes(c));
  const availableForPredictors = allColumns.filter((c) => !usedColumns.has(c));
  const availableForCovariates = allColumns.filter((c) => !usedColumns.has(c));

  const setOutcome = (name: string) => {
    onUpdateDraft({
      ...intentDraft,
      outcome_variable: name,
      predictors: predictors.filter((p) => p !== name),
      confounders: covariates.filter((c) => c !== name),
    });
  };

  const addPredictor = (name: string) => {
    onUpdateDraft({
      ...intentDraft,
      outcome_variable: outcome,
      predictors: [...predictors, name],
      confounders: covariates.filter((c) => c !== name),
    });
  };

  const removePredictor = (name: string) => {
    onUpdateDraft({
      ...intentDraft,
      outcome_variable: outcome,
      predictors: predictors.filter((p) => p !== name),
      confounders: covariates,
    });
  };

  const addCovariate = (name: string) => {
    onUpdateDraft({
      ...intentDraft,
      outcome_variable: outcome,
      predictors: predictors.filter((p) => p !== name),
      confounders: [...covariates, name],
    });
  };

  const removeCovariate = (name: string) => {
    onUpdateDraft({
      ...intentDraft,
      outcome_variable: outcome,
      predictors,
      confounders: covariates.filter((c) => c !== name),
    });
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 pt-2 pb-0 flex-shrink-0 flex items-center justify-between">
        <p className="text-[11px] font-semibold text-foreground uppercase tracking-wider">Study Design</p>
        <button
          onClick={onCollapse}
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="Collapse panel"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
      <div className="flex px-4 mt-2 gap-1">
        {([
          { key: "outcome" as DesignTab, label: "Outcome", icon: Target, count: outcome ? 1 : 0 },
          { key: "predictors" as DesignTab, label: "Predictors", icon: FlaskConical, count: predictors.length },
          { key: "covariates" as DesignTab, label: "Covariates", icon: Shield, count: covariates.length },
        ]).map(({ key, label, count }) => (
          <button
            key={key}
            onClick={() => setDesignTab(key)}
            className={`px-3 py-1.5 text-[11px] font-medium rounded-t transition-colors border-b-2 flex items-center gap-1.5 ${
              designTab === key
                ? "border-foreground text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {label}
            {count > 0 && (
              <span className="bg-foreground text-background text-[9px] font-bold rounded-full w-4 h-4 flex items-center justify-center">
                {count}
              </span>
            )}
          </button>
        ))}
      </div>
      <div className="border-t border-border px-4 py-3 flex-1 overflow-y-auto">
        {designTab === "outcome" && (
          <div>
            {outcome ? (
              <div className="flex items-center gap-2">
                <VariableChip name={outcome} onRemove={() => onUpdateDraft({ ...intentDraft, outcome_variable: "", predictors, confounders: covariates })} />
              </div>
            ) : (
              <p className="text-[11px] text-muted-foreground">Describe your outcome in the chat</p>
            )}
          </div>
        )}
        {designTab === "predictors" && (
          <div>
            {predictors.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {predictors.map((name) => (
                  <VariableChip key={name} name={name} onRemove={() => removePredictor(name)} />
                ))}
                <AddVariableDropdown options={availableForPredictors} onAdd={addPredictor} />
              </div>
            ) : (
              <p className="text-[11px] text-muted-foreground">Describe your predictors in the chat</p>
            )}
          </div>
        )}
        {designTab === "covariates" && (
          <div>
            {covariates.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {covariates.map((name) => (
                  <VariableChip key={name} name={name} onRemove={() => removeCovariate(name)} />
                ))}
                <AddVariableDropdown options={availableForCovariates} onAdd={addCovariate} />
              </div>
            ) : (
              <p className="text-[11px] text-muted-foreground">Covariates will be identified from the chat</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function IntentPhase() {
  const {
    sessionId,
    setPhase,
    completedPhases,
    setCompletedPhases,
    dataProfile,
    modifications,
  } = useAppState();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [intentReady, setIntentReady] = useState(false);
  const [intentDraft, setIntentDraft] = useState<IntentDraft | null>(null);
  const [initializing, setInitializing] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const [panelTab, setPanelTab] = useState<PanelTab>("variables");
  const [designTab, setDesignTab] = useState<DesignTab>("outcome");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const allColumns = dataProfile
    ? dataProfile.column_profiles.map((c) => c.name)
    : [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxHeight = 180;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [input]);

  // Send initial message to kick off the intent conversation
  useEffect(() => {
    if (!sessionId || !initializing) return;

    setMessages([{ role: "assistant", content: "" }]);
    sendIntentChatStream(sessionId, "__init__", [], undefined, {
      onChunk: (chunk) => {
        setMessages((prev) => {
          const current = prev[0]?.content || "";
          return [{ role: "assistant", content: current + chunk }];
        });
      },
    })
      .then((data) => {
        setMessages([{ role: "assistant", content: String(data.response || "") }]);
        setInitializing(false);
      })
      .catch(() => {
        setMessages([
          {
            role: "assistant",
            content:
              "Let's define your study. What is the primary outcome you're investigating, and what variables do you think predict it?",
          },
        ]);
        setInitializing(false);
      });
  }, [sessionId]);

  const handleSend = async () => {
    if (!input.trim() || !sessionId || loading) return;
    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    const updatedMessages = [...messages, userMsg];
    setMessages([...updatedMessages, { role: "assistant", content: "" }]);
    setInput("");
    setLoading(true);

    try {
      let streamed = "";
      const data = await sendIntentChatStream(sessionId, userMsg.content, updatedMessages, undefined, {
        onChunk: (chunk) => {
          streamed += chunk;
          setMessages([...updatedMessages, { role: "assistant", content: streamed }]);
        },
      });
      setMessages([
        ...updatedMessages,
        { role: "assistant", content: String(data.response || streamed) },
      ]);
      setIntentReady(Boolean(data.intent_ready));
      if (data.intent_draft) {
        setIntentDraft(data.intent_draft);
      }
    } catch {
      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleCommit = async () => {
    if (!sessionId) return;
    setLoading(true);

    try {
      const data = await sendIntentChatStream(
        sessionId,
        "__commit__",
        messages,
        intentDraft || undefined,
        {}
      );
      if (data.committed) {
        setCompletedPhases([...completedPhases, "intent"]);
        setPhase("analysis");
      }
    } catch {
      setCompletedPhases([...completedPhases, "intent"]);
      setPhase("analysis");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      <div className="border-b border-border px-6 flex items-center justify-between h-[57px]">
        <div>
          <h2 className="text-sm font-semibold">Study Intent</h2>
          <p className="text-xs text-muted-foreground">
            Define your outcome, predictors, and covariates
          </p>
        </div>
        {intentReady && (
          <button
            onClick={handleCommit}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-foreground text-background text-sm font-medium hover:bg-foreground/90 disabled:opacity-50 transition-colors"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Lock className="w-4 h-4" />
            )}
            Run Analysis
          </button>
        )}
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto hide-scrollbar py-6">
            <div className="max-w-2xl mx-auto px-6 space-y-6">
              {initializing && (
                <div className="flex justify-start">
                  <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                </div>
              )}
              {messages.map((msg, i) => (
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
              <div className="flex items-end gap-2 bg-accent border border-border rounded-2xl px-4 py-2 focus-within:ring-1 focus-within:ring-ring transition-shadow">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Describe your research question, outcome variable, hypothesized predictors..."
                  rows={1}
                  className="flex-1 resize-none bg-transparent text-sm leading-relaxed text-foreground placeholder:text-muted-foreground focus:outline-none py-1"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || loading}
                  className="p-1.5 rounded-lg bg-foreground text-background hover:bg-foreground/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
              {intentReady && (
                <p className="text-xs text-center text-muted-foreground mt-2">
                  Your study intent is ready. Click &quot;Run Analysis&quot; to begin.
                </p>
              )}
            </div>
          </div>
        </div>

        {dataProfile && (
          <>
            {!collapsed && (
              <div className="w-1.5 flex-shrink-0 bg-border" />
            )}
            <div
              className="flex flex-col border-l border-border overflow-hidden transition-[width] duration-200"
              style={{ width: collapsed ? 0 : 460, flexShrink: 0 }}
            >
              {!collapsed && (
                <>
                  {/* Top half: Study Design tracker */}
                  <div className="basis-1/2 grow-0 shrink-0 min-h-0 flex flex-col border-b border-border">
                    <StudyDesignPanel
                      intentDraft={intentDraft}
                      designTab={designTab}
                      setDesignTab={setDesignTab}
                      allColumns={allColumns}
                      onUpdateDraft={setIntentDraft}
                      onCollapse={() => setCollapsed(true)}
                    />
                  </div>

                  {/* Bottom half: Data Profile (Variables / Modifications) */}
                  <div className="basis-1/2 grow-0 shrink-0 min-h-0 flex flex-col">
                    <div className="flex-shrink-0">
                      <div className="flex items-center justify-between px-4 pt-3 pb-0">
                        <p className="text-xs text-muted-foreground">
                          {dataProfile.rows.toLocaleString()} rows &middot; {dataProfile.columns} cols &middot;{" "}
                          {dataProfile.missing_total_pct}% missing
                        </p>
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

                    <div className="flex-1 overflow-y-auto border-t border-border">
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
                              <ReadOnlyVariableRow key={col.name} profile={col} />
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : modifications.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full text-center px-6 py-12">
                        <Sparkles className="w-5 h-5 text-muted-foreground/40 mb-2" />
                        <p className="text-xs text-muted-foreground">No modifications yet</p>
                        <p className="text-[10px] text-muted-foreground/60 mt-1">
                          Changes to variables and data will appear here
                        </p>
                      </div>
                    ) : (
                      <div>
                        {modifications.map((mod) => (
                          <ReadOnlyModificationRow key={mod.timestamp} mod={mod} />
                        ))}
                      </div>
                    )}
                    </div>
                  </div>
                </>
              )}
            </div>

            {collapsed && (
              <button
                onClick={() => setCollapsed(false)}
                className="flex-shrink-0 w-7 border-l border-border bg-accent/30 hover:bg-accent transition-colors flex flex-col items-center justify-center gap-2 text-muted-foreground hover:text-foreground"
                title="Expand panel"
              >
                <ChevronRight className="w-3.5 h-3.5 rotate-180" />
                <span
                  className="text-[9px] font-medium uppercase tracking-widest"
                  style={{ writingMode: "vertical-rl" }}
                >
                  Design
                </span>
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
