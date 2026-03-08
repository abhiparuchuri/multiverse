"use client";

import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { useAppState, RegressionResult, ChatMessage } from "@/lib/store";
import { sendResultsChatStream } from "@/lib/api";
import {
  Search,
  ChevronDown,
  ChevronUp,
  LayoutGrid,
  List,
  X,
  Copy,
  Check,
  Download,
  Send,
  Loader2,
} from "lucide-react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  ErrorBar,
} from "recharts";
import { Markdown } from "@/components/markdown";

const API_BASE = "http://localhost:8000";

function plotUrl(filename: string): string {
  return `${API_BASE}/plots/${filename}`;
}

// ── Forest plot fallback (Recharts, used when no server image) ─────────────

function ForestPlotFallback({
  r,
  compact = false,
}: {
  r: RegressionResult;
  compact?: boolean;
}) {
  const data = [
    {
      name: r.predictor,
      coeff: r.coefficient,
      errorY: [
        r.coefficient - r.ci_lower,
        r.ci_upper - r.coefficient,
      ] as [number, number],
    },
  ];

  const absMax = Math.max(
    Math.abs(r.ci_lower),
    Math.abs(r.ci_upper),
    Math.abs(r.coefficient)
  );
  const domain: [number, number] = [-absMax * 1.3, absMax * 1.3];
  const height = compact ? 80 : 120;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 5, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="coeff"
          type="number"
          domain={domain}
          tick={{ fontSize: compact ? 9 : 11, fill: "#64748b" }}
          tickFormatter={(v) => v.toFixed(2)}
        />
        <YAxis hide />
        <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="5 5" />
        <Scatter data={data} fill={r.significant_corrected ? "#22c55e" : "#94a3b8"}>
          <ErrorBar
            dataKey="errorY"
            width={compact ? 4 : 6}
            strokeWidth={compact ? 1.5 : 2}
            stroke={r.significant_corrected ? "#22c55e" : "#94a3b8"}
            direction="x"
          />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ── Plot Image with fallback ──────────────────────────────────────────────

function PlotImage({
  specId,
  plotMap,
  regression,
  compact = false,
}: {
  specId: string;
  plotMap?: Record<string, string>;
  regression: RegressionResult;
  compact?: boolean;
}) {
  const filename = plotMap?.[specId];
  if (!filename) {
    return <ForestPlotFallback r={regression} compact={compact} />;
  }

  return (
    <img
      src={plotUrl(filename)}
      alt={`Forest plot: ${regression.predictor} → ${regression.outcome}`}
      className="w-full h-auto"
      loading="lazy"
    />
  );
}

// ── Regression Detail Modal ────────────────────────────────────────────────

function RegressionModal({
  r,
  plotMap,
  onClose,
}: {
  r: RegressionResult;
  plotMap?: Record<string, string>;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const tableRef = useRef<HTMLDivElement>(null);

  const pLabel =
    r.p_value_corrected < 0.001
      ? "<0.001"
      : r.p_value_corrected.toFixed(4);

  const statsRows = [
    ["Coefficient", r.coefficient.toFixed(4)],
    ["95% CI", `[${r.ci_lower.toFixed(3)}, ${r.ci_upper.toFixed(3)}]`],
    ["p-value (raw)", r.p_value.toFixed(4)],
    ["p-value (FDR corrected)", pLabel],
    ["Effect size", r.effect_size.toFixed(4)],
    ["N observations", r.n_obs.toString()],
    ...(r.r_squared != null ? [["R\u00B2", r.r_squared.toFixed(4)]] : []),
    ...(r.aic != null ? [["AIC", r.aic.toFixed(2)]] : []),
    ["Model family", r.model_family],
    ["Assumptions met", r.assumptions_met ? "Yes" : "No"],
  ];

  const assumptionRows = Object.entries(r.assumption_details || {}).map(
    ([k, v]) => [k.replace(/_/g, " "), String(v)]
  );

  const handleCopyImage = useCallback(async () => {
    const filename = plotMap?.[r.spec_id];
    if (!filename) return;
    const url = plotUrl(filename);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Failed to fetch image");
      const blob = await response.blob();
      // ClipboardItem is not universally supported; URL fallback below handles that path.
      if (typeof ClipboardItem !== "undefined") {
        await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
      } else {
        await navigator.clipboard.writeText(url);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [plotMap, r.spec_id]);

  const handleDownloadPlot = useCallback(async () => {
    const filename = plotMap?.[r.spec_id];
    if (!filename) return;
    try {
      const resp = await fetch(plotUrl(filename));
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${r.predictor}_${r.model_family.replace(/\s+/g, "_")}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch {
      // fallback: open in new tab
      window.open(plotUrl(filename), "_blank");
    }
  }, [plotMap, r.spec_id, r.predictor, r.model_family]);

  const hasPlotImage = !!plotMap?.[r.spec_id];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-background border border-border rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-start justify-between px-6 py-4 border-b border-border">
          <div>
            <div className="flex items-center gap-2">
              <span
                className={`inline-block w-2 h-2 rounded-full ${
                  r.significant_corrected ? "bg-success" : "bg-muted-foreground/40"
                }`}
              />
              <h3 className="text-sm font-semibold">{r.predictor}</h3>
              <span className="text-xs text-muted-foreground">&rarr; {r.outcome}</span>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              {r.model_family}
              {r.covariates.length > 0 && ` · adjusted for ${r.covariates.join(", ")}`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {hasPlotImage && (
              <button
                onClick={handleDownloadPlot}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              >
                <Download className="w-3 h-3" />
                Download plot
              </button>
            )}
            {hasPlotImage && (
              <button
                onClick={handleCopyImage}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              >
                {copied ? (
                  <Check className="w-3 h-3 text-success" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
                {copied ? "Copied!" : "Copy image"}
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5" ref={tableRef}>
          {/* Forest plot */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
              Coefficient &amp; 95% Confidence Interval
            </p>
            <div className="bg-card border border-border rounded-xl overflow-hidden">
              <PlotImage
                specId={r.spec_id}
                plotMap={plotMap}
                regression={r}
                compact={false}
              />
            </div>
          </div>

          {/* Stats table */}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
              Statistics
            </p>
            <div className="border border-border rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <tbody>
                  {statsRows.map(([k, v], i) => (
                    <tr
                      key={k}
                      className={`border-b border-border last:border-0 ${
                        i % 2 === 0 ? "bg-card" : "bg-background"
                      }`}
                    >
                      <td className="px-4 py-2.5 text-muted-foreground font-medium w-1/2">
                        {k}
                      </td>
                      <td
                        className={`px-4 py-2.5 font-mono text-right ${
                          k.includes("p-value") && r.p_value_corrected < 0.05
                            ? "text-success font-semibold"
                            : ""
                        }`}
                      >
                        {v}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Assumption details */}
          {assumptionRows.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Assumption Checks
              </p>
              <div className="border border-border rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <tbody>
                    {assumptionRows.map(([k, v], i) => (
                      <tr
                        key={k}
                        className={`border-b border-border last:border-0 ${
                          i % 2 === 0 ? "bg-card" : "bg-background"
                        }`}
                      >
                        <td className="px-4 py-2.5 text-muted-foreground font-medium capitalize w-1/2">
                          {k}
                        </td>
                        <td className="px-4 py-2.5 font-mono text-right text-xs">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Publication note */}
          <div className="bg-accent/50 rounded-xl px-4 py-3">
            <p className="text-xs text-muted-foreground">
              <span className="font-medium text-foreground">Publication note:</span>{" "}
              {hasPlotImage
                ? "Use \"Download plot\" for a 200 DPI publication-ready PNG. Use \"Copy image\" to copy the figure to your clipboard."
                : "Use \"Download plot\" when available to export the figure for reports or manuscripts."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Grid card for a single regression ─────────────────────────────────────

function GridCard({
  r,
  plotMap,
  onClick,
}: {
  r: RegressionResult;
  plotMap?: Record<string, string>;
  onClick: () => void;
}) {
  const pLabel =
    r.p_value_corrected < 0.001 ? "<0.001" : r.p_value_corrected.toFixed(3);

  const hasPlotImage = !!plotMap?.[r.spec_id];

  return (
    <button
      onClick={onClick}
      className="text-left bg-card border border-border rounded-xl overflow-hidden hover:border-foreground/30 hover:shadow-sm transition-all flex flex-col"
    >
      {/* Plot image or fallback */}
      <div className={hasPlotImage ? "border-b border-border" : "px-4 pt-3"}>
        {hasPlotImage ? (
          <img
            src={plotUrl(plotMap![r.spec_id])}
            alt={`${r.predictor} → ${r.outcome}`}
            className="w-full h-auto"
            loading="lazy"
          />
        ) : (
          <div className="bg-background rounded-lg border border-border px-1">
            <ForestPlotFallback r={r} compact />
          </div>
        )}
      </div>

      {/* Stats below plot */}
      <div className="p-4 flex flex-col gap-2.5">
        {/* Title row */}
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="flex items-center gap-1.5">
              <span
                className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${
                  r.significant_corrected ? "bg-success" : "bg-muted-foreground/30"
                }`}
              />
              <span className="text-sm font-medium leading-tight">{r.predictor}</span>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5 ml-3.5">{r.model_family}</p>
          </div>
          <span
            className={`text-xs font-mono px-2 py-0.5 rounded-full flex-shrink-0 ${
              r.significant_corrected
                ? "bg-success/10 text-success"
                : "bg-muted text-muted-foreground"
            }`}
          >
            p={pLabel}
          </span>
        </div>

        {/* Key stats */}
        <div className="grid grid-cols-3 gap-2">
          {[
            ["Coeff", r.coefficient.toFixed(3)],
            ["Effect", r.effect_size.toFixed(3)],
            ["N", r.n_obs.toString()],
          ].map(([label, val]) => (
            <div key={label} className="bg-accent/60 rounded-lg px-2 py-1.5 text-center">
              <p className="text-[10px] text-muted-foreground">{label}</p>
              <p className="text-xs font-mono font-medium">{val}</p>
            </div>
          ))}
        </div>

        {r.covariates.length > 0 && (
          <p className="text-[10px] text-muted-foreground truncate">
            Adjusted for: {r.covariates.join(", ")}
          </p>
        )}
      </div>
    </button>
  );
}

// ── Sort types ─────────────────────────────────────────────────────────────

type SortField = "coefficient" | "p_value" | "effect_size" | "model_family";
type SortDir = "asc" | "desc";

// ── Main component ─────────────────────────────────────────────────────────

export function ResultsPhase() {
  const { results, sessionId } = useAppState();
  const [tab, setTab] = useState<"regressions" | "chat">("regressions");
  const [viewMode, setViewMode] = useState<"list" | "grid">("grid");
  const [search, setSearch] = useState("");
  const [filterModel, setFilterModel] = useState<string>("all");
  const [filterSig, setFilterSig] = useState<string>("all");
  const [sortField, setSortField] = useState<SortField>("p_value");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [selectedReg, setSelectedReg] = useState<RegressionResult | null>(null);

  // AI Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  useEffect(() => {
    const el = chatInputRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxH = 140;
    el.style.height = `${Math.min(el.scrollHeight, maxH)}px`;
    el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden";
  }, [chatInput]);

  const handleChatSend = async () => {
    if (!chatInput.trim() || !sessionId || chatLoading) return;
    const userMsg: ChatMessage = { role: "user", content: chatInput.trim() };
    const updated = [...chatMessages, userMsg];
    setChatMessages([...updated, { role: "assistant", content: "" }]);
    setChatInput("");
    setChatLoading(true);

    try {
      let streamed = "";
      const data = await sendResultsChatStream(sessionId, userMsg.content, updated, {
        onChunk: (chunk) => {
          streamed += chunk;
          setChatMessages([...updated, { role: "assistant", content: streamed }]);
        },
      });
      setChatMessages([
        ...updated,
        { role: "assistant", content: String(data.response || streamed) },
      ]);
    } catch {
      setChatMessages([
        ...updated,
        { role: "assistant", content: "Sorry, something went wrong. Please try again." },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  if (!results) return null;

  const plotMap = results.plot_map;

  const modelFamilies = useMemo(() => {
    return [...new Set(results.regressions.map((r) => r.model_family))];
  }, [results.regressions]);

  const filteredRegressions = useMemo(() => {
    let filtered = results.regressions;
    if (search) {
      const q = search.toLowerCase();
      filtered = filtered.filter(
        (r) =>
          r.predictor.toLowerCase().includes(q) ||
          r.model_family.toLowerCase().includes(q) ||
          r.covariates.some((c) => c.toLowerCase().includes(q))
      );
    }
    if (filterModel !== "all") {
      filtered = filtered.filter((r) => r.model_family === filterModel);
    }
    if (filterSig === "significant") {
      filtered = filtered.filter((r) => r.significant_corrected);
    } else if (filterSig === "not_significant") {
      filtered = filtered.filter((r) => !r.significant_corrected);
    }
    filtered = [...filtered].sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDir === "asc"
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number);
    });
    return filtered;
  }, [results.regressions, search, filterModel, filterSig, sortField, sortDir]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDir === "asc" ? (
      <ChevronUp className="w-3 h-3 inline ml-1" />
    ) : (
      <ChevronDown className="w-3 h-3 inline ml-1" />
    );
  };

  return (
    <>
      {selectedReg && (
        <RegressionModal r={selectedReg} plotMap={plotMap} onClose={() => setSelectedReg(null)} />
      )}

      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <div className="border-b border-border px-6 flex items-center h-[57px]">
          <div>
            <h2 className="text-sm font-semibold">Results</h2>
            <p className="text-xs text-muted-foreground">
              Multiverse analysis of {results.outcome_variable} ({results.outcome_type})
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="px-6 border-b border-border">
          <div className="flex gap-6">
            {(["regressions", "chat"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`pb-3 pt-3 text-sm font-medium border-b-2 transition-colors ${
                  tab === t
                    ? "border-foreground text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {t === "chat"
                  ? "AI Chat"
                  : `All Regressions (${results.regressions.length})`}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {tab === "chat" ? (
            <div className="flex-1 flex flex-col overflow-hidden">
              <div className="flex-1 overflow-y-auto py-6">
                <div className="max-w-2xl mx-auto px-6 space-y-6">
                  {chatMessages.length === 0 && (
                    <div className="text-center py-12">
                      <p className="text-sm text-muted-foreground">
                        Ask questions about your analysis results — robustness, effect sizes, model agreement, and more.
                      </p>
                    </div>
                  )}
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
                  {chatLoading && chatMessages[chatMessages.length - 1]?.content === "" && (
                    <div className="flex justify-start">
                      <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
              </div>

              <div className="p-4 flex-shrink-0">
                <div className="max-w-2xl mx-auto">
                  <div className="flex items-end gap-2 bg-accent border border-border rounded-2xl px-4 py-2 focus-within:ring-1 focus-within:ring-ring transition-shadow">
                    <textarea
                      ref={chatInputRef}
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleChatSend();
                        }
                      }}
                      placeholder="Ask about robustness, effect sizes, model agreement..."
                      rows={1}
                      className="flex-1 resize-none bg-transparent text-sm leading-relaxed text-foreground placeholder:text-muted-foreground focus:outline-none py-1"
                    />
                    <button
                      onClick={handleChatSend}
                      disabled={!chatInput.trim() || chatLoading}
                      className="p-1.5 rounded-lg bg-foreground text-background hover:bg-foreground/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
                    >
                      <Send className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
              {/* Filters + view toggle */}
              <div className="flex gap-3 items-center">
                <div className="relative flex-1 max-w-sm">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search predictors, models..."
                    className="w-full bg-accent rounded-lg pl-10 pr-4 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                  />
                </div>
                <select
                  value={filterModel}
                  onChange={(e) => setFilterModel(e.target.value)}
                  className="bg-accent rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                >
                  <option value="all">All Models</option>
                  {modelFamilies.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <select
                  value={filterSig}
                  onChange={(e) => setFilterSig(e.target.value)}
                  className="bg-accent rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                >
                  <option value="all">All Results</option>
                  <option value="significant">Significant Only</option>
                  <option value="not_significant">Not Significant</option>
                </select>

                {/* View toggle */}
                <div className="ml-auto flex items-center border border-border rounded-lg overflow-hidden">
                  <button
                    onClick={() => setViewMode("list")}
                    className={`p-2 transition-colors ${
                      viewMode === "list"
                        ? "bg-foreground text-background"
                        : "bg-accent text-muted-foreground hover:text-foreground"
                    }`}
                    title="List view"
                  >
                    <List className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`p-2 transition-colors ${
                      viewMode === "grid"
                        ? "bg-foreground text-background"
                        : "bg-accent text-muted-foreground hover:text-foreground"
                    }`}
                    title="Grid view"
                  >
                    <LayoutGrid className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* List view */}
              {viewMode === "list" && (
                <div className="border border-border rounded-xl overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border bg-accent/50">
                        <th className="text-left px-4 py-3 font-medium text-muted-foreground">
                          Predictor
                        </th>
                        <th
                          className="text-left px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                          onClick={() => toggleSort("model_family")}
                        >
                          Model
                          <SortIcon field="model_family" />
                        </th>
                        <th className="text-left px-4 py-3 font-medium text-muted-foreground">
                          Covariates
                        </th>
                        <th
                          className="text-right px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                          onClick={() => toggleSort("coefficient")}
                        >
                          Coeff
                          <SortIcon field="coefficient" />
                        </th>
                        <th
                          className="text-right px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                          onClick={() => toggleSort("p_value")}
                        >
                          p-value
                          <SortIcon field="p_value" />
                        </th>
                        <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                          95% CI
                        </th>
                        <th
                          className="text-right px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                          onClick={() => toggleSort("effect_size")}
                        >
                          Effect
                          <SortIcon field="effect_size" />
                        </th>
                        <th className="text-center px-4 py-3 font-medium text-muted-foreground">
                          Sig
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredRegressions.map((r) => (
                        <tr
                          key={r.spec_id}
                          onClick={() => setSelectedReg(r)}
                          className="border-b border-border last:border-0 hover:bg-accent/40 transition-colors cursor-pointer"
                        >
                          <td className="px-4 py-3 font-medium">{r.predictor}</td>
                          <td className="px-4 py-3 text-muted-foreground">
                            {r.model_family}
                          </td>
                          <td className="px-4 py-3 text-muted-foreground text-xs">
                            {r.covariates.join(", ") || "\u2014"}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs">
                            {r.coefficient.toFixed(4)}
                          </td>
                          <td
                            className={`px-4 py-3 text-right font-mono text-xs ${
                              r.p_value_corrected < 0.05
                                ? "text-success"
                                : "text-muted-foreground"
                            }`}
                          >
                            {r.p_value_corrected < 0.001
                              ? "<0.001"
                              : r.p_value_corrected.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs text-muted-foreground">
                            [{r.ci_lower.toFixed(3)}, {r.ci_upper.toFixed(3)}]
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs">
                            {r.effect_size.toFixed(3)}
                          </td>
                          <td className="px-4 py-3 text-center">
                            <span
                              className={`inline-block w-2 h-2 rounded-full ${
                                r.significant_corrected
                                  ? "bg-success"
                                  : "bg-muted-foreground/30"
                              }`}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {filteredRegressions.length === 0 && (
                    <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                      No results match your filters
                    </div>
                  )}
                </div>
              )}

              {/* Grid view */}
              {viewMode === "grid" && (
                <>
                  {filteredRegressions.length === 0 ? (
                    <div className="py-8 text-center text-sm text-muted-foreground">
                      No results match your filters
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
                      {filteredRegressions.map((r) => (
                        <GridCard
                          key={r.spec_id}
                          r={r}
                          plotMap={plotMap}
                          onClick={() => setSelectedReg(r)}
                        />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
