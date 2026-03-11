"use client";

import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { useAppState, RegressionResult, CovariateRole, ClassifierResult, DistributionStat, ChatMessage } from "@/lib/store";
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
  dagMap,
  onClose,
}: {
  r: RegressionResult;
  plotMap?: Record<string, string>;
  dagMap?: Record<string, string>;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [dagCopied, setDagCopied] = useState(false);
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
  const dagFilename = dagMap?.[r.spec_id];

  const handleCopyDag = useCallback(async () => {
    if (!dagFilename) return;
    const url = plotUrl(dagFilename);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Failed to fetch DAG");
      const blob = await response.blob();
      if (typeof ClipboardItem !== "undefined") {
        await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
      } else {
        await navigator.clipboard.writeText(url);
      }
      setDagCopied(true);
      setTimeout(() => setDagCopied(false), 2000);
    } catch {
      await navigator.clipboard.writeText(url);
      setDagCopied(true);
      setTimeout(() => setDagCopied(false), 2000);
    }
  }, [dagFilename]);

  const handleDownloadDag = useCallback(async () => {
    if (!dagFilename) return;
    try {
      const resp = await fetch(plotUrl(dagFilename));
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `dag_${r.predictor}_${r.model_family.replace(/\s+/g, "_")}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch {
      window.open(plotUrl(dagFilename), "_blank");
    }
  }, [dagFilename, r.predictor, r.model_family]);

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

          {/* Covariate Roles */}
          {r.covariate_roles && r.covariate_roles.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Covariate Classification
              </p>
              <div className="border border-border rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-accent/30">
                      <th className="text-left px-4 py-2 font-medium text-muted-foreground text-xs">Variable</th>
                      <th className="text-left px-4 py-2 font-medium text-muted-foreground text-xs">Role</th>
                      <th className="text-right px-4 py-2 font-medium text-muted-foreground text-xs">Coeff. Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.covariate_roles.map((cr: CovariateRole) => {
                      const roleColors: Record<string, string> = {
                        confounder: "text-orange-500",
                        mediator: "text-blue-500",
                        precision: "text-muted-foreground",
                        neutral: "text-muted-foreground",
                      };
                      const roleBg: Record<string, string> = {
                        confounder: "bg-orange-500/10",
                        mediator: "bg-blue-500/10",
                        precision: "bg-accent",
                        neutral: "bg-accent",
                      };
                      return (
                        <tr key={cr.variable} className="border-b border-border last:border-0">
                          <td className="px-4 py-2 font-medium text-xs">{cr.variable}</td>
                          <td className="px-4 py-2">
                            <span className={`text-xs font-medium capitalize px-2 py-0.5 rounded-full ${roleColors[cr.role]} ${roleBg[cr.role]}`}>
                              {cr.role}
                            </span>
                          </td>
                          <td className={`px-4 py-2 text-right font-mono text-xs ${Math.abs(cr.coeff_change_pct) >= 10 ? (roleColors[cr.role] || "") : "text-muted-foreground"}`}>
                            {cr.coeff_change_pct > 0 ? "+" : ""}{cr.coeff_change_pct.toFixed(1)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-muted-foreground mt-1.5 leading-relaxed">
                Classification uses the change-in-estimate method. Covariates that shift the predictor coefficient by &ge;10% are classified as confounders or mediators based on their correlation pattern.
              </p>
            </div>
          )}

          {/* DAG */}
          {dagFilename && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Directed Acyclic Graph (DAG)
                </p>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={handleDownloadDag}
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg border border-border text-[11px] text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    Download
                  </button>
                  <button
                    onClick={handleCopyDag}
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg border border-border text-[11px] text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                  >
                    {dagCopied ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
                    {dagCopied ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>
              <div className="bg-card border border-border rounded-xl overflow-hidden">
                <img
                  src={plotUrl(dagFilename)}
                  alt={`DAG: ${r.predictor} → ${r.outcome}`}
                  className="w-full h-auto"
                />
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

// ── Classifier card & modal ────────────────────────────────────────────────

function ClassifierCard({
  cr,
  plotMap,
  onClick,
}: {
  cr: ClassifierResult;
  plotMap: Record<string, string>;
  onClick: () => void;
}) {
  const plotFile = plotMap[cr.spec_id];
  return (
    <div
      onClick={onClick}
      className="border border-border rounded-xl overflow-hidden hover:shadow-md transition-shadow cursor-pointer bg-card"
    >
      {plotFile ? (
        <img
          src={plotUrl(plotFile)}
          alt={`${cr.classifier} feature importance`}
          className="w-full bg-white"
          loading="lazy"
        />
      ) : (
        <div className="h-32 bg-accent/30 flex items-center justify-center text-xs text-muted-foreground">
          No plot available
        </div>
      )}
      <div className="px-4 py-3 space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium">{cr.classifier}</p>
          <span className="text-xs font-mono bg-accent px-2 py-0.5 rounded-full">
            {(cr.accuracy * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex gap-3 text-xs text-muted-foreground">
          {cr.auc != null && <span>AUC: {cr.auc.toFixed(3)}</span>}
          <span>n = {cr.n_obs}</span>
          <span>{cr.n_features} features</span>
        </div>
        <p className="text-[11px] text-muted-foreground">
          {cr.predictors_used.join(", ")}
          {cr.covariates_used.length > 0 && (
            <span className="text-muted-foreground/60"> + {cr.covariates_used.length} covariates</span>
          )}
        </p>
      </div>
    </div>
  );
}

function ClassifierModal({
  cr,
  plotMap,
  dagMap,
  onClose,
}: {
  cr: ClassifierResult;
  plotMap: Record<string, string>;
  dagMap: Record<string, string>;
  onClose: () => void;
}) {
  const plotFile = plotMap[cr.spec_id];
  const dagFile = dagMap[cr.spec_id];
  const [dagCopied, setDagCopied] = useState(false);

  const handleDownloadDag = useCallback(async () => {
    if (!dagFile) return;
    try {
      const res = await fetch(plotUrl(dagFile));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dag_${cr.classifier.replace(/\s+/g, "_")}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch { /* ignore */ }
  }, [dagFile, cr.classifier]);

  const handleCopyDag = useCallback(async () => {
    if (!dagFile) return;
    const url = plotUrl(dagFile);
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      if (typeof ClipboardItem !== "undefined") {
        await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
      } else {
        await navigator.clipboard.writeText(url);
      }
      setDagCopied(true);
      setTimeout(() => setDagCopied(false), 2000);
    } catch {
      await navigator.clipboard.writeText(url);
      setDagCopied(true);
      setTimeout(() => setDagCopied(false), 2000);
    }
  }, [dagFile]);

  const handleDownload = useCallback(async () => {
    if (!plotFile) return;
    try {
      const res = await fetch(plotUrl(plotFile));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${cr.classifier.replace(/\s+/g, "_")}_importance.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch { /* ignore */ }
  }, [plotFile, cr.classifier]);

  // Sort feature importance
  const sortedFeatures = useMemo(() => {
    return Object.entries(cr.feature_importance)
      .sort(([, a], [, b]) => b - a);
  }, [cr.feature_importance]);

  return (
    <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-6" onClick={onClose}>
      <div
        className="bg-card border border-border rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-card border-b border-border px-6 py-4 flex items-center justify-between z-10">
          <div>
            <h3 className="text-base font-semibold">{cr.classifier}</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              {cr.predictors_used.join(", ")}
              {cr.covariates_used.length > 0 && ` + ${cr.covariates_used.length} covariates`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {plotFile && (
              <button onClick={handleDownload} className="p-2 rounded-lg hover:bg-accent transition-colors" title="Download plot">
                <Download className="w-4 h-4 text-muted-foreground" />
              </button>
            )}
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-accent transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="px-6 py-4 space-y-6">
          {plotFile && (
            <img src={plotUrl(plotFile)} alt={`${cr.classifier} feature importance`} className="w-full rounded-lg bg-white" />
          )}

          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <tbody>
                <tr className="border-b border-border">
                  <td className="px-4 py-2 text-muted-foreground">Accuracy</td>
                  <td className="px-4 py-2 text-right font-mono">{(cr.accuracy * 100).toFixed(2)}% ± {(cr.accuracy_std * 100).toFixed(2)}%</td>
                </tr>
                {cr.auc != null && (
                  <tr className="border-b border-border">
                    <td className="px-4 py-2 text-muted-foreground">AUC</td>
                    <td className="px-4 py-2 text-right font-mono">{cr.auc.toFixed(4)}{cr.auc_std != null && ` ± ${cr.auc_std.toFixed(4)}`}</td>
                  </tr>
                )}
                {cr.precision != null && (
                  <tr className="border-b border-border">
                    <td className="px-4 py-2 text-muted-foreground">Precision</td>
                    <td className="px-4 py-2 text-right font-mono">{(cr.precision * 100).toFixed(2)}%</td>
                  </tr>
                )}
                {cr.recall != null && (
                  <tr className="border-b border-border">
                    <td className="px-4 py-2 text-muted-foreground">Recall</td>
                    <td className="px-4 py-2 text-right font-mono">{(cr.recall * 100).toFixed(2)}%</td>
                  </tr>
                )}
                <tr className="border-b border-border">
                  <td className="px-4 py-2 text-muted-foreground">Observations</td>
                  <td className="px-4 py-2 text-right font-mono">{cr.n_obs}</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-muted-foreground">Features</td>
                  <td className="px-4 py-2 text-right font-mono">{cr.n_features}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div>
            <h4 className="text-sm font-medium mb-2">Feature Importance</h4>
            <div className="border border-border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-accent/30">
                    <th className="text-left px-4 py-2 font-medium text-muted-foreground text-xs">Feature</th>
                    <th className="text-right px-4 py-2 font-medium text-muted-foreground text-xs">Importance</th>
                    <th className="px-4 py-2 w-1/3"></th>
                  </tr>
                </thead>
                <tbody>
                  {sortedFeatures.map(([name, value]) => (
                    <tr key={name} className="border-b border-border last:border-0">
                      <td className="px-4 py-2 font-medium text-xs">{name}</td>
                      <td className="px-4 py-2 text-right font-mono text-xs">{(value * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2">
                        <div className="w-full bg-accent rounded-full h-2">
                          <div
                            className="bg-foreground/60 h-2 rounded-full"
                            style={{ width: `${Math.min(value / (sortedFeatures[0]?.[1] || 1) * 100, 100)}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* DAG */}
          {dagFile && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium">Directed Acyclic Graph (DAG)</h4>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={handleDownloadDag}
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg border border-border text-[11px] text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    Download
                  </button>
                  <button
                    onClick={handleCopyDag}
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg border border-border text-[11px] text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                  >
                    {dagCopied ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
                    {dagCopied ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>
              <div className="border border-border rounded-xl overflow-hidden">
                <img src={plotUrl(dagFile)} alt={`DAG: ${cr.classifier}`} className="w-full h-auto bg-white" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ClassifiersTab({
  classifierResults,
  classifierPlotMap,
  classifierDagMap,
  outcomeType,
}: {
  classifierResults: ClassifierResult[];
  classifierPlotMap: Record<string, string>;
  classifierDagMap: Record<string, string>;
  outcomeType: string;
}) {
  const [selectedClf, setSelectedClf] = useState<ClassifierResult | null>(null);
  const [clfSearch, setClfSearch] = useState("");
  const [clfFilterModel, setClfFilterModel] = useState("all");
  const [clfViewMode, setClfViewMode] = useState<"list" | "grid">("grid");

  const clfModels = useMemo(() => {
    return Array.from(new Set(classifierResults.map((c) => c.classifier)));
  }, [classifierResults]);

  const filtered = useMemo(() => {
    return classifierResults.filter((cr) => {
      const q = clfSearch.toLowerCase();
      const matchSearch = !q || cr.classifier.toLowerCase().includes(q) ||
        cr.features.some((f) => f.toLowerCase().includes(q));
      const matchModel = clfFilterModel === "all" || cr.classifier === clfFilterModel;
      return matchSearch && matchModel;
    });
  }, [classifierResults, clfSearch, clfFilterModel]);

  if (classifierResults.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">No classifier results available.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
      {selectedClf && (
        <ClassifierModal cr={selectedClf} plotMap={classifierPlotMap} dagMap={classifierDagMap} onClose={() => setSelectedClf(null)} />
      )}

      <div className="flex gap-3 items-center">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            value={clfSearch}
            onChange={(e) => setClfSearch(e.target.value)}
            placeholder="Search classifiers, features..."
            className="w-full bg-accent rounded-lg pl-10 pr-4 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>
        <select
          value={clfFilterModel}
          onChange={(e) => setClfFilterModel(e.target.value)}
          className="bg-accent rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
        >
          <option value="all">All Classifiers</option>
          {clfModels.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <div className="ml-auto flex items-center border border-border rounded-lg overflow-hidden">
          <button
            onClick={() => setClfViewMode("list")}
            className={`p-2 transition-colors ${clfViewMode === "list" ? "bg-foreground text-background" : "bg-accent text-muted-foreground hover:text-foreground"}`}
            title="List view"
          >
            <List className="w-4 h-4" />
          </button>
          <button
            onClick={() => setClfViewMode("grid")}
            className={`p-2 transition-colors ${clfViewMode === "grid" ? "bg-foreground text-background" : "bg-accent text-muted-foreground hover:text-foreground"}`}
            title="Grid view"
          >
            <LayoutGrid className="w-4 h-4" />
          </button>
        </div>
      </div>

      {clfViewMode === "list" && (
        <div className="border border-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-accent/50">
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">Classifier</th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">Features</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Accuracy</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">AUC</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">n</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((cr) => (
                <tr
                  key={cr.spec_id}
                  onClick={() => setSelectedClf(cr)}
                  className="border-b border-border last:border-0 hover:bg-accent/40 transition-colors cursor-pointer"
                >
                  <td className="px-4 py-3 font-medium">{cr.classifier}</td>
                  <td className="px-4 py-3 text-muted-foreground text-xs">
                    {cr.predictors_used.join(", ")}
                    {cr.covariates_used.length > 0 && ` +${cr.covariates_used.length}`}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{(cr.accuracy * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{cr.auc != null ? cr.auc.toFixed(3) : "\u2014"}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{cr.n_obs}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">No results match your filters</div>
          )}
        </div>
      )}

      {clfViewMode === "grid" && (
        filtered.length === 0 ? (
          <div className="py-8 text-center text-sm text-muted-foreground">No results match your filters</div>
        ) : (
          <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
            {filtered.map((cr) => (
              <ClassifierCard key={cr.spec_id} cr={cr} plotMap={classifierPlotMap} onClick={() => setSelectedClf(cr)} />
            ))}
          </div>
        )
      )}
    </div>
  );
}

// ── Distributions tab ─────────────────────────────────────────────────────

function DistributionCard({
  stat,
  plotMap,
  onClick,
}: {
  stat: DistributionStat;
  plotMap: Record<string, string>;
  onClick: () => void;
}) {
  const plotFile = plotMap[stat.variable];
  const roleColors: Record<string, string> = {
    outcome: "text-purple-500",
    predictor: "text-green-500",
    covariate: "text-gray-500",
  };

  return (
    <div
      onClick={onClick}
      className="border border-border rounded-xl overflow-hidden hover:shadow-md transition-shadow cursor-pointer bg-card"
    >
      {plotFile ? (
        <img src={plotUrl(plotFile)} alt={`${stat.variable} distribution`} className="w-full bg-white" loading="lazy" />
      ) : (
        <div className="h-32 bg-accent/30 flex items-center justify-center text-xs text-muted-foreground">No plot</div>
      )}
      <div className="px-4 py-3 space-y-1">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium">{stat.variable}</p>
          <span className={`text-[11px] font-medium ${roleColors[stat.role] || "text-muted-foreground"}`}>
            {stat.role}
          </span>
        </div>
        <div className="flex gap-3 text-xs text-muted-foreground">
          <span>n = {stat.n}</span>
          {stat.mean != null && <span>μ = {stat.mean.toFixed(2)}</span>}
          {stat.std != null && <span>σ = {stat.std.toFixed(2)}</span>}
          {stat.missing_pct > 0 && <span>{stat.missing_pct}% missing</span>}
        </div>
      </div>
    </div>
  );
}

function DistributionModal({
  stat,
  plotMap,
  onClose,
}: {
  stat: DistributionStat;
  plotMap: Record<string, string>;
  onClose: () => void;
}) {
  const plotFile = plotMap[stat.variable];

  const handleDownload = useCallback(async () => {
    if (!plotFile) return;
    try {
      const res = await fetch(plotUrl(plotFile));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dist_${stat.variable}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch { /* ignore */ }
  }, [plotFile, stat.variable]);

  return (
    <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-6" onClick={onClose}>
      <div
        className="bg-card border border-border rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-card border-b border-border px-6 py-4 flex items-center justify-between z-10">
          <div>
            <h3 className="text-base font-semibold">{stat.variable}</h3>
            <p className="text-xs text-muted-foreground mt-0.5 capitalize">{stat.role}</p>
          </div>
          <div className="flex items-center gap-2">
            {plotFile && (
              <button onClick={handleDownload} className="p-2 rounded-lg hover:bg-accent transition-colors" title="Download plot">
                <Download className="w-4 h-4 text-muted-foreground" />
              </button>
            )}
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-accent transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="px-6 py-4 space-y-6">
          {plotFile && (
            <img src={plotUrl(plotFile)} alt={`${stat.variable} distribution`} className="w-full rounded-lg bg-white" />
          )}

          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <tbody>
                <tr className="border-b border-border">
                  <td className="px-4 py-2 text-muted-foreground">Count</td>
                  <td className="px-4 py-2 text-right font-mono">{stat.n}</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="px-4 py-2 text-muted-foreground">Missing</td>
                  <td className="px-4 py-2 text-right font-mono">{stat.missing} ({stat.missing_pct}%)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="px-4 py-2 text-muted-foreground">Unique values</td>
                  <td className="px-4 py-2 text-right font-mono">{stat.unique}</td>
                </tr>
                {stat.mean != null && (
                  <tr className="border-b border-border">
                    <td className="px-4 py-2 text-muted-foreground">Mean</td>
                    <td className="px-4 py-2 text-right font-mono">{stat.mean.toFixed(4)}</td>
                  </tr>
                )}
                {stat.std != null && (
                  <tr className="border-b border-border">
                    <td className="px-4 py-2 text-muted-foreground">Std Dev</td>
                    <td className="px-4 py-2 text-right font-mono">{stat.std.toFixed(4)}</td>
                  </tr>
                )}
                {stat.median != null && (
                  <tr className="border-b border-border">
                    <td className="px-4 py-2 text-muted-foreground">Median</td>
                    <td className="px-4 py-2 text-right font-mono">{stat.median.toFixed(4)}</td>
                  </tr>
                )}
                {stat.min != null && stat.max != null && (
                  <tr className="border-b border-border">
                    <td className="px-4 py-2 text-muted-foreground">Range</td>
                    <td className="px-4 py-2 text-right font-mono">[{stat.min.toFixed(4)}, {stat.max.toFixed(4)}]</td>
                  </tr>
                )}
                {stat.skewness != null && (
                  <tr>
                    <td className="px-4 py-2 text-muted-foreground">Skewness</td>
                    <td className="px-4 py-2 text-right font-mono">{stat.skewness.toFixed(4)}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DistributionsTab({
  distributionStats,
  distributionPlotMap,
}: {
  distributionStats: DistributionStat[];
  distributionPlotMap: Record<string, string>;
}) {
  const [selectedDist, setSelectedDist] = useState<DistributionStat | null>(null);
  const [distFilter, setDistFilter] = useState<string>("all");
  const [distViewMode, setDistViewMode] = useState<"list" | "grid">("grid");

  const filtered = useMemo(() => {
    if (distFilter === "all") return distributionStats;
    return distributionStats.filter((d) => d.role === distFilter);
  }, [distributionStats, distFilter]);

  if (distributionStats.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">No distribution data available.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
      {selectedDist && (
        <DistributionModal stat={selectedDist} plotMap={distributionPlotMap} onClose={() => setSelectedDist(null)} />
      )}

      <div className="flex gap-3 items-center">
        <select
          value={distFilter}
          onChange={(e) => setDistFilter(e.target.value)}
          className="bg-accent rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
        >
          <option value="all">All Variables</option>
          <option value="outcome">Outcome</option>
          <option value="predictor">Predictors</option>
          <option value="covariate">Covariates</option>
        </select>
        <div className="ml-auto flex items-center border border-border rounded-lg overflow-hidden">
          <button
            onClick={() => setDistViewMode("list")}
            className={`p-2 transition-colors ${distViewMode === "list" ? "bg-foreground text-background" : "bg-accent text-muted-foreground hover:text-foreground"}`}
            title="List view"
          >
            <List className="w-4 h-4" />
          </button>
          <button
            onClick={() => setDistViewMode("grid")}
            className={`p-2 transition-colors ${distViewMode === "grid" ? "bg-foreground text-background" : "bg-accent text-muted-foreground hover:text-foreground"}`}
            title="Grid view"
          >
            <LayoutGrid className="w-4 h-4" />
          </button>
        </div>
      </div>

      {distViewMode === "list" && (
        <div className="border border-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-accent/50">
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">Variable</th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">Role</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">n</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Mean</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Std</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Missing</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Skew</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d) => (
                <tr
                  key={d.variable}
                  onClick={() => setSelectedDist(d)}
                  className="border-b border-border last:border-0 hover:bg-accent/40 transition-colors cursor-pointer"
                >
                  <td className="px-4 py-3 font-medium">{d.variable}</td>
                  <td className="px-4 py-3 text-muted-foreground capitalize text-xs">{d.role}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{d.n}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{d.mean != null ? d.mean.toFixed(2) : "\u2014"}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{d.std != null ? d.std.toFixed(2) : "\u2014"}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{d.missing_pct}%</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">{d.skewness != null ? d.skewness.toFixed(2) : "\u2014"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {distViewMode === "grid" && (
        <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((d) => (
            <DistributionCard key={d.variable} stat={d} plotMap={distributionPlotMap} onClick={() => setSelectedDist(d)} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Sort types ─────────────────────────────────────────────────────────────

type SortField = "coefficient" | "p_value" | "effect_size" | "model_family";
type SortDir = "asc" | "desc";

// ── Main component ─────────────────────────────────────────────────────────

export function ResultsPhase() {
  const { results, sessionId } = useAppState();
  const [tab, setTab] = useState<"regressions" | "classifiers" | "distributions" | "chat">("regressions");
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
        <RegressionModal r={selectedReg} plotMap={plotMap} dagMap={results.dag_map} onClose={() => setSelectedReg(null)} />
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
            {(["regressions", "classifiers", "distributions", "chat"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`pb-3 pt-3 text-sm font-medium border-b-2 transition-colors ${
                  tab === t
                    ? "border-foreground text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {t === "regressions" ? "Regressions"
                  : t === "classifiers" ? "Classifiers"
                  : t === "distributions" ? "Distributions"
                  : "AI Chat"}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* AI Chat tab */}
          {tab === "chat" && (
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
          )}

          {/* Classifiers tab */}
          {tab === "classifiers" && (
            <ClassifiersTab
              classifierResults={results.classifier_results || []}
              classifierPlotMap={results.classifier_plot_map || {}}
              classifierDagMap={results.classifier_dag_map || {}}
              outcomeType={results.outcome_type}
            />
          )}

          {/* Distributions tab */}
          {tab === "distributions" && (
            <DistributionsTab
              distributionStats={results.distribution_stats || []}
              distributionPlotMap={results.distribution_plot_map || {}}
            />
          )}

          {/* Regressions tab */}
          {tab === "regressions" && (
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
