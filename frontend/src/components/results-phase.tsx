"use client";

import { useState, useMemo } from "react";
import { useAppState, RegressionResult } from "@/lib/store";
import {
  BarChart3,
  TrendingUp,
  Hash,
  ShieldCheck,
  Search,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useAppState as useTheme } from "@/lib/store";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

function StatCard({
  label,
  value,
  subtitle,
  icon: Icon,
}: {
  label: string;
  value: string;
  subtitle: string;
  icon: typeof BarChart3;
}) {
  return (
    <div className="bg-card border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-muted-foreground">{label}</p>
        <Icon className="w-4 h-4 text-muted-foreground" />
      </div>
      <p className="text-2xl font-semibold">{value}</p>
      <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
    </div>
  );
}

type SortField = "coefficient" | "p_value" | "effect_size" | "model_family";
type SortDir = "asc" | "desc";

export function ResultsPhase() {
  const { results, theme } = useAppState();
  const isDark = theme === "dark";
  const [tab, setTab] = useState<"overview" | "regressions">("overview");
  const [search, setSearch] = useState("");
  const [filterModel, setFilterModel] = useState<string>("all");
  const [filterSig, setFilterSig] = useState<string>("all");
  const [sortField, setSortField] = useState<SortField>("p_value");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  if (!results) return null;

  const specCurveData = useMemo(() => {
    return results.regressions
      .map((r, i) => ({
        index: i,
        coefficient: r.coefficient,
        significant: r.significant_corrected,
        model: r.model_family,
        predictor: r.predictor,
        p_value: r.p_value_corrected,
      }))
      .sort((a, b) => a.coefficient - b.coefficient)
      .map((d, i) => ({ ...d, index: i }));
  }, [results.regressions]);

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
        return sortDir === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
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
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      <div className="border-b border-border px-6 flex items-center h-[57px]">
        <div>
          <h2 className="text-sm font-semibold">Results</h2>
          <p className="text-xs text-muted-foreground">
            Multiverse analysis of {results.outcome_variable} ({results.outcome_type})
          </p>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-4 gap-4 px-6 py-4">
        <StatCard
          label="Robustness"
          value={`${results.robustness_pct}%`}
          subtitle="of specs support hypothesis"
          icon={ShieldCheck}
        />
        <StatCard
          label="Total Specifications"
          value={results.total_specs.toString()}
          subtitle={`${results.significant_specs} significant after FDR`}
          icon={Hash}
        />
        <StatCard
          label="Mean Effect Size"
          value={results.mean_effect_size.toFixed(3)}
          subtitle={
            Math.abs(results.mean_effect_size) < 0.2
              ? "Small effect"
              : Math.abs(results.mean_effect_size) < 0.5
              ? "Medium effect"
              : "Large effect"
          }
          icon={TrendingUp}
        />
        <StatCard
          label="Outcome Type"
          value={results.outcome_type}
          subtitle={results.outcome_variable}
          icon={BarChart3}
        />
      </div>

      {/* Tabs */}
      <div className="px-6 border-b border-border">
        <div className="flex gap-6">
          {(["overview", "regressions"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`pb-3 text-sm font-medium border-b-2 transition-colors capitalize ${
                tab === t
                  ? "border-foreground text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {t === "overview" ? "Overview" : `All Regressions (${results.regressions.length})`}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {tab === "overview" ? (
          <div className="space-y-6">
            {/* Summary */}
            <div className="bg-card border border-border rounded-xl p-6">
              <h3 className="text-sm font-medium mb-3">AI Summary</h3>
              <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                {results.summary}
              </p>
            </div>

            {/* Specification Curve */}
            <div className="bg-card border border-border rounded-xl p-6">
              <h3 className="text-sm font-medium mb-4">Specification Curve</h3>
              <p className="text-xs text-muted-foreground mb-4">
                Each dot is one model specification, sorted by effect size. Green = significant after FDR correction.
              </p>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#27272a" : "#e2e8f0"} />
                  <XAxis
                    dataKey="index"
                    type="number"
                    name="Specification"
                    tick={{ fontSize: 12, fill: isDark ? "#a1a1aa" : "#64748b" }}
                    label={{
                      value: "Specification (sorted by effect size)",
                      position: "bottom",
                      offset: 5,
                      style: { fontSize: 11, fill: isDark ? "#a1a1aa" : "#64748b" },
                    }}
                  />
                  <YAxis
                    dataKey="coefficient"
                    type="number"
                    name="Coefficient"
                    tick={{ fontSize: 12, fill: isDark ? "#a1a1aa" : "#64748b" }}
                    label={{
                      value: "Coefficient",
                      angle: -90,
                      position: "insideLeft",
                      style: { fontSize: 11, fill: isDark ? "#a1a1aa" : "#64748b" },
                    }}
                  />
                  <ReferenceLine y={0} stroke={isDark ? "#a1a1aa" : "#94a3b8"} strokeDasharray="5 5" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: isDark ? "#0a0a0f" : "#ffffff",
                      border: `1px solid ${isDark ? "#27272a" : "#e2e8f0"}`,
                      borderRadius: "8px",
                      fontSize: "12px",
                      color: isDark ? "#fafafa" : "#09090b",
                    }}
                    formatter={(value) => [
                      typeof value === "number" ? value.toFixed(4) : String(value),
                    ]}
                    labelFormatter={() => ""}
                  />
                  <Scatter data={specCurveData} name="Specifications">
                    {specCurveData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.significant ? "#22c55e" : isDark ? "#3f3f46" : "#cbd5e1"}
                        opacity={entry.significant ? 0.9 : 0.5}
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Filters */}
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
            </div>

            {/* Table */}
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
                      className="border-b border-border last:border-0 hover:bg-accent/30 transition-colors"
                    >
                      <td className="px-4 py-3 font-medium">{r.predictor}</td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {r.model_family}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground text-xs">
                        {r.covariates.join(", ") || "—"}
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
          </div>
        )}
      </div>
    </div>
  );
}
