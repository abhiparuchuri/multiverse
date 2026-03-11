"use client";

import { createContext, useContext } from "react";

export type Phase = "import" | "variables" | "intent" | "analysis" | "results";

export interface ColumnProfile {
  name: string;
  dtype: string;
  unique: number;
  missing: number;
  missing_pct: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  skewness?: number;
  is_skewed?: boolean;
  distribution?: string;
  histogram?: number[];
}

export interface DataProfile {
  rows: number;
  columns: number;
  column_profiles: ColumnProfile[];
  missing_total_pct: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface CovariateRole {
  variable: string;
  role: "confounder" | "mediator" | "precision" | "neutral";
  coeff_change_pct: number;
}

export interface RegressionResult {
  spec_id: string;
  model_family: string;
  covariates: string[];
  covariate_roles?: CovariateRole[];
  outcome: string;
  predictor: string;
  coefficient: number;
  p_value: number;
  p_value_corrected: number;
  ci_lower: number;
  ci_upper: number;
  effect_size: number;
  r_squared?: number;
  aic?: number;
  n_obs: number;
  significant: boolean;
  significant_corrected: boolean;
  assumptions_met: boolean;
  assumption_details: Record<string, unknown>;
}

export interface ClassifierResult {
  spec_id: string;
  classifier: string;
  features: string[];
  predictors_used: string[];
  covariates_used: string[];
  accuracy: number;
  accuracy_std: number;
  auc: number | null;
  auc_std: number | null;
  precision: number | null;
  recall: number | null;
  feature_importance: Record<string, number>;
  n_obs: number;
  n_features: number;
}

export interface ClassifierSummary {
  total_specs?: number;
  best_classifier?: string | null;
  best_accuracy?: number | null;
  mean_accuracy?: number;
  mean_auc?: number | null;
}

export interface DistributionStat {
  variable: string;
  role: "outcome" | "predictor" | "covariate";
  n: number;
  missing: number;
  missing_pct: number;
  unique: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  median?: number;
  skewness?: number;
}

export interface AnalysisResults {
  session_id: string;
  robustness_pct: number;
  total_specs: number;
  significant_specs: number;
  mean_effect_size: number;
  outcome_variable: string;
  outcome_type: string;
  regressions: RegressionResult[];
  plot_map?: Record<string, string>;
  classifier_results?: ClassifierResult[];
  classifier_summary?: ClassifierSummary;
  classifier_plot_map?: Record<string, string>;
  dag_map?: Record<string, string>;
  classifier_dag_map?: Record<string, string>;
  distribution_plot_map?: Record<string, string>;
  distribution_stats?: DistributionStat[];
}

export interface DataModification {
  type: "rename" | "retype" | "transform";
  variable: string;
  from: string;
  to: string;
  source: "user" | "ai";
  timestamp: number;
  description?: string;
  code?: string;
}

export interface AppState {
  phase: Phase;
  setPhase: (phase: Phase) => void;
  completedPhases: Phase[];
  setCompletedPhases: (phases: Phase[]) => void;
  sessionId: string | null;
  setSessionId: (id: string) => void;
  dataProfile: DataProfile | null;
  setDataProfile: (profile: DataProfile) => void;
  chatMessages: ChatMessage[];
  setChatMessages: (messages: ChatMessage[]) => void;
  results: AnalysisResults | null;
  setResults: (results: AnalysisResults) => void;
  columns: string[];
  setColumns: (columns: string[]) => void;
  modifications: DataModification[];
  setModifications: (mods: DataModification[]) => void;
  addModification: (mod: DataModification) => void;
}

export const AppContext = createContext<AppState | null>(null);

export function useAppState() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppState must be used within AppProvider");
  return ctx;
}
