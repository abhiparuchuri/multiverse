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

export interface RegressionResult {
  spec_id: string;
  model_family: string;
  covariates: string[];
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

export interface AnalysisResults {
  session_id: string;
  summary: string;
  robustness_pct: number;
  total_specs: number;
  significant_specs: number;
  mean_effect_size: number;
  outcome_variable: string;
  outcome_type: string;
  regressions: RegressionResult[];
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
}

export const AppContext = createContext<AppState | null>(null);

export function useAppState() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppState must be used within AppProvider");
  return ctx;
}
