"use client";

import { useEffect, useState } from "react";
import { useAppState } from "@/lib/store";
import { runAnalysis } from "@/lib/api";
import { Loader2 } from "lucide-react";

const STEPS = [
  "Checking assumptions...",
  "Testing normality (Shapiro-Wilk)...",
  "Checking homoscedasticity (Breusch-Pagan)...",
  "Computing VIF for multicollinearity...",
  "Running OLS regressions...",
  "Running penalized regressions (Lasso, Ridge)...",
  "Running logistic models...",
  "Computing effect measures (OR, RR, RD)...",
  "Applying transforms for violated assumptions...",
  "Applying FDR correction...",
  "Aggregating multiverse results...",
  "Generating summary with AI...",
];

export function AnalysisPhase() {
  const {
    sessionId,
    setResults,
    setPhase,
    completedPhases,
    setCompletedPhases,
  } = useAppState();
  const [stepIdx, setStepIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setStepIdx((prev) => (prev < STEPS.length - 1 ? prev + 1 : prev));
    }, 1800);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    runAnalysis(sessionId)
      .then((data) => {
        if (cancelled) return;
        setResults(data);
        setCompletedPhases([...completedPhases, "analysis"]);
        setPhase("results");
      })
      .catch(() => {
        if (!cancelled) setError("Analysis failed. Please try again.");
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const progress = ((stepIdx + 1) / STEPS.length) * 100;

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="w-full max-w-md text-center">
        <div className="relative mb-8">
          <div className="w-20 h-20 mx-auto rounded-full border-2 border-border flex items-center justify-center">
            <Loader2 className="w-8 h-8 text-foreground animate-spin" />
          </div>
        </div>

        <h2 className="text-xl font-semibold mb-2">Running Multiverse Analysis</h2>
        <p className="text-sm text-muted-foreground mb-8">
          Exhaustively testing all valid model specifications
        </p>

        {/* Progress bar */}
        <div className="w-full bg-accent rounded-full h-1.5 mb-4">
          <div
            className="bg-foreground h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Current step */}
        <p className="text-sm text-muted-foreground animate-pulse">
          {STEPS[stepIdx]}
        </p>

        {/* Step counter */}
        <p className="text-xs text-muted-foreground/60 mt-2">
          Step {stepIdx + 1} of {STEPS.length}
        </p>

        {error && (
          <div className="mt-6 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
