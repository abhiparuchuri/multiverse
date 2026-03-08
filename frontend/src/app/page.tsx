"use client";

import { Sidebar } from "@/components/sidebar";
import { ImportPhase } from "@/components/import-phase";
import { ChatPhase } from "@/components/chat-phase";
import { IntentPhase } from "@/components/intent-phase";
import { AnalysisPhase } from "@/components/analysis-phase";
import { ResultsPhase } from "@/components/results-phase";
import { useAppState } from "@/lib/store";

export default function Home() {
  const { phase } = useAppState();

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        {phase === "import" && <ImportPhase />}
        {phase === "variables" && <ChatPhase />}
        {phase === "intent" && <IntentPhase />}
        {phase === "analysis" && <AnalysisPhase />}
        {phase === "results" && <ResultsPhase />}
      </main>
    </div>
  );
}
