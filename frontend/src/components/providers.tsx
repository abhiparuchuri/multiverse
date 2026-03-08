"use client";

import { useState, ReactNode } from "react";
import {
  AppContext,
  Phase,
  DataProfile,
  ChatMessage,
  AnalysisResults,
} from "@/lib/store";

export function AppProvider({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>("import");
  const [completedPhases, setCompletedPhases] = useState<Phase[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataProfile, setDataProfile] = useState<DataProfile | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [columns, setColumns] = useState<string[]>([]);

  return (
    <AppContext.Provider
      value={{
        phase,
        setPhase,
        completedPhases,
        setCompletedPhases,
        sessionId,
        setSessionId,
        dataProfile,
        setDataProfile,
        chatMessages,
        setChatMessages,
        results,
        setResults,
        columns,
        setColumns,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
