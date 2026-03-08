"use client";

import { useState, ReactNode } from "react";
import {
  AppContext,
  Phase,
  DataProfile,
  ChatMessage,
  AnalysisResults,
  DataModification,
} from "@/lib/store";

export function AppProvider({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>("import");
  const [completedPhases, setCompletedPhases] = useState<Phase[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataProfile, setDataProfile] = useState<DataProfile | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [modifications, setModifications] = useState<DataModification[]>([]);

  const addModification = (mod: DataModification) => {
    setModifications((prev) => [...prev, mod]);
  };

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
        modifications,
        setModifications,
        addModification,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
