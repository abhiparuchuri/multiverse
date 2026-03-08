"use client";

import { useState, useEffect, useCallback, ReactNode } from "react";
import {
  AppContext,
  Phase,
  Theme,
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
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

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
        theme,
        toggleTheme,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
