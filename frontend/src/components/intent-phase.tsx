"use client";

import { useState, useRef, useEffect } from "react";
import { useAppState, ChatMessage } from "@/lib/store";
import { sendIntentChat } from "@/lib/api";
import { Send, Loader2, Lock } from "lucide-react";
import { Markdown } from "@/components/markdown";

export function IntentPhase() {
  const {
    sessionId,
    setPhase,
    completedPhases,
    setCompletedPhases,
  } = useAppState();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [intentReady, setIntentReady] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Send initial message to kick off the intent conversation
  useEffect(() => {
    if (!sessionId || !initializing) return;

    sendIntentChat(sessionId, "__init__", [])
      .then((data) => {
        setMessages([{ role: "assistant", content: data.response }]);
        setInitializing(false);
      })
      .catch(() => {
        setMessages([
          {
            role: "assistant",
            content:
              "Let's define your study. What is the primary outcome you're investigating, and what variables do you think predict it?",
          },
        ]);
        setInitializing(false);
      });
  }, [sessionId]);

  const handleSend = async () => {
    if (!input.trim() || !sessionId || loading) return;
    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    try {
      const data = await sendIntentChat(sessionId, userMsg.content, updatedMessages);
      setMessages([
        ...updatedMessages,
        { role: "assistant", content: data.response },
      ]);
      if (data.intent_ready) {
        setIntentReady(true);
      }
    } catch {
      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleCommit = async () => {
    if (!sessionId) return;
    setLoading(true);

    try {
      // Send a commit message to finalize intent
      const data = await sendIntentChat(sessionId, "__commit__", messages);
      if (data.committed) {
        setCompletedPhases([...completedPhases, "intent"]);
        setPhase("analysis");
      }
    } catch {
      // fallback
      setCompletedPhases([...completedPhases, "intent"]);
      setPhase("analysis");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="border-b border-border px-6 flex items-center justify-between h-[57px]">
        <div>
          <h2 className="text-sm font-semibold">Study Intent</h2>
          <p className="text-xs text-muted-foreground">
            Define your hypothesis — this becomes your pre-registration
          </p>
        </div>
        {intentReady && (
          <button
            onClick={handleCommit}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-foreground text-background text-sm font-medium hover:bg-foreground/90 disabled:opacity-50 transition-colors"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Lock className="w-4 h-4" />
            )}
            Commit & Run Analysis
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto py-6">
        <div className="max-w-2xl mx-auto px-6 space-y-6">
          {initializing && (
            <div className="flex justify-start">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            </div>
          )}
          {messages.map((msg, i) => (
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
          {loading && (
            <div className="flex justify-start">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="p-4">
        <div className="max-w-2xl mx-auto">
          <div className="flex items-center gap-2 bg-accent border border-border rounded-2xl px-4 py-2 focus-within:ring-1 focus-within:ring-ring transition-shadow">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
              placeholder="Describe your research question, outcome variable, hypothesized predictors..."
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none py-1"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="p-1.5 rounded-lg bg-foreground text-background hover:bg-foreground/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
          {intentReady && (
            <p className="text-xs text-center text-muted-foreground mt-2">
              Your study intent is ready. Click &quot;Commit & Run Analysis&quot; to lock it in and begin.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
