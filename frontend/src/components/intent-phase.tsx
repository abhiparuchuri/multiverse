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

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {initializing && (
          <div className="flex justify-start">
            <div className="bg-accent rounded-xl px-4 py-3">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[75%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-foreground text-background"
                  : "bg-accent text-foreground"
              }`}
            >
              {msg.role === "assistant" ? (
                <Markdown content={msg.content} />
              ) : (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-accent rounded-xl px-4 py-3">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-border p-4">
        <div className="flex gap-3 max-w-3xl mx-auto">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Describe your research question, outcome variable, hypothesized predictors..."
            className="flex-1 bg-accent rounded-lg px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="px-4 py-3 rounded-lg bg-foreground text-background hover:bg-foreground/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        {intentReady && (
          <p className="text-xs text-center text-muted-foreground mt-2">
            Your study intent is ready. Click &quot;Commit & Run Analysis&quot; to lock it in and begin.
          </p>
        )}
      </div>
    </div>
  );
}
