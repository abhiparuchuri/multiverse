const API_BASE = "http://localhost:8000";

export async function uploadCSV(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

export async function sendChatMessage(sessionId: string, message: string) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  return res.json();
}

type StreamEvent = {
  type: "chunk" | "final" | "error";
  content?: string;
  error?: string;
  [key: string]: unknown;
};

export type ChatStreamFinal = {
  type?: "final";
  response: string;
  profile_updated?: boolean;
  profile?: unknown;
  modifications?: unknown[];
};

export type IntentStreamFinal = {
  type?: "final";
  response?: string;
  intent_ready?: boolean;
  committed?: boolean;
  intent?: unknown;
};

async function consumeNdjsonStream(
  res: Response,
  onEvent: (event: StreamEvent) => void
) {
  if (!res.body) {
    throw new Error("Streaming not supported by the browser.");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newlineIdx = buffer.indexOf("\n");
    while (newlineIdx !== -1) {
      const line = buffer.slice(0, newlineIdx).trim();
      buffer = buffer.slice(newlineIdx + 1);
      if (line) {
        onEvent(JSON.parse(line) as StreamEvent);
      }
      newlineIdx = buffer.indexOf("\n");
    }
  }

  const finalLine = buffer.trim();
  if (finalLine) {
    onEvent(JSON.parse(finalLine) as StreamEvent);
  }
}

export async function sendChatMessageStream(
  sessionId: string,
  message: string,
  handlers: { onChunk?: (chunk: string) => void } = {}
): Promise<ChatStreamFinal> {
  const res = await fetch(`${API_BASE}/chat-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) {
    throw new Error(`Chat stream failed with status ${res.status}`);
  }

  let finalData: ChatStreamFinal | null = null;
  let streamError: string | null = null;

  await consumeNdjsonStream(res, (event) => {
    if (event.type === "chunk") {
      handlers.onChunk?.(String(event.content || ""));
      return;
    }
    if (event.type === "error") {
      streamError = String(event.error || "Unknown stream error");
      return;
    }
    if (event.type === "final") {
      finalData = event as unknown as ChatStreamFinal;
    }
  });

  if (streamError) throw new Error(streamError);
  if (!finalData) throw new Error("Chat stream ended without final payload.");
  return finalData;
}

export async function updateVariable(
  sessionId: string,
  originalName: string,
  updates: Record<string, unknown>
) {
  const res = await fetch(`${API_BASE}/update-variable`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      original_name: originalName,
      updates,
    }),
  });
  return res.json();
}

export async function revertModification(
  sessionId: string,
  timestamp: number
) {
  const res = await fetch(`${API_BASE}/revert-modification`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      timestamp,
    }),
  });
  return res.json();
}

export async function sendIntentChat(
  sessionId: string,
  message: string,
  chatHistory: { role: string; content: string }[]
) {
  const res = await fetch(`${API_BASE}/intent-chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      chat_history: chatHistory,
    }),
  });
  return res.json();
}

export async function sendIntentChatStream(
  sessionId: string,
  message: string,
  chatHistory: { role: string; content: string }[],
  handlers: { onChunk?: (chunk: string) => void } = {}
): Promise<IntentStreamFinal> {
  const res = await fetch(`${API_BASE}/intent-chat-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      chat_history: chatHistory,
    }),
  });
  if (!res.ok) {
    throw new Error(`Intent stream failed with status ${res.status}`);
  }

  let finalData: IntentStreamFinal | null = null;
  let streamError: string | null = null;

  await consumeNdjsonStream(res, (event) => {
    if (event.type === "chunk") {
      handlers.onChunk?.(String(event.content || ""));
      return;
    }
    if (event.type === "error") {
      streamError = String(event.error || "Unknown stream error");
      return;
    }
    if (event.type === "final") {
      finalData = event as unknown as IntentStreamFinal;
    }
  });

  if (streamError) throw new Error(streamError);
  if (!finalData) throw new Error("Intent stream ended without final payload.");
  return finalData;
}

export type ResultsChatStreamFinal = {
  type?: "final";
  response: string;
};

export async function sendResultsChatStream(
  sessionId: string,
  message: string,
  chatHistory: { role: string; content: string }[],
  handlers: { onChunk?: (chunk: string) => void } = {}
): Promise<ResultsChatStreamFinal> {
  const res = await fetch(`${API_BASE}/results-chat-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      chat_history: chatHistory,
    }),
  });
  if (!res.ok) {
    throw new Error(`Results chat stream failed with status ${res.status}`);
  }

  let finalData: ResultsChatStreamFinal | null = null;
  let streamError: string | null = null;

  await consumeNdjsonStream(res, (event) => {
    if (event.type === "chunk") {
      handlers.onChunk?.(String(event.content || ""));
      return;
    }
    if (event.type === "error") {
      streamError = String(event.error || "Unknown stream error");
      return;
    }
    if (event.type === "final") {
      finalData = event as unknown as ResultsChatStreamFinal;
    }
  });

  if (streamError) throw new Error(streamError);
  if (!finalData) throw new Error("Results chat stream ended without final payload.");
  return finalData;
}

export async function runAnalysis(sessionId: string) {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return res.json();
}

export async function getResults(sessionId: string) {
  const res = await fetch(`${API_BASE}/results/${sessionId}`);
  return res.json();
}
