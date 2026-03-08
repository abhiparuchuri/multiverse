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
