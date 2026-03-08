import uuid
import io
import re
import json
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analysis_engine import run_multiverse_analysis
from llm import generate_data_profile_chat, generate_chat_response, generate_intent_chat, extract_intent_from_conversation, generate_summary

app = FastAPI(title="Multiverse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: dict = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class IntentChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: list[dict]


class UpdateVariableRequest(BaseModel):
    session_id: str
    original_name: str
    updates: dict


class AnalyzeRequest(BaseModel):
    session_id: str


def profile_column(series: pd.Series) -> dict:
    """Profile a single column."""
    profile = {
        "name": series.name,
        "dtype": str(series.dtype),
        "unique": int(series.nunique()),
        "missing": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 1),
    }

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            profile["mean"] = round(float(clean.mean()), 4)
            profile["std"] = round(float(clean.std()), 4)
            profile["min"] = round(float(clean.min()), 4)
            profile["max"] = round(float(clean.max()), 4)
            skew_val = float(clean.skew())
            profile["skewness"] = round(skew_val, 4)
            profile["is_skewed"] = abs(skew_val) > 1.0
            # Infer distribution type
            if clean.nunique() == 2:
                profile["distribution"] = "binary"
                counts = clean.value_counts().sort_index()
                profile["histogram"] = [int(c) for c in counts.values]
            elif clean.nunique() < 10 and clean.dtype in ["int64", "int32"]:
                profile["distribution"] = "count/ordinal"
                counts = clean.value_counts().sort_index()
                profile["histogram"] = [int(c) for c in counts.values]
            else:
                profile["distribution"] = "continuous"
                hist_counts, _ = np.histogram(clean, bins=12)
                profile["histogram"] = [int(c) for c in hist_counts]
    else:
        profile["distribution"] = "categorical"
        counts = series.dropna().value_counts().sort_index()
        # Limit to top 12 categories
        if len(counts) > 12:
            counts = counts.nlargest(12).sort_index()
        profile["histogram"] = [int(c) for c in counts.values]

    return profile


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Profile the entire dataframe."""
    column_profiles = [profile_column(df[col]) for col in df.columns]
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_profiles": column_profiles,
        "missing_total_pct": round(float(df.isna().mean().mean() * 100), 1),
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file and get a data profile."""
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    session_id = str(uuid.uuid4())
    profile = profile_dataframe(df)

    # Generate initial chat message from LLM
    chat_messages = generate_data_profile_chat(profile)

    sessions[session_id] = {
        "df": df,
        "profile": profile,
        "chat_history": chat_messages,
        "intent": None,
    }

    return {
        "session_id": session_id,
        "profile": profile,
        "chat_messages": chat_messages,
    }


def _parse_variable_edits(response: str) -> tuple[str, list[dict]]:
    """Extract variable_edits JSON block from LLM response. Returns (clean_response, edits)."""
    pattern = r"```variable_edits\s*\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return response, []
    try:
        edits = json.loads(match.group(1).strip())
        clean_response = response[:match.start()].rstrip()
        return clean_response, edits
    except (json.JSONDecodeError, TypeError):
        return response, []


def _apply_variable_edits(session: dict, edits: list[dict]) -> None:
    """Apply variable edits to session profile and dataframe."""
    profile = session["profile"]
    for edit in edits:
        original_name = edit.get("original_name")
        updates = edit.get("updates", {})
        if not original_name or not updates:
            continue
        for col in profile["column_profiles"]:
            if col["name"] == original_name:
                new_name = updates.get("name", col["name"])
                if new_name != col["name"]:
                    session["df"] = session["df"].rename(columns={col["name"]: new_name})
                    col["name"] = new_name
                if "distribution" in updates:
                    col["distribution"] = updates["distribution"]
                break


@app.post("/chat")
async def chat(request: ChatRequest):
    """Send a chat message about the data."""
    session = sessions.get(request.session_id)
    if not session:
        return {"error": "Session not found"}

    response = generate_chat_response(
        session["profile"],
        session["chat_history"],
        request.message,
    )

    # Parse and apply any variable edits from the LLM response
    clean_response, edits = _parse_variable_edits(response)
    if edits:
        _apply_variable_edits(session, edits)

    session["chat_history"].append({"role": "user", "content": request.message})
    session["chat_history"].append({"role": "assistant", "content": clean_response})

    result = {"response": clean_response}
    if edits:
        result["profile_updated"] = True
        result["profile"] = session["profile"]
    return result


@app.post("/update-variable")
async def update_variable(request: UpdateVariableRequest):
    """Update a variable's name or type in the session."""
    session = sessions.get(request.session_id)
    if not session:
        return {"error": "Session not found"}

    profile = session["profile"]
    updates = request.updates

    for col in profile["column_profiles"]:
        if col["name"] == request.original_name:
            # Rename column in dataframe if name changed
            new_name = updates.get("name", col["name"])
            if new_name != col["name"]:
                session["df"] = session["df"].rename(
                    columns={col["name"]: new_name}
                )
                col["name"] = new_name

            # Update distribution type if changed
            if "distribution" in updates:
                col["distribution"] = updates["distribution"]
            break

    return {"status": "updated", "profile": profile}


@app.post("/intent-chat")
async def intent_chat(request: IntentChatRequest):
    """Chat-based study intent definition."""
    session = sessions.get(request.session_id)
    if not session:
        return {"error": "Session not found"}

    profile = session["profile"]
    columns = [c["name"] for c in profile["column_profiles"]]

    # Handle init message — kick off the conversation
    if request.message == "__init__":
        response = generate_intent_chat(
            profile=profile,
            columns=columns,
            chat_history=[],
            user_message=None,
        )
        return {"response": response, "intent_ready": False}

    # Handle commit — extract structured intent from conversation
    if request.message == "__commit__":
        intent = extract_intent_from_conversation(
            profile=profile,
            columns=columns,
            chat_history=request.chat_history,
        )
        intent["timestamp"] = pd.Timestamp.now().isoformat()
        session["intent"] = intent
        return {"committed": True, "intent": intent}

    # Regular chat message
    response = generate_intent_chat(
        profile=profile,
        columns=columns,
        chat_history=request.chat_history,
        user_message=request.message,
    )

    # Check if we have enough info to show the commit button
    intent_ready = _check_intent_completeness(request.chat_history + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": response},
    ])

    return {"response": response, "intent_ready": intent_ready}


def _check_intent_completeness(chat_history: list[dict]) -> bool:
    """Simple heuristic: intent is ready if conversation has enough back-and-forth."""
    user_messages = [m for m in chat_history if m.get("role") == "user"]
    # Need at least 2 substantive user messages
    if len(user_messages) < 2:
        return False
    # Check if total content length suggests enough detail
    total_content = " ".join(m["content"] for m in user_messages)
    return len(total_content) > 80


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Run the multiverse analysis."""
    session = sessions.get(request.session_id)
    if not session:
        return {"error": "Session not found"}

    intent = session["intent"]
    if not intent:
        return {"error": "No study intent submitted"}

    df = session["df"]

    try:
        results = run_multiverse_analysis(
            df=df,
            outcome=intent["outcome_variable"],
            predictors=intent["predictors"],
            confounders=intent["confounders"],
            hypothesis=intent["hypothesis"],
        )

        # Generate LLM summary
        summary = generate_summary(results, intent)
        results["summary"] = summary
        results["session_id"] = request.session_id

        return results
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """Get cached results (if any)."""
    session = sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    return session.get("results", {"error": "No results yet"})
