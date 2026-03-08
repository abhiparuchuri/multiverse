import uuid
import io
import re
import json
import asyncio
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles
from pathlib import Path

from analysis_engine import run_multiverse_analysis
from llm import generate_data_profile_chat, generate_chat_response, generate_intent_chat, extract_intent_from_conversation, generate_summary
from plot_generator import generate_all_plots, PLOT_DIR

app = FastAPI(title="Multiverse API")

# Serve generated plot images
app.mount("/plots", StaticFiles(directory=str(PLOT_DIR)), name="plots")

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
        "df_original": df.copy(),
        "profile": profile,
        "chat_history": chat_messages,
        "intent": None,
        "modifications": [],
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


def _parse_data_transform(response: str) -> tuple[str, Optional[dict]]:
    """Extract data_transform JSON block from LLM response. Returns (clean_response, transform)."""
    pattern = r"```data_transform\s*\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return response, None
    try:
        transform = json.loads(match.group(1).strip())
        clean_response = response[:match.start()].rstrip()
        return clean_response, transform
    except (json.JSONDecodeError, TypeError):
        return response, None


# Allowlist of safe names for transform execution
_TRANSFORM_SAFE_BUILTINS = {
    "abs": abs, "len": len, "min": min, "max": max,
    "round": round, "int": int, "float": float, "str": str,
    "bool": bool, "list": list, "dict": dict, "tuple": tuple,
    "range": range, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "sum": sum, "any": any, "all": all,
    "True": True, "False": False, "None": None,
}


def _execute_transform(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute a pandas transform code string safely. Returns the modified df."""
    safe_globals = {
        "__builtins__": _TRANSFORM_SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "stats": stats,
        "df": df.copy(),
    }
    exec(code, safe_globals)
    result = safe_globals["df"]
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Transform code must assign result to 'df'")
    return result


def _apply_transform(session: dict, transform: dict) -> Optional[dict]:
    """Apply a data transform to the session. Returns the modification record or None on failure."""
    import time
    code = transform.get("code", "")
    description = transform.get("description", "Data transformation")
    if not code:
        return None
    try:
        new_df = _execute_transform(session["df"], code)
        session["df"] = new_df
        session["profile"] = profile_dataframe(new_df)
        mod = {
            "type": "transform",
            "variable": "",
            "from": description,
            "to": f"{len(new_df)} rows",
            "source": "ai",
            "timestamp": time.time(),
            "code": code,
            "description": description,
        }
        session.setdefault("modifications", []).append(mod)
        return mod
    except Exception as e:
        print(f"Transform execution failed: {e}")
        return None


def _apply_variable_edits(session: dict, edits: list[dict], source: str = "ai") -> list[dict]:
    """Apply variable edits to session profile and dataframe. Returns list of modification records."""
    import time
    profile = session["profile"]
    new_mods = []
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
                    mod = {"type": "rename", "variable": new_name, "from": col["name"], "to": new_name, "source": source, "timestamp": time.time()}
                    new_mods.append(mod)
                    col["name"] = new_name
                if "distribution" in updates:
                    old_dist = col.get("distribution", col["dtype"])
                    col["distribution"] = updates["distribution"]
                    mod = {"type": "retype", "variable": col["name"], "from": old_dist, "to": updates["distribution"], "source": source, "timestamp": time.time()}
                    new_mods.append(mod)
                break
    session.setdefault("modifications", []).extend(new_mods)
    return new_mods


def _iter_text_chunks(text: str, size: int = 24):
    """Yield fixed-size text chunks for incremental UI streaming."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


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
    new_mods = []
    if edits:
        new_mods = _apply_variable_edits(session, edits, source="ai")

    # Parse and apply any data transforms from the LLM response
    clean_response, transform = _parse_data_transform(clean_response)
    if transform:
        transform_mod = _apply_transform(session, transform)
        if transform_mod:
            new_mods.append(transform_mod)

    session["chat_history"].append({"role": "user", "content": request.message})
    session["chat_history"].append({"role": "assistant", "content": clean_response})

    result = {"response": clean_response}
    if edits or transform:
        result["profile_updated"] = True
        result["profile"] = session["profile"]
    if new_mods:
        result["modifications"] = new_mods
    return result


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response chunks, then emit final metadata event."""
    session = sessions.get(request.session_id)
    if not session:
        async def missing_session():
            payload = {"type": "error", "error": "Session not found"}
            yield json.dumps(payload) + "\n"
        return StreamingResponse(missing_session(), media_type="application/x-ndjson")

    async def event_stream():
        try:
            response = generate_chat_response(
                session["profile"],
                session["chat_history"],
                request.message,
            )

            clean_response, edits = _parse_variable_edits(response)
            new_mods = []
            if edits:
                new_mods = _apply_variable_edits(session, edits, source="ai")

            clean_response, transform = _parse_data_transform(clean_response)
            if transform:
                transform_mod = _apply_transform(session, transform)
                if transform_mod:
                    new_mods.append(transform_mod)

            session["chat_history"].append({"role": "user", "content": request.message})
            session["chat_history"].append({"role": "assistant", "content": clean_response})

            for chunk in _iter_text_chunks(clean_response):
                yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
                await asyncio.sleep(0)

            final_payload = {"type": "final", "response": clean_response}
            if edits or transform:
                final_payload["profile_updated"] = True
                final_payload["profile"] = session["profile"]
            if new_mods:
                final_payload["modifications"] = new_mods
            yield json.dumps(final_payload) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


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


class RevertModificationRequest(BaseModel):
    session_id: str
    timestamp: float


@app.post("/revert-modification")
async def revert_modification(request: RevertModificationRequest):
    """Revert a specific modification by timestamp. Rebuilds state from original df."""
    session = sessions.get(request.session_id)
    if not session:
        return {"error": "Session not found"}

    mods = session.get("modifications", [])
    # Remove the targeted modification
    remaining = [m for m in mods if m["timestamp"] != request.timestamp]
    session["modifications"] = remaining

    # Rebuild from original dataframe
    session["df"] = session["df_original"].copy()
    profile = profile_dataframe(session["df"])

    # Replay remaining modifications in order
    for mod in remaining:
        if mod["type"] == "rename":
            old_name = mod["from"]
            new_name = mod["to"]
            if old_name in session["df"].columns:
                session["df"] = session["df"].rename(columns={old_name: new_name})
                for col in profile["column_profiles"]:
                    if col["name"] == old_name:
                        col["name"] = new_name
                        break
        elif mod["type"] == "retype":
            for col in profile["column_profiles"]:
                if col["name"] == mod["variable"]:
                    col["distribution"] = mod["to"]
                    break
        elif mod["type"] == "transform" and mod.get("code"):
            try:
                session["df"] = _execute_transform(session["df"], mod["code"])
                profile = profile_dataframe(session["df"])
            except Exception:
                pass  # Skip failed transforms on replay

    session["profile"] = profile
    return {
        "status": "reverted",
        "profile": profile,
        "modifications": remaining,
    }


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


@app.post("/intent-chat-stream")
async def intent_chat_stream(request: IntentChatRequest):
    """Stream study intent chat response chunks, then emit final metadata event."""
    session = sessions.get(request.session_id)
    if not session:
        async def missing_session():
            payload = {"type": "error", "error": "Session not found"}
            yield json.dumps(payload) + "\n"
        return StreamingResponse(missing_session(), media_type="application/x-ndjson")

    profile = session["profile"]
    columns = [c["name"] for c in profile["column_profiles"]]

    async def event_stream():
        try:
            if request.message == "__commit__":
                intent = extract_intent_from_conversation(
                    profile=profile,
                    columns=columns,
                    chat_history=request.chat_history,
                )
                intent["timestamp"] = pd.Timestamp.now().isoformat()
                session["intent"] = intent
                yield json.dumps({"type": "final", "committed": True, "intent": intent}) + "\n"
                return

            if request.message == "__init__":
                response = generate_intent_chat(
                    profile=profile,
                    columns=columns,
                    chat_history=[],
                    user_message=None,
                )
                for chunk in _iter_text_chunks(response):
                    yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
                    await asyncio.sleep(0)
                yield json.dumps({"type": "final", "response": response, "intent_ready": False}) + "\n"
                return

            response = generate_intent_chat(
                profile=profile,
                columns=columns,
                chat_history=request.chat_history,
                user_message=request.message,
            )

            intent_ready = _check_intent_completeness(request.chat_history + [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": response},
            ])

            for chunk in _iter_text_chunks(response):
                yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
                await asyncio.sleep(0)

            yield json.dumps({"type": "final", "response": response, "intent_ready": intent_ready}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


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

        # Generate publication-ready plots
        try:
            plot_map = generate_all_plots(results)
            results["plot_map"] = plot_map
        except Exception as plot_err:
            print(f"Plot generation failed: {plot_err}")
            results["plot_map"] = {}

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
