import uuid
import io
import os
import re
import json
import asyncio
import traceback
import math
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

from llm import generate_data_profile_chat, generate_chat_response, generate_intent_chat, extract_intent_from_conversation, stream_results_chat, stream_intent_chat, stream_chat_response
from modal_client import (
    run_analysis,
    run_classifiers_call,
    generate_all_plots_call,
    generate_classifier_artifacts_call,
    generate_distribution_plots_call,
    PLOT_DIR,
)

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
    intent_override: Optional[dict] = None


class UpdateVariableRequest(BaseModel):
    session_id: str
    original_name: str
    updates: dict


class AnalyzeRequest(BaseModel):
    session_id: str


def _to_json_safe(value):
    """Recursively convert numpy/pandas scalars and containers to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
            # Stream tokens from LLM, collecting full response
            full_response = ""
            for token, full in stream_chat_response(
                session["profile"],
                session["chat_history"],
                request.message,
            ):
                full_response = full
                # Stream raw tokens to client for progressive rendering
                yield json.dumps({"type": "chunk", "content": token}) + "\n"
                await asyncio.sleep(0)

            # Parse structured blocks from complete response
            clean_response, edits = _parse_variable_edits(full_response)
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
        return {
            "response": response,
            "intent_ready": False,
            "missing_fields": ["outcome", "predictors", "covariates"],
            "intent_draft": None,
            "covariates_auto_assigned": False,
        }

    # Handle commit — extract structured intent from conversation
    if request.message == "__commit__":
        if request.intent_override:
            intent = request.intent_override
            column_set = set(columns)
            missing_fields = []
            outcome = intent.get("outcome_variable")
            predictors = [p for p in (intent.get("predictors") or []) if p in column_set]
            covariates = [c for c in (intent.get("confounders") or []) if c in column_set]
            intent["predictors"] = predictors
            intent["confounders"] = covariates
            if not outcome or outcome not in column_set:
                missing_fields.append("outcome")
            if len(predictors) == 0:
                missing_fields.append("predictors")
            if len(covariates) == 0:
                missing_fields.append("covariates")
            intent_ready = len(missing_fields) == 0
        else:
            intent_ready, missing_fields, intent, _ = _assess_intent_completeness(
                profile=profile,
                chat_history=request.chat_history,
            )
        if not intent_ready:
            return {
                "committed": False,
                "intent_ready": False,
                "missing_fields": missing_fields,
            }
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
    convo = request.chat_history + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": response},
    ]
    intent_ready, missing_fields, intent_draft, covariates_auto_assigned = _assess_intent_completeness(
        profile=profile,
        chat_history=convo,
    )
    return {
        "response": response,
        "intent_ready": intent_ready,
        "missing_fields": missing_fields,
        "intent_draft": intent_draft,
        "covariates_auto_assigned": covariates_auto_assigned,
    }


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
                if request.intent_override:
                    intent = request.intent_override
                    column_set = set(columns)
                    missing_fields = []
                    outcome = intent.get("outcome_variable")
                    predictors = [p for p in (intent.get("predictors") or []) if p in column_set]
                    covariates = [c for c in (intent.get("confounders") or []) if c in column_set]
                    intent["predictors"] = predictors
                    intent["confounders"] = covariates
                    if not outcome or outcome not in column_set:
                        missing_fields.append("outcome")
                    if len(predictors) == 0:
                        missing_fields.append("predictors")
                    if len(covariates) == 0:
                        missing_fields.append("covariates")
                    intent_ready = len(missing_fields) == 0
                else:
                    intent_ready, missing_fields, intent, _ = _assess_intent_completeness(
                        profile=profile,
                        chat_history=request.chat_history,
                    )
                if not intent_ready:
                    yield json.dumps({
                        "type": "final",
                        "committed": False,
                        "intent_ready": False,
                        "missing_fields": missing_fields,
                    }) + "\n"
                    return
                intent["timestamp"] = pd.Timestamp.now().isoformat()
                session["intent"] = intent
                yield json.dumps({"type": "final", "committed": True, "intent": intent}) + "\n"
                return

            # True token streaming for both __init__ and regular messages
            user_msg = None if request.message == "__init__" else request.message
            history = [] if request.message == "__init__" else request.chat_history

            response = ""
            for token, full in stream_intent_chat(
                profile=profile,
                columns=columns,
                chat_history=history,
                user_message=user_msg,
            ):
                response = full
                yield json.dumps({"type": "chunk", "content": token}) + "\n"
                await asyncio.sleep(0)

            intent_ready = False
            missing_fields = ["outcome", "predictors", "covariates"]
            intent_draft = None
            covariates_auto_assigned = False
            if request.message != "__init__":
                convo = request.chat_history + [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": response},
                ]
                intent_ready, missing_fields, intent_draft, covariates_auto_assigned = _assess_intent_completeness(
                    profile=profile,
                    chat_history=convo,
                )
            yield json.dumps({
                "type": "final",
                "response": response,
                "intent_ready": intent_ready,
                "missing_fields": missing_fields,
                "intent_draft": intent_draft,
                "covariates_auto_assigned": covariates_auto_assigned,
            }) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


class ResultsChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: list[dict]


@app.post("/results-chat-stream")
async def results_chat_stream(request: ResultsChatRequest):
    """Stream results chat response chunks, then emit final metadata event."""
    session = sessions.get(request.session_id)
    if not session:
        async def missing_session():
            yield json.dumps({"type": "error", "error": "Session not found"}) + "\n"
        return StreamingResponse(missing_session(), media_type="application/x-ndjson")

    results = session.get("results")
    if not results:
        async def no_results():
            yield json.dumps({"type": "error", "error": "No analysis results available"}) + "\n"
        return StreamingResponse(no_results(), media_type="application/x-ndjson")

    async def event_stream():
        try:
            response = ""
            for token, full in stream_results_chat(
                results=results,
                chat_history=request.chat_history,
                user_message=request.message,
            ):
                response = full
                yield json.dumps({"type": "chunk", "content": token}) + "\n"
                await asyncio.sleep(0)

            yield json.dumps({"type": "final", "response": response}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def _user_explicitly_declared_no_covariates(chat_history: list[dict]) -> bool:
    """Detect explicit user statements that no covariates/confounders are needed."""
    user_text = " ".join(
        str(m.get("content", "")).lower() for m in chat_history if m.get("role") == "user"
    )
    patterns = [
        r"\bno\s+covariates?\b",
        r"\bno\s+confounders?\b",
        r"\bno\s+controls?\b",
        r"\bnone\s+for\s+(?:covariates?|confounders?|controls?)\b",
    ]
    return any(re.search(p, user_text) for p in patterns)


def _assess_intent_completeness(
    profile: dict,
    chat_history: list[dict],
) -> tuple[bool, list[str], dict, bool]:
    """
    Readiness rule for enabling commit:
    - outcome defined
    - predictors defined
    - covariates defined (mapped to confounders field)
    """
    columns = [c["name"] for c in profile["column_profiles"]]
    column_set = set(columns)
    intent = extract_intent_from_conversation(
        profile=profile,
        columns=columns,
        chat_history=chat_history,
    )

    missing_fields: list[str] = []
    covariates_auto_assigned = False

    outcome = intent.get("outcome_variable")
    if not outcome or outcome not in column_set:
        missing_fields.append("outcome")

    predictors = [p for p in (intent.get("predictors") or []) if p in column_set]
    intent["predictors"] = predictors
    if len(predictors) == 0:
        missing_fields.append("predictors")

    covariates = [c for c in (intent.get("confounders") or []) if c in column_set]
    if len(covariates) == 0 and outcome in column_set and len(predictors) > 0:
        covariates = [c for c in columns if c != outcome and c not in set(predictors)]
        covariates_auto_assigned = True
    intent["confounders"] = covariates
    if len(covariates) == 0 and not _user_explicitly_declared_no_covariates(chat_history):
        missing_fields.append("covariates")

    return len(missing_fields) == 0, missing_fields, intent, covariates_auto_assigned


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
        results = run_analysis(
            df=df,
            outcome=intent["outcome_variable"],
            predictors=intent["predictors"],
            confounders=intent["confounders"],
            hypothesis=intent.get("hypothesis", ""),
        )

        # Generate publication-ready regression plots + DAGs
        try:
            plot_map, dag_map = generate_all_plots_call(results, df=session["df"])
            results["plot_map"] = plot_map
            results["dag_map"] = dag_map
        except Exception as plot_err:
            print(f"Plot generation failed: {plot_err}")
            results["plot_map"] = {}
            results["dag_map"] = {}

        # Run classifiers (for all outcomes when enabled)
        enable_classifiers = os.environ.get("ANALYSIS_ENABLE_CLASSIFIERS", "1").strip().lower() in ("1", "true", "yes")
        if enable_classifiers:
            try:
                clf_results = run_classifiers_call(
                    df=df,
                    outcome=intent["outcome_variable"],
                    predictors=intent["predictors"],
                    confounders=intent["confounders"],
                )
                results["classifier_results"] = clf_results.get("classifier_results", [])
                results["classifier_summary"] = {
                    "total_specs": clf_results.get("total_specs", 0),
                    "best_classifier": clf_results.get("best_classifier"),
                    "best_accuracy": clf_results.get("best_accuracy"),
                    "mean_accuracy": clf_results.get("mean_accuracy"),
                    "mean_auc": clf_results.get("mean_auc"),
                }

                # Batch-generate classifier artifacts (single remote call when USE_MODAL=1).
                clf_plot_map, clf_dag_map = generate_classifier_artifacts_call(
                    results["classifier_results"],
                    intent["outcome_variable"],
                )
                results["classifier_dag_map"] = clf_dag_map
                results["classifier_plot_map"] = clf_plot_map
            except Exception as clf_err:
                traceback.print_exc()
                print(f"Classifier analysis failed: {clf_err}")
                results["classifier_results"] = []
                results["classifier_summary"] = {}
                results["classifier_plot_map"] = {}
                results["classifier_dag_map"] = {}
        else:
            results["classifier_results"] = []
            results["classifier_summary"] = {}
            results["classifier_plot_map"] = {}
            results["classifier_dag_map"] = {}

        # Generate distribution plots for all analysis variables (batched).
        try:
            variable_roles = [{"col_name": intent["outcome_variable"], "role": "outcome"}]
            variable_roles.extend(
                {"col_name": p, "role": "predictor"} for p in intent["predictors"] if p in df.columns
            )
            variable_roles.extend(
                {"col_name": c, "role": "covariate"}
                for c in intent["confounders"]
                if c in df.columns and c != intent["outcome_variable"] and c not in intent["predictors"]
            )
            # Deduplicate while preserving order.
            seen_cols = set()
            deduped_variable_roles = []
            for item in variable_roles:
                col = item["col_name"]
                if col in seen_cols:
                    continue
                seen_cols.add(col)
                deduped_variable_roles.append(item)

            dist_plot_map = generate_distribution_plots_call(df, deduped_variable_roles)
            results["distribution_plot_map"] = dist_plot_map
        except Exception as dist_err:
            print(f"Distribution plot generation failed: {dist_err}")
            results["distribution_plot_map"] = {}

        # Build distribution stats for frontend
        try:
            dist_stats = []
            all_vars = (
                [(intent["outcome_variable"], "outcome")]
                + [(p, "predictor") for p in intent["predictors"]]
                + [(c, "covariate") for c in intent["confounders"]]
            )
            for var_name, role in all_vars:
                if var_name not in df.columns:
                    continue
                col = df[var_name]
                is_numeric = np.issubdtype(col.dtype, np.number)
                stat = {
                    "variable": var_name,
                    "role": role,
                    "n": int(col.count()),
                    "missing": int(col.isna().sum()),
                    "missing_pct": round(float(col.isna().mean() * 100), 1),
                    "unique": int(col.nunique()),
                }
                if is_numeric:
                    stat.update({
                        "mean": round(float(col.mean()), 4),
                        "std": round(float(col.std()), 4),
                        "min": round(float(col.min()), 4),
                        "max": round(float(col.max()), 4),
                        "median": round(float(col.median()), 4),
                        "skewness": round(float(col.skew()), 4),
                    })
                results.setdefault("distribution_stats", []).append(stat)
        except Exception as stat_err:
            print(f"Distribution stats failed: {stat_err}")

        results["session_id"] = request.session_id
        results = _to_json_safe(results)
        session["results"] = results
        return results
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/sessions")
async def list_sessions():
    return {sid: {"has_results": "results" in s, "clf_count": len((s.get("results") or {}).get("classifier_results") or [])} for sid, s in sessions.items()}


@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """Get cached results (if any)."""
    session = sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    return session.get("results", {"error": "No results yet"})
