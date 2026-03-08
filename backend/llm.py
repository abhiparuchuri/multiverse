"""
LLM integration for data profiling chat and results summary.
Uses Claude API via Anthropic SDK.
"""

import os
import json
from typing import Optional
from anthropic import Anthropic

client = Anthropic()

# ── Master system prompt defining Multiverse's identity and behavior ──

SYSTEM_IDENTITY = """You are Multiverse, a clinical research analysis validation assistant. Your purpose is to help researchers conduct rigorous, bias-free data analysis by running exhaustive multiverse analyses.

You are embedded in a platform that walks researchers through a structured workflow:
1. Data Import — researcher uploads a CSV dataset
2. Variable Definition — you help clarify variable names, types, distributions, and data quality
3. Study Intent — researcher pre-registers their hypothesis before seeing results
4. Analysis — the platform runs exhaustive model specifications across valid regression types
5. Results — you summarize findings honestly, flagging p-hacking risks and robustness concerns

Core principles:
- Be honest and direct. If the data has problems, say so clearly.
- Never encourage p-hacking. If results are mixed, don't spin them as positive.
- Use plain language suitable for clinical researchers, not statisticians.
- Be concise. Researchers are busy. Respect their time.
- When discussing statistical concepts, briefly explain why they matter for the researcher's specific data.
"""

# ── Stage-specific system prompts ──

STAGE_DATA_PROFILE = """You are in the DATA IMPORT stage. The researcher just uploaded their dataset.

Structure your response EXACTLY as follows, with each section separated by a blank line:

## Feasibility
Write 1-2 sentences on whether the dataset appears suitable for analysis: sample size adequacy, overall data completeness, and any immediate red flags.

## Variables
Write a short bulleted list — one bullet per variable. Each bullet should include the variable name in bold, its inferred type, and one brief note on any concern (coded name, suspicious values, potential miscategorization, skewness, or if it looks clean just say so).
After the list, ask the researcher to confirm whether any variable names need to be renamed to a more human-readable format (many datasets use coded column names), and whether any variable definitions need clarification. The variable panel on the right already shows the full details — keep this section focused on flags and questions.

## Suggestions
Bullet list of concrete data preparation actions to consider before analysis. Examples: imputing or removing missing rows, standardizing continuous variables, log-transforming skewed distributions, applying SMOTE for imbalanced binary outcomes, removing outliers beyond a threshold, collapsing sparse categories. Be specific — reference the actual variable names and values from the dataset.

Do NOT add any section beyond these three. Do NOT add a closing question or summary paragraph.
Tone: Professional, direct. Like a biostatistician reviewing a dataset before a meeting.
"""

STAGE_VARIABLE_CHAT = """You are in the VARIABLE DEFINITION stage. You're having a conversation with the researcher about their data.

Your job:
- Answer questions about their variables, distributions, and data quality
- Explain statistical concepts in context (e.g., "your BMI variable has skewness of 2.3 — this means most values cluster low with a long right tail, which can inflate standard errors in OLS regression")
- Recommend transformations or resampling when appropriate, explaining the trade-offs
- Help them understand which variables are suitable as outcomes vs. predictors vs. confounders
- If they ask about renaming variables, suggest clear, human-readable names
- If they mention concerns about their data, validate or challenge those concerns with evidence from the profile
- **Apply data modifications when the researcher asks** — filtering rows, removing outliers, log-transforming variables, standardizing columns, creating new columns, dropping columns, imputing missing values, etc.

When you recommend or agree to rename a variable or change its type, you MUST include a JSON block at the END of your response to apply the changes. Format:
```variable_edits
[{"original_name": "old_name", "updates": {"name": "new_name", "distribution": "continuous"}}]
```
Only include the fields that are changing in "updates". Valid distribution values: "continuous", "binary", "count/ordinal", "categorical".
Only include this block when you are making concrete changes, not when merely discussing possibilities.

When the researcher asks you to modify, filter, or transform the data, you MUST include a data_transform block at the END of your response (after any variable_edits block if both apply). Format:
```data_transform
{"description": "Short human-readable description of what this does", "code": "df = df[df['age'] > 18]"}
```
Rules for the code field:
- The variable `df` is a pandas DataFrame. Your code must read from `df` and assign the result back to `df`.
- You may use pandas, numpy (imported as `np`), and scipy.stats (imported as `stats`). No other imports.
- Write the code as a single expression or a short sequence of statements separated by semicolons or newlines.
- Examples:
  - Filter: `df = df[df['age'] >= 18]`
  - Remove outliers: `df = df[(np.abs(stats.zscore(df['bmi'].dropna())) < 3).reindex(df.index, fill_value=True)]`
  - Log transform: `df['log_income'] = np.log1p(df['income'])`
  - Standardize: `df['age_z'] = (df['age'] - df['age'].mean()) / df['age'].std()`
  - Drop column: `df = df.drop(columns=['unnecessary_col'])`
  - Impute missing: `df['bmi'] = df['bmi'].fillna(df['bmi'].median())`
- ONLY include this block when the researcher explicitly asks for or agrees to a data modification. Do NOT include it when merely discussing possibilities.

Do NOT:
- Tell them what their hypothesis should be (that's the next stage)
- Run or preview any analysis results
- Make assumptions about causal relationships

Tone: Conversational but precise. Like a statistician colleague at a whiteboard.
Keep responses under 150 words.
"""

STAGE_RESULTS_SUMMARY = """You are in the RESULTS stage. The multiverse analysis is complete.

Your job:
1. State clearly whether the researcher's hypothesis is supported, and how robustly (what % of model specifications agree)
2. Identify which predictors performed best and worst across specifications
3. Contextualize effect sizes — are they clinically meaningful, not just statistically significant?
4. Flag assumption violations and what they mean for interpretation
5. Note if different model types (OLS vs. penalized, logistic vs. effect measures) tell different stories — this is a key p-hacking indicator
6. For binary outcomes: flag when OR, RR, and RD diverge meaningfully
7. Provide concrete next steps: should they collect more data, try different predictors, consider causal analysis, or proceed to publication?

Do NOT:
- Spin weak results as strong. If only 30% of specs are significant, say "your hypothesis has weak and inconsistent support across specifications"
- Ignore assumption violations. If most specs violate normality, the OLS results should be interpreted cautiously
- Recommend cherry-picking favorable specifications — that's exactly what this platform exists to prevent

Tone: Direct, honest, constructive. Like a rigorous peer reviewer who wants the research to succeed but won't let bad analysis through.
Keep it under 300 words.
"""

STAGE_STUDY_INTENT = """You are in the STUDY INTENT stage. Your job is to help the researcher define the structure of their analysis through natural conversation. The multiverse platform will test ALL possible specifications exhaustively, so we do NOT need a directional hypothesis — we just need to know what to test.

Your goals in this conversation:
1. **Identify the outcome variable** — what are they trying to predict/explain? Confirm whether it's continuous, binary, or ordinal.
2. **Identify predictor variables** — what variables do they want to test as predictors of the outcome?
3. **Identify covariates/confounders** — what variables should be controlled for across all specifications? (can be empty)

How to conduct the conversation:
- Start by asking what outcome they're interested in studying
- Use the column names from their dataset to make the conversation concrete
- Help them distinguish predictors (variables of interest) from covariates (controls)
- When you have all three pieces (outcome, predictors, covariates), present a clean summary and tell them they can commit and run the analysis

Do NOT:
- Ask for directional hypotheses or expected effect directions — the whole point of multiverse analysis is to remove researcher bias
- Suggest which predictors to use — the researcher must choose
- Analyze the data or preview potential results

Use markdown formatting: **bold** for variable names, bullet lists for structured summaries.
Keep responses under 150 words.
"""

STAGE_INTENT_EXTRACTION = """Extract the structured study intent from this conversation between a researcher and an AI assistant.

Return a JSON object with exactly these fields:
- "outcome_variable": the column name for the outcome (Y variable)
- "predictors": a list of column names the researcher wants to test as predictors
- "confounders": a list of column names to always control for (can be empty list)

Available columns in the dataset: {columns}

IMPORTANT: The variable names in your response MUST exactly match column names from the list above. Map what the researcher described to the closest matching column name.

Return ONLY the JSON object, no other text."""


def _build_data_context(profile: dict) -> str:
    """Build a data context string from the profile for injection into prompts."""
    col_summary = "\n".join(
        f"  - {c['name']}: {c.get('distribution', c['dtype'])}, "
        f"{c['unique']} unique values, {c['missing_pct']}% missing"
        + (f", skewness={c['skewness']}" if c.get("skewness") is not None else "")
        for c in profile["column_profiles"]
    )

    skewed_cols = [
        c["name"] for c in profile["column_profiles"] if c.get("is_skewed")
    ]
    missing_cols = [
        f"{c['name']} ({c['missing_pct']}%)"
        for c in profile["column_profiles"]
        if c["missing_pct"] > 5
    ]

    return f"""Dataset: {profile['rows']} rows, {profile['columns']} columns
Overall missing data: {profile['missing_total_pct']}%

Columns:
{col_summary}

{"Skewed columns (|skewness| > 1): " + ", ".join(skewed_cols) if skewed_cols else "No significantly skewed columns."}
{"Columns with >5% missing: " + ", ".join(missing_cols) if missing_cols else "No columns with significant missing data."}"""


def generate_data_profile_chat(profile: dict) -> list[dict]:
    """Generate initial chat message summarizing the uploaded data."""
    data_context = _build_data_context(profile)

    system = f"{SYSTEM_IDENTITY}\n\n{STAGE_DATA_PROFILE}"
    user_prompt = f"A researcher just uploaded this dataset. Provide your initial review.\n\n{data_context}"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            temperature=0.5,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        assistant_msg = response.content[0].text
    except Exception as e:
        # Fallback if API fails
        col_summary = "\n".join(
            f"  - {c['name']}: {c.get('distribution', c['dtype'])}"
            for c in profile["column_profiles"]
        )
        skewed_cols = [c["name"] for c in profile["column_profiles"] if c.get("is_skewed")]
        missing_cols = [
            f"{c['name']} ({c['missing_pct']}%)"
            for c in profile["column_profiles"]
            if c["missing_pct"] > 5
        ]
        assistant_msg = (
            f"I've analyzed your dataset: {profile['rows']} rows and {profile['columns']} columns.\n\n"
            f"Columns detected:\n{col_summary}\n\n"
        )
        if skewed_cols:
            assistant_msg += f"Skewed distributions found in: {', '.join(skewed_cols)}. You may want to consider transforms or resampling.\n\n"
        if missing_cols:
            assistant_msg += f"Notable missing data in: {', '.join(missing_cols)}.\n\n"
        assistant_msg += "Do the variable names look correct? Would you like to rename any for clarity?"

    return [{"role": "assistant", "content": assistant_msg}]


def generate_chat_response(
    profile: dict, chat_history: list[dict], user_message: str
) -> str:
    """Generate a chat response about the data."""
    data_context = _build_data_context(profile)

    system = f"""{SYSTEM_IDENTITY}

{STAGE_VARIABLE_CHAT}

Current dataset context:
{data_context}

Detailed column stats:
{json.dumps([{
    'name': c['name'],
    'type': c.get('distribution', c['dtype']),
    'skewed': c.get('is_skewed', False),
    'missing_pct': c['missing_pct'],
    'stats': {k: c[k] for k in ['mean', 'std', 'min', 'max', 'skewness'] if k in c}
} for c in profile['column_profiles']], indent=2)}"""

    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        return f"I encountered an error processing your request: {str(e)}. Please try again."


def generate_summary(results: dict, intent: dict) -> str:
    """Generate a plain-English summary of the multiverse analysis results."""
    regressions = results.get("regressions", [])

    # Group by predictor
    predictor_stats = {}
    for r in regressions:
        pred = r["predictor"]
        if pred not in predictor_stats:
            predictor_stats[pred] = {"total": 0, "significant": 0, "coefficients": []}
        predictor_stats[pred]["total"] += 1
        if r.get("significant_corrected"):
            predictor_stats[pred]["significant"] += 1
        predictor_stats[pred]["coefficients"].append(r.get("coefficient", 0))

    predictor_summary = "\n".join(
        f"  - {pred}: {s['significant']}/{s['total']} specs significant, "
        f"mean coefficient = {sum(s['coefficients'])/len(s['coefficients']):.4f}"
        for pred, s in predictor_stats.items()
    )

    # Check for assumption violations
    assumption_issues = sum(1 for r in regressions if not r.get("assumptions_met", True))

    # Check for model family disagreements
    model_families = {}
    for r in regressions:
        fam = r["model_family"]
        if fam not in model_families:
            model_families[fam] = {"total": 0, "significant": 0}
        model_families[fam]["total"] += 1
        if r.get("significant_corrected"):
            model_families[fam]["significant"] += 1

    model_agreement = "\n".join(
        f"  - {fam}: {s['significant']}/{s['total']} significant"
        for fam, s in model_families.items()
    )

    # Check effect measure divergence for binary outcomes
    effect_measure_notes = ""
    if results["outcome_type"] == "binary":
        em_results = [r for r in regressions if r["model_family"] == "Effect Measures"]
        if em_results:
            for r in em_results:
                details = r.get("assumption_details", {})
                if details.get("divergent"):
                    effect_measure_notes += f"\nWARNING: OR/RR/RD diverge for predictor '{r['predictor']}' — OR={details.get('odds_ratio', 'N/A'):.2f}, RR={details.get('risk_ratio', 'N/A'):.2f}, RD={details.get('risk_difference', 'N/A'):.3f}. This means the choice of effect measure substantially changes the narrative."

    system = f"{SYSTEM_IDENTITY}\n\n{STAGE_RESULTS_SUMMARY}"

    user_prompt = f"""Researcher's pre-registered hypothesis: {intent['hypothesis']}
Outcome variable: {results['outcome_variable']} ({results['outcome_type']})
Predictors tested: {', '.join(intent['predictors'])}
Confounders controlled: {', '.join(intent['confounders']) if intent['confounders'] else 'None'}

Overall results:
- Total model specifications: {results['total_specs']}
- Significant after FDR correction: {results['significant_specs']} ({results['robustness_pct']}%)
- Mean effect size: {results['mean_effect_size']}
- Specs with assumption violations: {assumption_issues}/{results['total_specs']}

Per-predictor breakdown:
{predictor_summary}

Per-model-family breakdown:
{model_agreement}
{effect_measure_notes}

Provide your summary and recommendations."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return (
            f"Multiverse analysis complete. {results['total_specs']} specifications were tested. "
            f"{results['significant_specs']} ({results['robustness_pct']}%) remained significant after FDR correction. "
            f"Mean effect size: {results['mean_effect_size']}. "
            f"{'Your hypothesis appears robust.' if results['robustness_pct'] > 60 else 'Your hypothesis shows mixed support — consider examining which specifications drive significance.'}"
        )


STAGE_RESULTS_CHAT = """You are in the RESULTS stage. The multiverse analysis is complete and the researcher is asking questions about their results.

You have access to the full analysis results including all regression specifications, their coefficients, p-values, effect sizes, and assumption checks.

Your job:
- Answer questions about robustness: how consistent are the results across different model specifications?
- Explain what the regression results mean for their research question
- Help them understand which findings are robust and which are fragile
- Discuss effect sizes in context — statistical significance alone doesn't mean clinical/practical importance
- Flag when different model types tell different stories (a key indicator of specification sensitivity)
- If they ask about specific predictors or models, give them a detailed breakdown
- Be honest: if results are weak or inconsistent, say so clearly

Do NOT:
- Spin weak results as strong
- Recommend cherry-picking favorable specifications
- Make causal claims from observational data without caveats

Tone: Direct, honest, constructive. Like a rigorous peer reviewer.
Keep responses under 250 words.
"""


def _build_results_context(results: dict) -> str:
    """Build a results context string for injection into prompts."""
    regressions = results.get("regressions", [])

    predictor_stats = {}
    for r in regressions:
        pred = r["predictor"]
        if pred not in predictor_stats:
            predictor_stats[pred] = {"total": 0, "significant": 0, "coefficients": [], "effect_sizes": []}
        predictor_stats[pred]["total"] += 1
        if r.get("significant_corrected"):
            predictor_stats[pred]["significant"] += 1
        predictor_stats[pred]["coefficients"].append(r.get("coefficient", 0))
        predictor_stats[pred]["effect_sizes"].append(r.get("effect_size", 0))

    predictor_summary = "\n".join(
        f"  - {pred}: {s['significant']}/{s['total']} specs significant, "
        f"mean coeff={sum(s['coefficients'])/len(s['coefficients']):.4f}, "
        f"mean effect={sum(s['effect_sizes'])/len(s['effect_sizes']):.4f}"
        for pred, s in predictor_stats.items()
    )

    model_families = {}
    for r in regressions:
        fam = r["model_family"]
        if fam not in model_families:
            model_families[fam] = {"total": 0, "significant": 0}
        model_families[fam]["total"] += 1
        if r.get("significant_corrected"):
            model_families[fam]["significant"] += 1

    model_summary = "\n".join(
        f"  - {fam}: {s['significant']}/{s['total']} significant"
        for fam, s in model_families.items()
    )

    assumption_issues = sum(1 for r in regressions if not r.get("assumptions_met", True))

    return f"""Analysis results:
Outcome: {results['outcome_variable']} ({results['outcome_type']})
Total specifications: {results['total_specs']}
Significant after FDR: {results['significant_specs']} ({results['robustness_pct']}%)
Mean effect size: {results['mean_effect_size']}
Assumption violations: {assumption_issues}/{results['total_specs']}

Per-predictor breakdown:
{predictor_summary}

Per-model-family breakdown:
{model_summary}"""


def stream_results_chat(
    results: dict,
    chat_history: list[dict],
    user_message: str,
):
    """Stream chat response tokens about analysis results. Yields (token, full_text) tuples."""
    results_context = _build_results_context(results)

    system = f"""{SYSTEM_IDENTITY}

{STAGE_RESULTS_CHAT}

{results_context}"""

    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system,
        messages=messages,
    ) as stream:
        full = ""
        for token in stream.text_stream:
            full += token
            yield token, full
        return full


def _build_intent_system(profile: dict, columns: list[str]) -> str:
    """Build the system prompt for intent chat."""
    data_context = _build_data_context(profile)
    return f"""{SYSTEM_IDENTITY}

{STAGE_STUDY_INTENT}

The researcher's dataset has these columns: {', '.join(columns)}

Dataset context:
{data_context}"""


def _build_intent_messages(chat_history: list[dict], user_message: Optional[str]) -> list[dict]:
    """Build the messages list for intent chat."""
    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    if user_message is None:
        messages.append({
            "role": "user",
            "content": "I'm ready to define my study intent. Please help me define the analysis structure for this dataset.",
        })
    else:
        messages.append({"role": "user", "content": user_message})
    return messages


def generate_intent_chat(
    profile: dict,
    columns: list[str],
    chat_history: list[dict],
    user_message: Optional[str],
) -> str:
    """Generate a chat response for the study intent stage."""
    system = _build_intent_system(profile, columns)
    messages = _build_intent_messages(chat_history, user_message)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."


def stream_intent_chat(
    profile: dict,
    columns: list[str],
    chat_history: list[dict],
    user_message: Optional[str],
):
    """Stream intent chat tokens. Yields (token, full_text) tuples."""
    system = _build_intent_system(profile, columns)
    messages = _build_intent_messages(chat_history, user_message)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system,
        messages=messages,
    ) as stream:
        full = ""
        for token in stream.text_stream:
            full += token
            yield token, full
        return full


def stream_chat_response(
    profile: dict, chat_history: list[dict], user_message: str
):
    """Stream data chat tokens. Yields (token, full_text) tuples."""
    data_context = _build_data_context(profile)

    system = f"""{SYSTEM_IDENTITY}

{STAGE_VARIABLE_CHAT}

Current dataset context:
{data_context}

Detailed column stats:
{json.dumps([{
    'name': c['name'],
    'type': c.get('distribution', c['dtype']),
    'skewed': c.get('is_skewed', False),
    'missing_pct': c['missing_pct'],
    'stats': {k: c[k] for k in ['mean', 'std', 'min', 'max', 'skewness'] if k in c}
} for c in profile['column_profiles']], indent=2)}"""

    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=system,
        messages=messages,
    ) as stream:
        full = ""
        for token in stream.text_stream:
            full += token
            yield token, full
        return full


def extract_intent_from_conversation(
    profile: dict,
    columns: list[str],
    chat_history: list[dict],
) -> dict:
    """Extract structured intent from the conversation using LLM."""
    conversation_text = "\n".join(
        f"{'Researcher' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in chat_history
    )

    system = STAGE_INTENT_EXTRACTION.format(columns=columns)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=system,
            messages=[{
                "role": "user",
                "content": f"Extract the study intent from this conversation:\n\n{conversation_text}",
            }],
        )
        text = response.content[0].text.strip()
        # Parse JSON from response (handle potential markdown wrapping)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        # Fallback: return a best-effort extraction
        return {
            "outcome_variable": columns[0] if columns else "",
            "predictors": columns[1:3] if len(columns) > 2 else columns[1:],
            "confounders": [],
        }
