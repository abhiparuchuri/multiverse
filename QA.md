# Omniverse — Hackathon Q&A Prep

## The Pitch

**One sentence?**
Omniverse makes clinical research findings reproducible by running every valid statistical model simultaneously and showing you how robust your result actually is.

**What's the hook?**
P-hacking is endemic in research. A researcher can unconsciously pick the model specification that gives them a significant p-value. We make that impossible to hide — you run all of them at once.

**Who cares?**
Journals rejecting papers for lack of robustness checks. FDA reviewers scrutinizing trial analyses. Clinical researchers who want their findings to hold up. Anyone who's been burned by a result that didn't replicate.

---

## Product Questions

**What does it actually do?**
Upload a CSV, tell it your outcome variable, predictors, and confounders in plain English. It automatically runs every defensible model specification — different covariate combinations, model families, and transformations — then tells you what % of specifications agree with each other.

**What is multiverse analysis?**
Instead of fitting one regression, you fit every valid combination. If your effect holds across 90% of specifications, it's robust. If it only appears in 10%, it's likely an artifact of your modeling choices. Steiger & McCulloch coined the term; it's increasingly required by top journals.

**What does the "robustness %" mean?**
The share of specifications where the predictor's effect is statistically significant after FDR correction. 80%+ = strong finding. Below 50% = treat with caution.

**What's the spec curve?**
Every model sorted by effect size. Green dots = significant after correction, grey = not. A tight band of consistent effects means your finding is real. A scattered mess means it depends heavily on arbitrary choices.

**What are covariate roles?**
We classify each covariate automatically using the change-in-estimate method: confounder (changes the estimate >10%), mediator (on the causal pathway), precision variable (reduces variance only), or neutral. Shown on the DAG for each specification.

---

## Technical Questions

**Tech stack?**
- Frontend: Next.js 14 (App Router), TypeScript, Tailwind CSS, shadcn/ui, Recharts, TanStack Table
- Backend: FastAPI (Python 3.11), single-process, in-memory session store
- Analysis: statsmodels, scikit-learn, scipy, pandas, numpy
- LLM: Claude claude-sonnet-4-6 via Anthropic API, streaming responses
- Optional cloud compute: Modal serverless containers (already integrated, flip `USE_MODAL=1`)

**Where does AI come in?**
Three places: (1) variable profiling chat — Claude reads your columns, flags skewed distributions, encoding issues, and suggests roles; (2) study intent — you describe your hypothesis in plain English and Claude extracts the outcome, predictors, and confounders in a structured format; (3) results chat — ask plain-English questions about your findings after analysis completes.

**What statistical models do you run?**

*Continuous outcomes:*
- OLS (baseline), with Lasso / Ridge / Elastic Net penalized variants
- If assumptions violated: log-transform OLS, Box-Cox OLS, quantile regression (median)
- Across every valid covariate subset (powerset up to size 4)

*Binary outcomes:*
- Logistic regression (simple + multiple), penalized logistic (Lasso/Ridge)
- Effect measures: Odds Ratio, Risk Ratio, Risk Difference

*Classifiers (across feature subsets):*
- Random Forest, Gradient Boosting, XGBoost, SVM (RBF), Logistic Regression
- 3-fold stratified cross-validation, single `cross_validate` pass for accuracy / AUC / precision / recall

**What assumption checks run automatically?**
- **Shapiro-Wilk** — normality of residuals
- **Breusch-Pagan** — homoscedasticity
- **VIF** — multicollinearity (flags >5)
- **Residual linearity** — correlation of residuals vs fitted
- Binary: minimum class size, complete separation detection
- Violated assumptions → model family automatically switches to a robust alternative

**How are covariate roles classified?**
Change-in-estimate method per covariate:
- **Confounder** — adding it shifts the primary coefficient >10%
- **Mediator** — on the causal pathway (detected via temporal/domain heuristics)
- **Precision variable** — reduces SE without meaningfully changing estimate
- **Neutral** — no material effect either way
Each DAG is colored by role so you can see the causal structure per specification.

**What is FDR correction and why does it matter here?**
Benjamini-Hochberg applied across *all* p-values from all specifications simultaneously. If you run 300 models without correction, you'd expect 15 spurious significant results at α=0.05. BH controls the expected proportion of false discoveries across the whole family. Corrected p-values are what drive the robustness % and the green/grey coloring.

**How many specifications does a typical run generate?**
With k predictors and n confounders, covariate subsets grow as O(2^n × k). A typical Pima diabetes run (3 predictors, 5 confounders) produces ~350–400 regression specs and ~25 classifier specs. All visible and filterable in the results table.

**How does it scale beyond a laptop?**
Modal integration is already wired in. `USE_MODAL=1` routes `run_analysis`, `run_classifiers`, and all plot generation to Modal serverless containers with auto-scaling. The FastAPI backend stays as the orchestrator; Modal handles the compute. Cold start ~3s, then parallelizable across specs.

**How are plots generated and served?**
All server-side via matplotlib (Agg backend, no display). Per-specification scatter plots with regression overlays, forest plots, DAGs per spec, specification curve, feature importance bar charts, distribution histograms. Saved as PNGs to `backend/plots/`, served via FastAPI `StaticFiles` at `/plots/`. Frontend fetches with `<img src>` — no base64 blobs in the JSON response.

---

## Hardball Questions

**Why not just use R or Python directly?**
You could, but you'd spend hours writing the loop yourself, handling assumption violations, applying FDR correction, and generating interpretable visualizations. This does it in under 5 minutes with a CSV and plain English.

**Is this novel research?**
Multiverse analysis as a concept isn't new. The novel part is the UX: making it accessible to any researcher without statistical programming skills, with LLM-guided setup and automatic assumption handling.

**Doesn't running hundreds of models inflate false positives?**
That's exactly why FDR correction (Benjamini-Hochberg) is applied across all specifications simultaneously, not per-model. We penalize for multiplicity correctly.

**What about confounding you can't see?**
Unmeasured confounders are a fundamental limitation of observational data — Omniverse can't fix that. What it does is make the measured modeling choices fully transparent and exhaustive.

**Is it production-ready?**
No — it's a hackathon POC. In-memory only, no auth, no persistence. The architecture (FastAPI + Modal) is production-viable; it just needs a database, auth layer, and async job queue.

**What's next if you won?**
Persistent results storage, PDF report export, HIPAA-compliant deployment, integration with REDCap/STATA file formats, and a team audit trail so co-authors can review every specification choice.
