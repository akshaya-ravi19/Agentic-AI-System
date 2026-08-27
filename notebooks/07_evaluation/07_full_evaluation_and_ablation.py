# ============================================================
# NOTEBOOK 07 — Full Evaluation and Ablation Study
# Produces all tables and figures for Chapter 5.
# ============================================================
import pandas as pd, numpy as np, json
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *

print("=" * 60)
print("AGENTIC AI FULL EVALUATION")
print("=" * 60)

# ── 1. Classifier comparison ──────────────────────────────────
clf_path = EVAL_DIR / "classifier" / "comparison.csv"
if clf_path.exists():
    df_clf = pd.read_csv(clf_path)
    print("\n--- CLASSIFIER COMPARISON ---")
    print(df_clf.to_string(index=False))
    best = df_clf.loc[df_clf["pr_auc"].idxmax(), "model"]
    print(f"\nBest model (PR-AUC, consistent with notebook 04's selection logic): {best}")
else:
    print("Running Classification models to generate severity results.")

# ── 2. Agent vs rule-based ────────────────────────────────────
agent_path = EVAL_DIR / "agent" / "agent_results.csv"
expert_path = EVAL_DIR / "agent" / "expert_labelled_50_cases.csv"
agent_eval_ran = agent_path.exists() and expert_path.exists()
print("\n--- AGENT EVALUATION ---")
if agent_eval_ran:
    df_agent  = pd.read_csv(agent_path)
    df_expert = pd.read_csv(expert_path)
    level_map = {TRIAGE_LOG:0, TRIAGE_REVIEW:1, TRIAGE_ESCALATE:2}
    expert_num = df_expert["expert_triage"].map(level_map)
    agent_num  = df_agent["agent_triage"].map(level_map)
    rule_num   = df_agent["rule_triage"].map(level_map)

    kappa_agent = cohen_kappa_score(expert_num, agent_num)
    kappa_rule  = cohen_kappa_score(expert_num, rule_num)
    print(f"Cohen's Kappa — Agent:      {kappa_agent:.4f}")
    print(f"Cohen's Kappa — Rule-based: {kappa_rule:.4f}")
    print(f"Delta (agent adds):         {kappa_agent - kappa_rule:+.4f}")
    if kappa_agent >= KAPPA_THRESHOLD:
        print(f"PASS: agent kappa >= {KAPPA_THRESHOLD}")
    else:
        print(f"BELOW THRESHOLD: investigate agent reasoning quality")

    print("\nPer-level F1 (Agent vs Expert):")
    print(classification_report(expert_num, agent_num,
          target_names=[TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE]))
    print("Per-level F1 (Rule-based vs Expert):")
    print(classification_report(expert_num, rule_num,
          target_names=[TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE]))

    if "hallucination_count" in df_agent.columns:
        hall_rate = df_agent["hallucination_count"].sum() / len(df_agent)
        print(f"\nHallucination rate: {hall_rate:.2f} per case")
    if "tool_calls_correct" in df_agent.columns:
        tool_acc = df_agent["tool_calls_correct"].mean()
        print(f"Tool selection accuracy: {tool_acc:.1%}")
else:
    print("Steps to complete agent evaluation:")
    print("1. Create evaluation/agent/expert_labelled_50_cases.csv")
    print("   with columns: complaint_id, descriptor, camis,")
    print("   bilstm_pred, last_grade, days_since_inspection,")
    print("   complaint_count_30d, expert_triage")
    print("2. Run agent on each case in notebook 05")
    print("3. Save results to evaluation/agent/agent_results.csv")
    print("   with columns: complaint_id, agent_triage, rule_triage,")
    print("   tool_calls, hallucination_count, tool_calls_correct")

# ── 3. Clustering validation ──────────────────────────────────
cluster_path = EVAL_DIR / "clustering" / "cluster_results.csv"
print("\n--- CLUSTERING VALIDATION ---")
if cluster_path.exists():
    df_cl = pd.read_csv(cluster_path)
    print(f"Clusters: {df_cl['cluster_id'].nunique()-1}")
    print(f"In cluster: {(df_cl['cluster_id']>=0).mean():.1%}")
    print("See notebook 06 for Silhouette score and chi-square results.")
else:
    print("Run notebook 06 first.")

# ── 4. Ablation study template ────────────────────────────────
print("\n--- ABLATION STUDY ---")
print("""
For each pipeline version, run triage on same 50 eval cases,
compute Cohen's Kappa vs expert labels:

  Version A: No BiLSTM — all complaints enter agent directly
             (set bilstm_pred=1 for all, measure routing accuracy)
  Version B: No cluster context tool
             (disable get_cluster_context in agent tools)
  Version C: No inspection history tool
             (disable get_inspection_history in agent tools)
  Version D: Full pipeline — all components

Expected result: removing any core component degrades Kappa.
If it does not: discuss whether that component is necessary.
""")

# ── 5. Fairness / bias audit ────────────────────────────────────
# Checks whether the classifier behaves consistently across
# neighbourhood (borough, as a proxy -- true "neighbourhood" would
# need NTA-level data) and cuisine subgroups. This matters
# specifically for this system: if severe predictions systematically
# skew toward certain boroughs or cuisines independent of actual
# risk, that's a real equity problem for a tool influencing which
# restaurants get inspected.
print("\n--- FAIRNESS / BIAS AUDIT ---")
labelled_path = DATA_LABELLED / "labelled_complaints.csv"
if labelled_path.exists() and clf_path.exists():
    df_fair = pd.read_csv(labelled_path)
    # Needs model predictions on this data -- if you already saved
    # per-row predictions from notebook 04, load them here instead
    # of re-predicting. This assumes a `predicted_label` column;
    # adjust the load path if notebook 04 saves it elsewhere.
    pred_path = EVAL_DIR / "classifier" / "test_predictions.csv"
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        try:
            from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
            from sklearn.metrics import recall_score

            for group_col in ["borough", "cuisine_description"]:
                if group_col not in df_pred.columns:
                    print(f"  [skip] '{group_col}' not present in test_predictions.csv")
                    continue
                mf = MetricFrame(
                    metrics={"selection_rate": selection_rate, "recall": recall_score},
                    y_true=df_pred["y_true"], y_pred=df_pred["y_pred"],
                    sensitive_features=df_pred[group_col],
                )
                print(f"\nFairness by {group_col}:")
                print(mf.by_group)
                print(f"Selection rate range: {mf.difference(method='between_groups')['selection_rate']:.3f}")
                print("(Large differences here mean the classifier flags some groups as")
                print(" severe far more/less often than others -- worth discussing, not")
                print(" necessarily a bug, since real risk may genuinely differ by area.)")
        except ImportError:
            print("  fairlearn not installed. Run: pip install fairlearn")
            print("  (added to requirements.txt -- see CHANGES_APPLIED.md)")
    else:
        print(f"  Missing {pred_path}. Notebook 04 needs to save per-row test-set")
        print("  predictions with borough/cuisine_description columns attached")
        print("  before this audit can run. See CHANGES_APPLIED.md for the change needed.")

    # SHAP on the Logistic Regression baseline only -- it's the one
    # model here with a directly interpretable linear structure, so
    # it gives the most legible "which features drove this decision"
    # story for a dissertation without needing a BiLSTM-specific
    # explainer (e.g. DeepSHAP) that adds real complexity for
    # marginal extra insight at this project's scope.
    try:
        import shap
        print("\nSHAP is installed -- run notebook 04's Logistic Regression model")
        print("through shap.LinearExplainer(lr, X_train) and shap.summary_plot()")
        print("to get feature-importance figures for your fairness discussion.")
    except ImportError:
        print("\n  shap not installed. Run: pip install shap")
else:
    print("Run notebooks 03 and 04 first.")

# ── 6. User evaluation questionnaire ────────────────────────────
print("\n--- USER EVALUATION QUESTIONNAIRE ---")
questionnaire_path = EVAL_DIR / "user_questionnaire.csv"
if not questionnaire_path.exists():
    questions = pd.DataFrame([
        {"id": "Q1", "dimension": "Trust", "question": "I trust the triage tier the system recommended.", "scale": "1-5 Likert"},
        {"id": "Q2", "dimension": "Trust", "question": "The evidence shown was sufficient to understand why this tier was chosen.", "scale": "1-5 Likert"},
        {"id": "Q3", "dimension": "Usability", "question": "The recommendation was easy to understand.", "scale": "1-5 Likert"},
        {"id": "Q4", "dimension": "Usability", "question": "I could quickly find the information I needed to make my own decision.", "scale": "1-5 Likert"},
        {"id": "Q5", "dimension": "Perceived accuracy", "question": "Based on my own judgement, this recommendation matched what I would have decided.", "scale": "1-5 Likert"},
        {"id": "Q6", "dimension": "Workload", "question": "Using this tool would reduce my workload compared to reviewing complaints manually.", "scale": "1-5 Likert"},
        {"id": "Q7", "dimension": "Autonomy", "question": "The tool made it clear that the final decision is mine to make.", "scale": "1-5 Likert"},
        {"id": "Q8", "dimension": "Overall", "question": "I would want to use this tool in my actual work.", "scale": "1-5 Likert"},
        {"id": "Q9", "dimension": "Open-ended", "question": "What, if anything, would make you distrust a recommendation from this system?", "scale": "free text"},
        {"id": "Q10", "dimension": "Open-ended", "question": "What information is missing that would help you make a faster or more confident decision?", "scale": "free text"},
    ])
    questions.to_csv(questionnaire_path, index=False)
    print(f"Saved a starter questionnaire template -> {questionnaire_path}")
    print("Administer this to inspectors/EHOs reviewing a sample of agent")
    print("recommendations (ideally the same 50 cases used for Kappa evaluation),")
    print("then analyse Likert responses descriptively and open-ended responses")
    print("thematically for your user evaluation chapter.")
else:
    print(f"Questionnaire template already exists at {questionnaire_path}")

# ── 7. Save final results table ───────────────────────────────
print("\n--- SAVING DISSERTATION TABLES ---")
summary = {
    "classifier_best_severe_recall": df_clf["severe_recall"].max() if clf_path.exists() else "TBD",
    "classifier_best_pr_auc":        df_clf["pr_auc"].max() if clf_path.exists() else "TBD",
    # NOTE: previously checked only agent_path.exists() here, but
    # kappa_agent/kappa_rule are only ever defined when BOTH agent_path
    # AND expert_path exist (see the block above) -- checking the
    # wrong condition could raise a NameError if only one file existed.
    "agent_kappa":                   kappa_agent if agent_eval_ran else "TBD",
    "rule_kappa":                    kappa_rule  if agent_eval_ran else "TBD",
}
pd.DataFrame([summary]).to_csv(EVAL_DIR/"final_results_summary.csv", index=False)
print(f"Saved → {EVAL_DIR}/final_results_summary.csv")
print("\nEvaluation complete")
print("Then move to pipeline/ for GCP deployment.")