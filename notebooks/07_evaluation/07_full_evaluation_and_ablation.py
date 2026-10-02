# ============================================================
# NOTEBOOK 07 — Full Evaluation and Ablation Study
# ============================================================
import pandas as pd, numpy as np, json
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *


print("AGENTIC AI FULL EVALUATION")

# ── 1. Classifier comparison ──────────────────────────────────
clf_path = EVAL_DIR / "classifier" / "comparison.csv"
if clf_path.exists():
    df_clf = pd.read_csv(clf_path)
    print("\nCLASSIFIER COMPARISON")
    print(df_clf.to_string(index=False))
    best = df_clf.loc[df_clf["pr_auc"].idxmax(), "model"]
    print(f"\nBest model (PR-AUC, consistent with notebook 04's selection logic): {best}")
else:
    print("Running Classification models to generate severity results.")

# ── 2. Agent vs rule-based ────────────────────────────────────
agent_path = EVAL_DIR / "agent" / "agent_results.csv"
expert_path = EVAL_DIR / "agent" / "expert_labelled_50_cases.csv"
agent_eval_ran = agent_path.exists() and expert_path.exists()
print("\nAGENT EVALUATION")
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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm_agent = confusion_matrix(expert_num, agent_num, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_agent, annot=True, fmt="d", cmap="YlGnBu", 
                xticklabels=[TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE], 
                yticklabels=[TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE])
    plt.title("Agent vs Expert Triage Decisions")
    plt.ylabel("Expert Label (Ground Truth)")
    plt.xlabel("Agent Prediction")
    plt.tight_layout()
    cm_agent_path = EVAL_DIR / "agent" / "agent_confusion_matrix.png"
    plt.savefig(cm_agent_path, dpi=150)
    plt.close()
    print(f"Saved Agent confusion matrix to {cm_agent_path}")

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
print("\nCLUSTERING VALIDATION")
if cluster_path.exists():
    df_cl = pd.read_csv(cluster_path)
    print(f"Clusters: {df_cl['cluster_id'].nunique()-1}")
    print(f"In cluster: {(df_cl['cluster_id']>=0).mean():.1%}")
    print("See notebook 06 for Silhouette score and chi-square results.")
else:
    print("Run notebook 06 first.")


# ── 5. Fairness / bias audit ────────────────────────────────────
# Checks whether the classifier behaves consistently across
# neighbourhood (borough, as a proxy -- true "neighbourhood" would
# need NTA-level data) and cuisine subgroups. This matters
# specifically for this system: if severe predictions systematically
# skew toward certain boroughs or cuisines independent of actual
# risk, that's a real equity problem for a tool influencing which
# restaurants get inspected.
print("\nFAIRNESS/BIAS AUDIT")
labelled_path = DATA_LABELLED / "labelled_complaints_ground_truth.csv"
if labelled_path.exists() and clf_path.exists():
    df_fair = pd.read_csv(labelled_path, low_memory=False)
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

    # ── BiLSTM SHAP Explainability (GradientExplainer) ───────────────────────
    # We explain the PRODUCTION BiLSTM (bilstm_food_safety.keras) rather than
    # the Logistic Regression baseline, because the BiLSTM is the actual
    # deployed model. shap.GradientExplainer works directly with TF/Keras and
    # computes expected gradients as SHAP values across embedding dimensions.
    print("\n--- BILSTM SHAP EXPLAINABILITY ---")
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tensorflow as tf

        model_path = MODELS_DIR / "bilstm" / "bilstm_food_safety.keras"
        embeddings_path = DATA_PROCESSED / "embeddings.npy"

        if not model_path.exists():
            print("  [skip] bilstm_food_safety.keras not found. Run notebook 04 first.")
        elif not embeddings_path.exists():
            print("  [skip] embeddings.npy not found. Run notebook 04 first.")
        else:
            print("  Loading production BiLSTM and embeddings...")
            bilstm_model = tf.keras.models.load_model(str(model_path))
            X_all = np.load(str(embeddings_path))  # shape: (N, 384)
            X_seq = X_all.reshape(-1, 1, 384)      # → (N, 1, 384)

            rng = np.random.default_rng(42)
            background_idx = rng.choice(len(X_seq), size=min(100, len(X_seq)), replace=False)
            X_background = X_seq[background_idx]
            explain_idx = rng.choice(len(X_seq), size=min(50, len(X_seq)), replace=False)
            X_explain = X_seq[explain_idx]

            print("  Running SHAP GradientExplainer on BiLSTM (this may take ~1 min)...")
            explainer = shap.GradientExplainer(bilstm_model, X_background)
            shap_values = explainer.shap_values(X_explain)

            sv = np.array(shap_values).squeeze()  # → (N, 384)
            X_exp_2d = X_explain.squeeze(axis=1)  # → (N, 384)

            mean_abs_shap = np.abs(sv).mean(axis=0)
            top_k = 20
            top_dims = np.argsort(mean_abs_shap)[::-1][:top_k]

            # Bar chart: top 20 embedding dimensions by mean |SHAP|
            plt.figure(figsize=(10, 6))
            plt.barh(
                [f"Dim {d}" for d in top_dims[::-1]],
                mean_abs_shap[top_dims[::-1]],
                color="steelblue"
            )
            plt.xlabel("Mean |SHAP Value|")
            plt.title(f"Production BiLSTM — Top {top_k} Most Influential Embedding Dimensions\n"
                      "(Higher = Stronger influence on Actionable Hazard prediction)")
            plt.tight_layout()
            shap_bar_path = EVAL_DIR / "classifier" / "bilstm_shap_top_dims.png"
            plt.savefig(shap_bar_path, dpi=150)
            plt.close()
            print(f"  Saved SHAP bar chart → {shap_bar_path}")

            # Beeswarm summary plot
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                sv[:, top_dims],
                X_exp_2d[:, top_dims],
                feature_names=[f"Dim {d}" for d in top_dims],
                show=False,
                plot_type="dot",
                max_display=top_k
            )
            plt.title("BiLSTM SHAP Summary Plot (Beeswarm) — Top 20 Embedding Dimensions")
            plt.tight_layout()
            shap_summary_path = EVAL_DIR / "classifier" / "bilstm_shap_summary.png"
            plt.savefig(shap_summary_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved SHAP summary plot → {shap_summary_path}")
            print("\n  Top 5 most influential embedding dimensions by mean |SHAP|:")
            for rank, dim in enumerate(top_dims[:5], 1):
                print(f"    #{rank}: Dim {dim:>3d}  (mean |SHAP| = {mean_abs_shap[dim]:.5f})")

    except ImportError as e:
        print(f"  SHAP not available: {e}. Run: pip install shap")
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")

    # ── Fairness bar chart (selection rate by borough) ───────────────────────
    print("\n--- FAIRNESS VISUALISATIONS ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pred_path_fair = EVAL_DIR / "classifier" / "test_predictions.csv"
        if pred_path_fair.exists():
            df_pred_fair = pd.read_csv(pred_path_fair)
            if "borough" in df_pred_fair.columns:
                boro_sr = (
                    df_pred_fair.groupby("borough")
                    .apply(lambda g: g["y_pred"].mean())
                    .reset_index(name="selection_rate")
                    .sort_values("selection_rate", ascending=True)
                )
                overall_mean = df_pred_fair["y_pred"].mean()
                colours = ["#ef5350" if v > overall_mean else "#42a5f5"
                           for v in boro_sr["selection_rate"]]
                plt.figure(figsize=(8, 5))
                plt.barh(boro_sr["borough"], boro_sr["selection_rate"] * 100, color=colours)
                plt.axvline(overall_mean * 100, color="orange", linestyle="--",
                            linewidth=1.5, label=f"Overall mean ({overall_mean*100:.1f}%)")
                plt.xlabel("Selection Rate (%)")
                plt.title("Classifier Selection Rate by Borough\n"
                          "(% of complaints flagged as Actionable Hazard)")
                plt.legend()
                plt.tight_layout()
                fair_bar_path = EVAL_DIR / "classifier" / "fairness_selection_rate_borough.png"
                plt.savefig(fair_bar_path, dpi=150)
                plt.close()
                print(f"  Saved fairness bar chart → {fair_bar_path}")
        else:
            print("  test_predictions.csv not found. Run notebook 04 first.")
    except Exception as e:
        print(f"  Fairness chart error: {e}")

else:
    print("Run notebooks 03 and 04 first.")

# Final results table
print("\nDISSERTATION TABLE")
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
print(f"Saved -> {EVAL_DIR}/final_results_summary.csv")
print("\nEvaluation complete")