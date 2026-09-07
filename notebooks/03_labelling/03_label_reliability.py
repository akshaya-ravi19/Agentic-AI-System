"""
notebooks/03_labelling/03c_label_reliability.py
---------------------------------------------------
Run this AFTER you've filled in manual_severity_label in
data/labelled/manual_review_sample.csv by hand.

Computes Cohen's Kappa between the automated distant-supervision
label ('label', from linked inspection outcomes) and your own manual
judgement ('manual_severity_label') on the held-out review sample.
This is the label-reliability number for your methodology chapter --
it tells you how much a human agrees with the automated labelling
rule, which matters because that rule is itself only a proxy for
true severity, not verified ground truth.

INTERPRETING THE KAPPA VALUE (standard Landis & Koch bands):
  < 0.00        no agreement
  0.00 - 0.20   slight
  0.21 - 0.40   fair
  0.41 - 0.60   moderate
  0.61 - 0.80   substantial
  0.81 - 1.00   almost perfect
Report both the number and the band in your write-up -- a raw kappa
number means little to a reader without that context.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import DATA_LABELLED

path = DATA_LABELLED / "manual_review_sample.csv"
df = pd.read_csv(path)

df["manual_severity_label"] = df["manual_severity_label"].astype(str).str.strip()
missing = df["manual_severity_label"].isin(["", "nan", "NaN"])
if missing.any():
    print(f"WARNING: {missing.sum()} of {len(df)} rows still have manual_severity_label "
          f"blank -- fill these in before trusting this result. Excluding them for now.")
    df = df[~missing]

if len(df) == 0:
    print("No labelled rows found. Fill in manual_severity_label in "
          f"{path} first, then re-run this script.")
    sys.exit(1)

df["manual_severity_label"] = df["manual_severity_label"].astype(int)
df["label"] = df["label"].astype(int)

kappa = cohen_kappa_score(df["label"], df["manual_severity_label"])

if kappa < 0:
    band = "no agreement (worse than chance)"
elif kappa <= 0.20:
    band = "slight agreement"
elif kappa <= 0.40:
    band = "fair agreement"
elif kappa <= 0.60:
    band = "moderate agreement"
elif kappa <= 0.80:
    band = "substantial agreement"
else:
    band = "almost perfect agreement"

print(f"\n{'='*50}")
print(f"LABEL RELIABILITY (n={len(df)})")
print(f"{'='*50}")
print(f"Cohen's Kappa: {kappa:.3f}  ->  {band}")

cm = confusion_matrix(df["label"], df["manual_severity_label"])
print("\nConfusion matrix (rows=automated label, cols=your manual label):")
print(f"                  manual=0   manual=1")
print(f"  automated=0     {cm[0][0]:>8}   {cm[0][1]:>8}")
print(f"  automated=1     {cm[1][0]:>8}   {cm[1][1]:>8}")

disagreements = df[df["label"] != df["manual_severity_label"]]
if len(disagreements):
    print(f"\n{len(disagreements)} disagreements -- worth quoting a couple in your "
          f"limitations discussion:")
    for _, row in disagreements.head(5).iterrows():
        print(f"  - {row.get('descriptor', '(no descriptor column)')!r}: "
              f"automated={row['label']}, you said={row['manual_severity_label']}")

out_path = DATA_LABELLED / "label_reliability_report.txt"
with open(out_path, "w") as f:
    f.write(f"Cohen's Kappa: {kappa:.3f} ({band})\n")
    f.write(f"n = {len(df)}\n")
    f.write(f"Confusion matrix:\n{cm}\n")
print(f"\nSaved summary to {out_path}")








"""
notebooks/03_labelling/03_ground_truth_construction.py
-------------------------------------------------------

STRONG VERSION: Expert-Validated Gold Standard + Weak Supervision

PURPOSE
-------
Construct a defensible ground-truth / gold-standard dataset for NYC
311 food-safety complaint prioritisation.

CORE DESIGN
-----------
We distinguish THREE different concepts:

1. GOLD STANDARD
   ----------------
   Human expert annotation of a stratified sample of 311 complaints.
   Multiple annotators independently assess complaint priority using
   a predefined rubric. Disagreements are subsequently adjudicated.

2. WEAK LABELS
   ------------
   A scalable rule-based taxonomy assigns priority tiers to the full
   311 corpus using NYC 311 complaint descriptors and complaint text.
   These labels are NOT called ground truth.

3. EXTERNAL VALIDATION
   -------------------
   DOHMH inspection outcomes (Grade C / closure / serious violations)
   are linked separately and used as an external validity criterion.
   They do NOT define the ground-truth label.

TARGET
------
Primary target:

    complaint_priority

    0 = Low / administrative or non-food-safety concern
    1 = Moderate food-safety concern
    2 = High-priority food-safety concern

For binary experiments:

    priority_binary = 1 if complaint_priority == 2
                      0 otherwise

IMPORTANT METHODOLOGICAL RULE
-----------------------------
The human gold-standard label must be assigned independently of:
    - DOHMH inspection outcomes
    - Grade
    - Closure
    - CAMIS
    - automatically generated weak labels
    - model predictions

This prevents circular validation / target leakage.

WORKFLOW
--------
STEP 1  Load and clean 311 complaints
STEP 2  Construct a documented rule-based weak-label taxonomy
STEP 3  Create a stratified human-review sample
STEP 4  Export blinded annotation file
STEP 5  Human annotators independently assign labels
STEP 6  Calculate inter-rater agreement
STEP 7  Adjudicate disagreements
STEP 8  Freeze final GOLD STANDARD
STEP 9  Compare weak labels against gold standard
STEP 10 Save weak-labelled full corpus + gold standard

DOHMH OUTCOMES
--------------
DOHMH data should be linked in a SEPARATE notebook after the gold
standard is frozen. This avoids contaminating the human annotation
process.

OUTPUTS
-------
data/labelled/
    weak_labelled_complaints.csv
    gold_standard_annotation_sample.csv
    gold_standard_final.csv
    gold_standard_disagreements.csv

NOTE
----
This notebook constructs an expert-validated gold-standard subset.
It does not claim that the automatically generated labels are
ground truth.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Optional agreement metrics
try:
    from sklearn.metrics import (
        cohen_kappa_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.config import (
    DATA_RAW,
    DATA_LABELLED,
    RANDOM_SEED,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of complaints to send for manual annotation.
# Increase if annotation resources allow.
GOLD_STANDARD_SIZE = 1500

# Number of independent annotators.
# Recommended: 3.
N_ANNOTATORS = 3

# Minimum number of complaints per weak-label stratum.
MIN_PER_STRATUM = 100

# Random seed for reproducibility.
SEED = RANDOM_SEED


# ============================================================
# STEP 1 — LOAD NYC 311 DATA
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING NYC 311 FOOD COMPLAINTS")
print("=" * 70)

df = pd.read_csv(
    DATA_RAW / "nyc_311_food_complaints.csv",
    parse_dates=["created_date"],
    dtype={
        "unique_key": str,
        "incident_zip": str,
    },
    low_memory=False,
)

print(f"Loaded: {len(df):,} complaints")

# Keep complaints with a usable descriptor.
df = df.dropna(subset=["descriptor"]).copy()

# Make optional columns safe.
for col in [
    "descriptor_2",
    "resolution_description",
    "incident_address",
    "incident_zip",
    "borough",
]:
    if col not in df.columns:
        df[col] = ""

    df[col] = df[col].fillna("")

print(f"After descriptor filtering: {len(df):,}")


# ============================================================
# STEP 2 — CONSTRUCT COMPLAINT TEXT
# ============================================================
#
# IMPORTANT:
# Only fields available at / near complaint submission should be
# considered model inputs.
#
# resolution_description is deliberately NOT included because it
# may contain information generated after the complaint was filed.
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — CONSTRUCTING COMPLAINT REPRESENTATION")
print("=" * 70)

df["complaint_text"] = (
    df["descriptor"].astype(str).str.strip()
    + " | "
    + df["descriptor_2"].astype(str).str.strip()
)

df["complaint_text"] = (
    df["complaint_text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip(" |")
)

print("Complaint representation created from:")
print("  - descriptor")
print("  - descriptor_2")
print("Excluded from label construction:")
print("  - resolution_description")


# ============================================================
# STEP 3 — DEFINE WEAK-LABEL PRIORITY TAXONOMY
# ============================================================
#
# IMPORTANT:
# These are WEAK LABELS, not ground truth.
#
# The taxonomy should be documented BEFORE model training and,
# ideally, reviewed by a food-safety/public-health expert.
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — APPLYING WEAK-LABEL PRIORITY TAXONOMY")
print("=" * 70)


# ------------------------------------------------------------
# HIGH PRIORITY
# ------------------------------------------------------------
#
# These represent complaints that potentially indicate an
# immediate or substantial food-safety/public-health concern.
# ------------------------------------------------------------

HIGH_PRIORITY_DESCRIPTORS = {
    "Rodents/Insects/Garbage",
    "Food Temperature",
    "Food Contaminated",
    "Food Spoiled",
    "Sewage",
    "Food Source",
    "Cross Contamination",
}


# ------------------------------------------------------------
# MODERATE PRIORITY
# ------------------------------------------------------------

MODERATE_PRIORITY_DESCRIPTORS = {
    "Bare Hands in Contact w/ Food",
    "Food Worker Hygiene",
    "Food Worker Activity",
    "Kitchen/Food Prep Area",
    "Food Protection",
    "Food Preparation Location",
    "Dishwashing/Utensils",
    "Handwashing",
    "Food Contains Foreign Object",
}


# ------------------------------------------------------------
# HIGH-PRIORITY KEYWORDS
# ------------------------------------------------------------
#
# These should be treated as supplementary rules only.
# Descriptor categories take precedence.
#
# Avoid simple medical words such as "sick" on their own because
# they can create false positives through negation/context.
# ------------------------------------------------------------

HIGH_PRIORITY_KEYWORDS = [
    "food poisoning",
    "food poisoned",
    "vomiting",
    "diarrhea",
    "diarrhoea",
    "hospitalized",
    "hospital",
    "undercooked chicken",
    "raw chicken",
    "raw meat",
    "rodent",
    "mice",
    "rats",
    "roach",
    "cockroach",
    "sewage",
    "chemical contamination",
    "pesticide",
    "mold on food",
    "food contaminated",
]


# ------------------------------------------------------------
# MODERATE-PRIORITY KEYWORDS
# ------------------------------------------------------------

MODERATE_PRIORITY_KEYWORDS = [
    "bare hand",
    "no gloves",
    "dirty kitchen",
    "unsanitary",
    "cross contamination",
    "cutting board",
    "food handling",
    "handwashing",
    "flies",
    "insects",
]


def assign_weak_priority(row):
    """
    Assign a rule-based WEAK LABEL.

    Returns
    -------
    priority : int
        0 = Low
        1 = Moderate
        2 = High

    source : str
        Explanation for the weak label.

    IMPORTANT
    ---------
    This function DOES NOT create ground truth.
    It creates scalable weak supervision labels.
    """

    descriptor = str(row.get("descriptor", "")).strip()
    text = str(row.get("complaint_text", "")).lower()

    # --------------------------------------------------------
    # Descriptor-based rules
    # --------------------------------------------------------

    if descriptor in HIGH_PRIORITY_DESCRIPTORS:
        return 2, "descriptor_high"

    if descriptor in MODERATE_PRIORITY_DESCRIPTORS:
        return 1, "descriptor_moderate"

    # --------------------------------------------------------
    # Keyword fallback
    # --------------------------------------------------------
    #
    # Use conservative keyword matching.
    # --------------------------------------------------------

    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in text:
            return 2, f"keyword_high:{keyword}"

    for keyword in MODERATE_PRIORITY_KEYWORDS:
        if keyword in text:
            return 1, f"keyword_moderate:{keyword}"

    return 0, "default_low"


weak_results = df.apply(assign_weak_priority, axis=1)

df["weak_priority"] = [x[0] for x in weak_results]
df["weak_label_source"] = [x[1] for x in weak_results]

df["weak_priority_binary"] = (
    df["weak_priority"] >= 2
).astype(int)


# ============================================================
# STEP 4 — INSPECT WEAK-LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — WEAK-LABEL DISTRIBUTION")
print("=" * 70)

tier_names = {
    0: "Low",
    1: "Moderate",
    2: "High",
}

tier_counts = (
    df["weak_priority"]
    .value_counts()
    .sort_index()
)

for tier, count in tier_counts.items():
    print(
        f"  Tier {tier} ({tier_names.get(tier, 'Unknown')}): "
        f"{count:,} ({count / len(df):.1%})"
    )

print("\nWeak-label source:")
print(
    df["weak_label_source"]
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# STEP 5 — CREATE STRATIFIED GOLD-STANDARD SAMPLE
# ============================================================
#
# IMPORTANT:
# The sample should NOT simply be a random sample of all complaints.
#
# We want representation from:
#   - each weak-priority tier
#   - common complaint descriptors
#
# This ensures that the gold standard contains both easy and
# difficult cases.
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — CREATING STRATIFIED GOLD-STANDARD SAMPLE")
print("=" * 70)


rng = np.random.default_rng(SEED)

# ------------------------------------------------------------
# Primary stratification variable
# ------------------------------------------------------------

df["_stratum"] = (
    "weak_"
    + df["weak_priority"].astype(str)
)

# Desired sample size per tier.
available_tiers = sorted(df["weak_priority"].unique())

n_tiers = len(available_tiers)

base_per_tier = max(
    MIN_PER_STRATUM,
    GOLD_STANDARD_SIZE // n_tiers
)

samples = []

for tier in available_tiers:

    subset = df[df["weak_priority"] == tier].copy()

    n = min(
        base_per_tier,
        len(subset),
    )

    if n > 0:
        sampled = subset.sample(
            n=n,
            random_state=SEED + int(tier),
        )

        samples.append(sampled)

gold_sample = pd.concat(
    samples,
    ignore_index=True,
)


# ------------------------------------------------------------
# Fill remaining slots randomly if necessary
# ------------------------------------------------------------

remaining_n = GOLD_STANDARD_SIZE - len(gold_sample)

if remaining_n > 0:

    remaining_ids = set(gold_sample["unique_key"])

    remaining = df[
        ~df["unique_key"].isin(remaining_ids)
    ].copy()

    if len(remaining) > 0:

        n = min(
            remaining_n,
            len(remaining),
        )

        additional = remaining.sample(
            n=n,
            random_state=SEED + 999,
        )

        gold_sample = pd.concat(
            [gold_sample, additional],
            ignore_index=True,
        )


gold_sample = gold_sample.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

print(
    f"Gold-standard sample created: "
    f"{len(gold_sample):,} complaints"
)

print("\nGold sample by weak tier:")

print(
    gold_sample["weak_priority"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# STEP 6 — CREATE BLINDED ANNOTATION FILE
# ============================================================
#
# The human annotator must NOT see:
#   - weak_priority
#   - weak_label_source
#   - DOHMH information
#   - CAMIS
#   - model predictions
#
# The annotation file therefore deliberately excludes those fields.
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — CREATING BLINDED ANNOTATION FILE")
print("=" * 70)


annotation = gold_sample[
    [
        c
        for c in [
            "unique_key",
            "created_date",
            "descriptor",
            "descriptor_2",
            "complaint_text",
            "incident_address",
            "incident_zip",
            "borough",
        ]
        if c in gold_sample.columns
    ]
].copy()


# ------------------------------------------------------------
# Add independent annotator columns.
#
# Leave these blank.
# ------------------------------------------------------------

for annotator in range(1, N_ANNOTATORS + 1):

    annotation[
        f"annotator_{annotator}_priority"
    ] = ""

    annotation[
        f"annotator_{annotator}_notes"
    ] = ""


annotation["adjudicated_priority"] = ""
annotation["adjudicator_notes"] = ""


# ------------------------------------------------------------
# Human annotation rubric
# ------------------------------------------------------------

annotation["ANNOTATION_INSTRUCTIONS"] = (
    "Assign complaint priority using the following rubric: "
    "0=Low/administrative or no clear food-safety hazard; "
    "1=Moderate food-safety concern requiring attention; "
    "2=High-priority concern indicating a potentially substantial "
    "or immediate food-safety/public-health risk. "
    "Judge the complaint itself. Do NOT use DOHMH inspection "
    "outcomes, grades, closures, CAMIS, or automated labels."
)


annotation_path = (
    DATA_LABELLED
    / "gold_standard_annotation_sample.csv"
)

DATA_LABELLED.mkdir(
    parents=True,
    exist_ok=True,
)

annotation.to_csv(
    annotation_path,
    index=False,
)

print(
    f"Saved blinded annotation file:\n"
    f"  {annotation_path}"
)

print("\nNEXT STEP:")
print(
    "Have independent annotators complete the "
    "annotator_1/2/3_priority columns."
)

print(
    "\nIMPORTANT: Do not expose weak_priority or DOHMH "
    "inspection outcomes to annotators."
)


# ============================================================
# STOP POINT
# ============================================================
#
# At this point the automated pipeline STOPS.
#
# Human annotators must complete:
#
#   annotator_1_priority
#   annotator_2_priority
#   annotator_3_priority
#
# Then save the completed file.
#
# Only AFTER independent annotation should the following
# section be executed.
# ============================================================

print("\n" + "=" * 70)
print("MANUAL ANNOTATION REQUIRED")
print("=" * 70)

print(
    """
Complete the annotation file manually before continuing.

Required columns:
    annotator_1_priority
    annotator_2_priority
    annotator_3_priority

Valid values:
    0 = Low
    1 = Moderate
    2 = High

Do not enter:
    weak labels
    DOHMH outcomes
    model predictions
    CAMIS information
"""
)


# ============================================================
# STEP 7 — LOAD COMPLETED ANNOTATION FILE
# ============================================================
#
# The same file is re-read after human annotation.
# ============================================================

completed_annotation_path = (
    DATA_LABELLED
    / "gold_standard_annotation_completed.csv"
)

if not completed_annotation_path.exists():

    print(
        "\n[STOP] Completed annotation file not found."
    )

    print(
        "\nAfter annotation, save the completed file as:"
    )

    print(
        f"  {completed_annotation_path}"
    )

    print(
        "\nThen rerun this notebook from STEP 7."
    )

else:

    print("\n" + "=" * 70)
    print("STEP 7 — ANALYSING HUMAN ANNOTATION")
    print("=" * 70)

    annotated = pd.read_csv(
        completed_annotation_path,
        dtype={"unique_key": str},
    )

    annotator_cols = [
        f"annotator_{i}_priority"
        for i in range(1, N_ANNOTATORS + 1)
    ]

    # Convert to numeric.
    for col in annotator_cols:

        annotated[col] = pd.to_numeric(
            annotated[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Validate values
    # --------------------------------------------------------

    invalid_values = {}

    for col in annotator_cols:

        bad = annotated[
            annotated[col].notna()
            & ~annotated[col].isin([0, 1, 2])
        ]

        if len(bad) > 0:
            invalid_values[col] = len(bad)

    if invalid_values:

        print(
            "\nWARNING: Invalid annotation values found:"
        )

        print(invalid_values)

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    print("\nAnnotation counts:")

    for col in annotator_cols:

        print(
            f"\n{col}:"
        )

        print(
            annotated[col]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )


    # ========================================================
    # STEP 8 — INTER-RATER AGREEMENT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 8 — INTER-RATER AGREEMENT")
    print("=" * 70)

    if SKLEARN_AVAILABLE:

        for i in range(len(annotator_cols)):

            for j in range(i + 1, len(annotator_cols)):

                a = annotator_cols[i]
                b = annotator_cols[j]

                valid = (
                    annotated[a].notna()
                    & annotated[b].notna()
                )

                if valid.sum() == 0:
                    continue

                kappa = cohen_kappa_score(
                    annotated.loc[valid, a],
                    annotated.loc[valid, b],
                )

                print(
                    f"{a} vs {b}: "
                    f"Cohen's kappa = {kappa:.3f}"
                )

    else:

        print(
            "scikit-learn unavailable; "
            "agreement metrics skipped."
        )


    # ========================================================
    # STEP 9 — ADJUDICATION
    # ========================================================
    #
    # Majority vote is calculated as a transparent starting point.
    #
    # If all three annotators disagree, an expert adjudicator
    # MUST review the case.
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 9 — ADJUDICATION")
    print("=" * 70)

    annotated["human_vote_count"] = (
        annotated[annotator_cols]
        .notna()
        .sum(axis=1)
    )

    annotated["human_priority_mode"] = (
        annotated[annotator_cols]
        .mode(axis=1, dropna=True)
        .iloc[:, 0]
    )


    def requires_adjudication(row):

        values = [
            row[col]
            for col in annotator_cols
            if pd.notna(row[col])
        ]

        return len(set(values)) > 1


    annotated["requires_adjudication"] = (
        annotated.apply(
            requires_adjudication,
            axis=1,
        )
    )

    disagreements = annotated[
        annotated["requires_adjudication"]
    ].copy()

    print(
        f"Total annotated: "
        f"{len(annotated):,}"
    )

    print(
        f"Disagreements requiring review: "
        f"{len(disagreements):,}"
    )

    # --------------------------------------------------------
    # Save disagreements for expert adjudication.
    # --------------------------------------------------------

    disagreement_path = (
        DATA_LABELLED
        / "gold_standard_disagreements.csv"
    )

    disagreement_cols = [
        c
        for c in [
            "unique_key",
            "created_date",
            "descriptor",
            "descriptor_2",
            "complaint_text",
            "annotator_1_priority",
            "annotator_2_priority",
            "annotator_3_priority",
            "human_priority_mode",
            "adjudicated_priority",
            "adjudicator_notes",
        ]
        if c in annotated.columns
    ]

    disagreements[
        disagreement_cols
    ].to_csv(
        disagreement_path,
        index=False,
    )

    print(
        f"Saved disagreement cases to:\n"
        f"  {disagreement_path}"
    )

    print(
        "\nAn expert adjudicator should complete "
        "adjudicated_priority for these cases."
    )


    # ========================================================
    # STEP 10 — CONSTRUCT FINAL GOLD STANDARD
    # ========================================================
    #
    # If annotators agree:
    #
    #       majority vote = final label
    #
    # If they disagree:
    #
    #       expert adjudication = final label
    #
    # No DOHMH information is used.
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 10 — CONSTRUCTING FINAL GOLD STANDARD")
    print("=" * 70)


    annotated["gold_priority"] = np.nan

    # Cases where all annotators agree.
    unanimous = (
        annotated[annotator_cols]
        .nunique(axis=1, dropna=True)
        == 1
    )

    annotated.loc[
        unanimous,
        "gold_priority"
    ] = (
        annotated.loc[
            unanimous,
            annotator_cols
        ]
        .iloc[:, 0]
    )


    # For disagreement cases, use adjudicated label.
    if "adjudicated_priority" in annotated.columns:

        annotated["adjudicated_priority"] = pd.to_numeric(
            annotated["adjudicated_priority"],
            errors="coerce",
        )

        annotated.loc[
            ~unanimous
            & annotated["adjudicated_priority"].notna(),
            "gold_priority"
        ] = annotated.loc[
            ~unanimous
            & annotated["adjudicated_priority"].notna(),
            "adjudicated_priority"
        ]


    # --------------------------------------------------------
    # Check for unresolved cases.
    # --------------------------------------------------------

    unresolved = annotated[
        annotated["gold_priority"].isna()
    ].copy()

    print(
        f"Final gold-standard labels available: "
        f"{annotated['gold_priority'].notna().sum():,}"
    )

    print(
        f"Unresolved cases: "
        f"{len(unresolved):,}"
    )

    if len(unresolved) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Some disagreement cases have not yet "
            "been adjudicated."
        )

        print(
            "Do NOT use them in final model evaluation "
            "until adjudication is complete."
        )


    # ========================================================
    # STEP 11 — VALIDATE WEAK LABELS AGAINST GOLD STANDARD
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 11 — WEAK LABEL VS GOLD STANDARD")
    print("=" * 70)

    valid_gold = annotated[
        annotated["gold_priority"].notna()
    ].copy()

    # Merge weak labels.
    weak_lookup = df[
        [
            "unique_key",
            "weak_priority",
            "weak_priority_binary",
        ]
    ].copy()

    valid_gold = valid_gold.merge(
        weak_lookup,
        on="unique_key",
        how="left",
    )

    if len(valid_gold) > 0:

        print(
            "\nGold-standard class distribution:"
        )

        print(
            valid_gold["gold_priority"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\nWeak-label class distribution:"
        )

        print(
            valid_gold["weak_priority"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        if SKLEARN_AVAILABLE:

            y_true = valid_gold[
                "gold_priority"
            ].astype(int)

            y_pred = valid_gold[
                "weak_priority"
            ].astype(int)

            print(
                "\nWeak-label classification report:"
            )

            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1, 2],
                    target_names=[
                        "Low",
                        "Moderate",
                        "High",
                        ],
                    zero_division=0,
                )
            )

            print(
                "\nConfusion matrix:"
            )

            print(
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1, 2],
                )
            )

            # Binary high-priority comparison.
            gold_binary = (
                y_true == 2
            ).astype(int)

            weak_binary = (
                y_pred == 2
            ).astype(int)

            print(
                "\nBinary high-priority validation:"
            )

            print(
                f"Precision: "
                f"{precision_score(gold_binary, weak_binary, zero_division=0):.3f}"
            )

            print(
                f"Recall: "
                f"{recall_score(gold_binary, weak_binary, zero_division=0):.3f}"
            )

            print(
                f"F1: "
                f"{f1_score(gold_binary, weak_binary, zero_division=0):.3f}"
            )

            print(
                f"Cohen's kappa: "
                f"{cohen_kappa_score(gold_binary, weak_binary):.3f}"
            )


    # ========================================================
    # STEP 12 — SAVE FINAL GOLD STANDARD
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 12 — SAVING FINAL GOLD STANDARD")
    print("=" * 70)

    gold_cols = [
        c
        for c in [
            "unique_key",
            "created_date",
            "descriptor",
            "descriptor_2",
            "complaint_text",
            "incident_address",
            "incident_zip",
            "borough",
            "annotator_1_priority",
            "annotator_2_priority",
            "annotator_3_priority",
            "human_priority_mode",
            "adjudicated_priority",
            "gold_priority",
            "requires_adjudication",
        ]
        if c in annotated.columns
    ]

    final_gold = annotated[
        annotated["gold_priority"].notna()
    ][gold_cols].copy()

    gold_path = (
        DATA_LABELLED
        / "gold_standard_final.csv"
    )

    final_gold.to_csv(
        gold_path,
        index=False,
    )

    print(
        f"\n[SUCCESS] Gold standard saved:"
    )

    print(
        f"  {gold_path}"
    )

    print(
        f"  N = {len(final_gold):,}"
    )


# ============================================================
# STEP 13 — SAVE FULL WEAK-LABELLED DATASET
# ============================================================
#
# This dataset is intended for training / development.
# It is NOT the gold standard.
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 — SAVING FULL WEAK-LABELLED DATASET")
print("=" * 70)


weak_output_cols = [
    c
    for c in [
        "unique_key",
        "created_date",
        "descriptor",
        "descriptor_2",
        "complaint_text",
        "incident_address",
        "incident_zip",
        "borough",
        "weak_priority",
        "weak_priority_binary",
        "weak_label_source",
    ]
    if c in df.columns
]


weak_output_path = (
    DATA_LABELLED
    / "weak_labelled_complaints.csv"
)

df[
    weak_output_cols
].to_csv(
    weak_output_path,
    index=False,
)

print(
    f"Saved weak-labelled corpus:"
)

print(
    f"  {weak_output_path}"
)

print(
    f"  N = {len(df):,}"
)


# ============================================================
# FINAL METHODOLOGICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GROUND-TRUTH CONSTRUCTION COMPLETE")
print("=" * 70)

print(
    """
FINAL DATASETS
--------------

1. weak_labelled_complaints.csv
   --------------------------------
   Full 311 corpus with rule-based priority labels.

   USE:
       Model training / weak supervision.

   DO NOT CALL:
       Ground truth.


2. gold_standard_annotation_sample.csv
   -------------------------------------
   Blinded sample exported for human annotation.

   USE:
       Independent human annotation.


3. gold_standard_disagreements.csv
   ---------------------------------
   Cases requiring expert adjudication.

   USE:
       Final consensus process.


4. gold_standard_final.csv
   ------------------------
   Human-validated and adjudicated labels.

   USE:
       FINAL MODEL EVALUATION.

   This is the dataset that should be referred to as
   the GOLD STANDARD.


IMPORTANT
---------

DOHMH inspection outcomes are intentionally absent from the
human annotation process.

They should be linked in a later analysis as an EXTERNAL
VALIDATION criterion.

Recommended final architecture:

    Weak-labelled corpus
          |
          v
    Model training
          |
          v
    Model predictions
          |
          v
    GOLD STANDARD
    (human adjudication)
          |
          v
    Final evaluation

Separate analysis:

    Gold-standard priority
            |
            v
    DOHMH C / closure / violations

to assess external validity.
"""
)