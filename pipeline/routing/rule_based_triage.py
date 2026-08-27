"""
pipeline/routing/rule_based_triage.py
-----------------------------------------
Deterministic, non-LLM triage baseline. Built independently of the
LangGraph agent for two reasons (see project point 11 -- baseline
built early, not last):

  1. It's what the agent gets benchmarked AGAINST in notebook 07's
     evaluation (agent vs. rule-based comparison) -- you need this
     running before that comparison means anything.
  2. pipeline/agent/service.py uses it as a deterministic FALLBACK
     when the LLM's response can't be parsed for a clear tier, so a
     complaint never silently falls through with no triage decision
     at all.

Kept deliberately simple and auditable -- a human should be able to
read this function top to bottom and know exactly why any given
complaint got the tier it got, no black box involved.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE


def rule_based_triage(bilstm_pred: int, last_grade: str,
                       days_since_inspection: int, complaint_count_30d: int) -> str:
    """
    Parameters
    ----------
    bilstm_pred : 1 if the classifier predicted "severe", else 0.
    last_grade : most recent DOHMH grade for this establishment
                 (e.g. "A", "B", "C", "Z", "Not Yet Graded", or "").
    days_since_inspection : days since the most recent inspection.
                             A large number (e.g. 999) signals
                             "unknown" -- treated conservatively.
    complaint_count_30d : number of complaints against this
                           establishment in the last 30 days.

    Returns
    -------
    One of TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE.
    """
    last_grade = (last_grade or "").strip().upper()

    # ESCALATE: classifier flags severe AND there's independent
    # corroborating evidence (a recent bad grade, or a repeat-offender
    # pattern of complaints) -- mirrors the agent's own evidence-based
    # reasoning, just with fixed thresholds instead of an LLM.
    if bilstm_pred == 1 and (last_grade == "C" or complaint_count_30d >= 3):
        return TRIAGE_ESCALATE

    # REVIEW: any one red flag on its own -- severe prediction alone,
    # a C grade on record, a stale inspection (unknown/very old, i.e.
    # not recently verified safe), or a moderate complaint pattern.
    if bilstm_pred == 1 or last_grade == "C" or days_since_inspection >= 365 \
            or complaint_count_30d >= 2:
        return TRIAGE_REVIEW

    # LOG: no red flags -- routine logging only.
    return TRIAGE_LOG
