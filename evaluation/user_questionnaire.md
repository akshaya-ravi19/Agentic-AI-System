# User Evaluation Questionnaire — FoodGuard Triage System

Administer this to Environmental Health Officers / DOHMH inspectors
(or domain-knowledgeable proxies, if real inspectors aren't
accessible for a dissertation timeline — note which in your
methodology chapter, since it affects how you can generalise the
results).

**Format:** show each reviewer a set of sample complaints alongside
the agent's triage recommendation and supporting evidence (pull
these from `evaluation/agent/agent_results.csv`, the same set used
for the Cohen's Kappa comparison in notebook 07). 8-12 cases is a
reasonable number to keep the session under 30 minutes.

## Section A — Per-case ratings (repeat for each case shown)

For each case, ask the reviewer to rate 1 (strongly disagree) to 5
(strongly agree):

1. The recommended triage tier (LOG / REVIEW / ESCALATE) is
   appropriate for this complaint.
2. The evidence cited (inspection history, cluster context) is
   relevant and sufficient to support the recommendation.
3. I would trust this recommendation enough to act on it without
   independently re-checking the underlying records.
4. The reasoning is clearly explained, not just a bare tier label.

Plus one open-ended question per case:
5. What, if anything, would you have decided differently, and why?

## Section B — Overall system impressions (asked once, after all cases)

Rate 1-5 unless noted:

6. Overall, this system would help me prioritise my inspection
   workload more effectively than my current process.
7. I am concerned this system could miss genuinely severe cases
   (false negatives).
8. I am concerned this system could flood me with false alarms
   (false positives).
9. The system's explanations give me enough information to override
   its recommendation when I disagree.
10. I would be comfortable using a system like this as a supporting
    tool in my actual work. (Yes / No / Unsure — with a follow-up
    "why" if No or Unsure)

Open-ended:
11. What's the single most useful thing about this system?
12. What's the single most concerning or unhelpful thing about it?
13. Any patterns you noticed across cases (e.g. certain cuisines,
    neighbourhoods, or complaint types where it seemed to do
    noticeably better or worse)?

## Section C — Reviewer background (for reporting response context)

- Years of experience in food safety inspection / public health:
- Primary borough(s)/area(s) typically covered:
- Prior familiarity with AI/ML-based tools (none / some / extensive):

## Analysis notes for your dissertation

- Report Section A ratings both overall and broken down by triage
  tier (agreement may differ a lot between LOG and ESCALATE cases —
  worth showing separately, not just as one pooled average).
- Cross-reference open-ended answers (Q5, Q13) against the
  fairness/bias audit in notebook 07 (Section 5) — if reviewers
  independently flag the same subgroup patterns SHAP/Fairlearn
  surfaced statistically, that's a strong corroborating finding
  worth highlighting.
- If you only get a small number of reviewers (common for a
  dissertation timeline), say so explicitly and treat this as
  qualitative/exploratory evidence, not a statistically powered
  study — don't overclaim generalisability from n=2-3 reviewers.
