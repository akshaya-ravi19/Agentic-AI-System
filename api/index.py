"""
api/index.py
------------
Serverless Entrypoint for the Digital Health Syndromic Triage Platform.

Optimized for Cloud and Serverless Functions:
1. Fast, lightweight public health syndromic hazard scoring (CDC FoodNet & FDA Model Code criteria)
2. Interactive Digital Health Web UI & REST API (/health, /investigate, /)
"""
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Digital Health Syndromic Triage Platform")

# ── Syndromic Hazard Taxonomy ──────────────────────────────────
CRITICAL_HAZARDS = [
    "food poisoned", "food poisoning", "vomit", "sick", "diarrhea", "fever", "nausea",
    "rodent infestation", "mice", "rats", "roaches", "pesticide", "chemical", "sewage",
    "temperature", "undercooked", "raw chicken", "raw meat", "spoilage"
]
MODERATE_HAZARDS = [
    "food spoiled", "food contaminated", "food contains foreign object",
    "bare hands in contact w/ food", "food worker hygiene", "kitchen/food prep area",
    "unsanitary condition", "insects", "flies", "filth flies", "glove"
]

def compute_syndromic_score(text: str, category: Optional[str] = None) -> tuple[float, str]:
    combined = f"{category or ''} {text}".lower()
    for kw in CRITICAL_HAZARDS:
        if kw in combined:
            return 0.88, "CRITICAL PATHOGEN / IMMINENT BIOHAZARD"
    for kw in MODERATE_HAZARDS:
        if kw in combined:
            return 0.52, "MODERATE HYGIENE / CONTAMINATION HAZARD"
    return 0.18, "ROUTINE / ADMINISTRATIVE"


class InvestigateRequest(BaseModel):
    complaint_id: Optional[str] = "SYNDROMIC-DEMO"
    category: Optional[str] = None
    restaurant_name: Optional[str] = None
    location: Optional[str] = None
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "framework": "Digital Health Syndromic Surveillance"
    }


@app.post("/investigate")
def investigate(req: InvestigateRequest):
    if not req.text.strip() and not (req.category and req.category.strip()):
        raise HTTPException(status_code=400, detail="Please enter a complaint description or select a category.")

    score, hazard_flag = compute_syndromic_score(req.text, req.category)
    
    # Epidemiological Reasoning & Triage
    est_desc = []
    if req.restaurant_name:
        est_desc.append(f"Name: {req.restaurant_name}")
    if req.location:
        est_desc.append(f"Location: {req.location}")
    est_str = ", ".join(est_desc) if est_desc else "Unspecified"

    tier = "LOG"
    if score >= 0.70:
        tier = "ESCALATE"
    elif score >= 0.45:
        tier = "REVIEW"

    # Action Directives
    if tier == "ESCALATE":
        action_directive = (
            "Critical biological contamination or acute foodborne pathogen symptoms detected. "
            "Prioritize for immediate environmental health on-site inspection within 48 hours to mitigate community transmission."
        )
    elif tier == "REVIEW":
        action_directive = (
            "Secondary hygiene failure or physical contamination hazard identified. "
            "Schedule for secondary regulatory review within 5 business days."
        )
    else:
        action_directive = (
            "Routine non-pathogenic or administrative complaint. Log for next regular cycle inspection."
        )

    return {
        "complaint_id": req.complaint_id,
        "category": req.category or "Not Specified",
        "establishment": est_str,
        "complaint_summary": req.text,
        "syndromic_score": score,
        "hazard_flag": hazard_flag,
        "triage_level": tier,
        "action_directive": action_directive,
        "surveillance_framework": "Aligned with CDC FoodNet Priority Syndromes & FDA Model Food Code"
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Digital Health Syndromic Surveillance Platform</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #0b132b; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            .card { background-color: #1c2541; border: 1px solid #3a506b; border-radius: 12px; }
            .badge-LOG { background-color: #10b981; }
            .badge-REVIEW { background-color: #f59e0b; color: #000; }
            .badge-ESCALATE { background-color: #ef4444; }
            .btn-primary { background-color: #00b4d8; border-color: #00b4d8; color: #03045e; font-weight: 700; }
            .btn-primary:hover { background-color: #90e0ef; border-color: #90e0ef; color: #03045e; }
            .section-title { color: #48cae4; font-weight: 700; border-bottom: 2px solid #0096c7; padding-bottom: 8px; display: inline-block; }
            
            /* Enhanced Assessment Styling without markdown hashes/asterisks */
            .assessment-section { background-color: #0d1b2a; border-radius: 10px; padding: 20px; border: 1px solid #283e58; }
            .assessment-header { font-size: 1.15rem; font-weight: 600; color: #38bdf8; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; margin-bottom: 16px; }
            .item-row { display: flex; margin-bottom: 10px; font-size: 0.95rem; }
            .item-label { color: #94a3b8; width: 220px; flex-shrink: 0; font-weight: 600; }
            .item-value { color: #f1f5f9; flex-grow: 1; }
            .highlight-box { background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-top: 14px; }
            .highlight-box.critical { background: rgba(239, 68, 68, 0.1); border-left-color: #ef4444; }
            .highlight-box.review { background: rgba(245, 158, 11, 0.1); border-left-color: #f59e0b; }
            .highlight-box.log { background: rgba(16, 185, 129, 0.1); border-left-color: #10b981; }
        </style>
    </head>
    <body class="py-5">
        <div class="container" style="max-width: 950px;">
            <div class="text-center mb-4">
                <h2 class="fw-bold text-light">Digital Health Syndromic Surveillance & Triage</h2>
                <p class="text-secondary mb-1">Computational Public Health Intelligence · Early Outbreak Detection · Environmental Health CDSS</p>
                <span class="badge bg-info bg-opacity-25 text-info border border-info border-opacity-25 px-3 py-1">CDC & FDA Aligned</span>
            </div>

            <div class="card p-4 shadow-sm mb-4">
                <div class="mb-3">
                    <h5 class="section-title mb-0">Submit Syndromic Complaint for Public Health Triage</h5>
                </div>

                <div class="mb-3">
                    <label class="form-label text-secondary fw-semibold">Complaint Category</label>
                    <select id="categorySelect" class="form-select bg-dark text-light border-secondary">
                        <option value="">Select an affected category...</option>
                        <optgroup label="Acute Health & Pathogen Symptoms">
                            <option value="Food Poisoning / Acute Illness (Vomiting, Diarrhea, Fever)">Food Poisoning / Acute Illness (Vomiting, Diarrhea, Fever)</option>
                            <option value="Nausea / Stomach Cramps after meal">Nausea / Stomach Cramps after meal</option>
                            <option value="Undercooked Meat / Raw Poultry Hazard">Undercooked Meat / Raw Poultry Hazard</option>
                            <option value="Chemical / Pesticide Contamination">Chemical / Pesticide Contamination</option>
                        </optgroup>
                        <optgroup label="Biological & Pest Infestation">
                            <option value="Rodents / Mice / Rats Infestation">Rodents / Mice / Rats Infestation</option>
                            <option value="Roaches / Pest Contamination in kitchen">Roaches / Pest Contamination in kitchen</option>
                            <option value="Filth Flies / Insects on food">Filth Flies / Insects on food</option>
                            <option value="Sewage / Drainage Backup">Sewage / Drainage Backup</option>
                        </optgroup>
                        <optgroup label="Food Storage & Contamination">
                            <option value="Food Temperature Abuse / Spoiled Food">Food Temperature Abuse / Spoiled Food</option>
                            <option value="Food Contains Foreign Object">Food Contains Foreign Object</option>
                            <option value="Cross-Contamination in food preparation">Cross-Contamination in food preparation</option>
                        </optgroup>
                        <optgroup label="Hygiene & Sanitation">
                            <option value="Bare Hands in Contact with Ready-to-Eat Food">Bare Hands in Contact with Ready-to-Eat Food</option>
                            <option value="Food Worker Hygiene / Lack of Gloves">Food Worker Hygiene / Lack of Gloves</option>
                            <option value="Unsanitary Kitchen / Prep Surface Condition">Unsanitary Kitchen / Prep Surface Condition</option>
                        </optgroup>
                        <optgroup label="Facility & Routine Operations">
                            <option value="Facility Condition / Odor">Facility Condition / Odor</option>
                            <option value="Pet / Animal Present in Dining Area">Pet / Animal Present in Dining Area</option>
                            <option value="Letter Grade Missing / Inspection Certificate">Letter Grade Missing / Inspection Certificate</option>
                            <option value="General Sanitation Inconvenience">General Sanitation Inconvenience</option>
                        </optgroup>
                    </select>
                </div>
                
                <div class="mb-3">
                    <label class="form-label text-secondary fw-semibold">Complaint / Symptom Description <span class="text-danger">*</span></label>
                    <textarea id="complaintText" class="form-control bg-dark text-light border-secondary" rows="3" placeholder="Describe symptoms or observations (e.g., Acute onset of vomiting, high fever, and severe abdominal cramps after consuming seafood)."></textarea>
                </div>

                <div class="row g-3 mb-3">
                    <div class="col-md-6">
                        <label class="form-label text-secondary fw-semibold">Restaurant Name</label>
                        <input type="text" id="restaurantNameInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. Ocean Blue Seafood">
                    </div>
                    <div class="col-md-6">
                        <label class="form-label text-secondary fw-semibold">Location / Address</label>
                        <input type="text" id="locationInput" class="form-control bg-dark text-light border-secondary" placeholder="e.g. 420 Lexington Ave, Manhattan">
                    </div>
                </div>

                <button onclick="runTriage()" class="btn btn-primary w-100 py-2" id="btnSubmit">
                    Execute Digital Health Triage
                </button>
            </div>

            <div id="resultsCard" class="card p-4 shadow-sm d-none mb-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0 text-light">Epidemiological Triage Assessment</h5>
                    <span id="triageBadge" class="badge fs-6 px-3 py-2"></span>
                </div>
                
                <div class="row text-center mb-4">
                    <div class="col-md-6 mb-2">
                        <div class="p-3 bg-dark rounded border border-secondary">
                            <small class="text-secondary d-block">Syndromic Hazard Score</small>
                            <h3 id="severityScore" class="fw-bold mb-0 text-info"></h3>
                        </div>
                    </div>
                    <div class="col-md-6 mb-2">
                        <div class="p-3 bg-dark rounded border border-secondary">
                            <small class="text-secondary d-block">Public Health Hazard Flag</small>
                            <h3 id="severityLabel" class="fw-bold mb-0"></h3>
                        </div>
                    </div>
                </div>

                <!-- Clean, formatted assessment without markdown symbols -->
                <div class="assessment-section">
                    <div class="assessment-header">Clinical Symptom & Hazard Screening</div>
                    <div class="item-row">
                        <div class="item-label">Target Establishment:</div>
                        <div class="item-value" id="displayEstablishment">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Category Selected:</div>
                        <div class="item-value" id="displayCategory">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Complaint Analysis:</div>
                        <div class="item-value" id="displayAnalysis">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Syndromic Risk Score:</div>
                        <div class="item-value" id="displayRiskScore">-</div>
                    </div>
                    <div class="item-row">
                        <div class="item-label">Surveillance Category:</div>
                        <div class="item-value" id="displaySurveillance">-</div>
                    </div>

                    <div class="assessment-header mt-4">Epidemiological Decision Support</div>
                    <div class="item-row">
                        <div class="item-label">Recommended Triage Tier:</div>
                        <div class="item-value fw-bold" id="displayTier">-</div>
                    </div>
                    <div class="highlight-box" id="actionBox">
                        <span class="fw-bold d-block mb-1 text-light">Action Directives:</span>
                        <span id="displayAction">-</span>
                    </div>

                    <div class="text-muted small mt-3">
                        Note: AI decision support recommendation for public health and environmental clinical officers.
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Autofill complaint text when a category is selected if text is empty
            document.getElementById('categorySelect').addEventListener('change', function() {
                const textElem = document.getElementById('complaintText');
                if (!textElem.value.trim() && this.value) {
                    textElem.value = this.value;
                }
            });

            async function runTriage() {
                const category = document.getElementById('categorySelect').value.trim();
                const text = document.getElementById('complaintText').value.trim();
                const restaurant_name = document.getElementById('restaurantNameInput').value.trim();
                const location = document.getElementById('locationInput').value.trim();
                const btn = document.getElementById('btnSubmit');
                const resultsCard = document.getElementById('resultsCard');

                if (!text && !category) { 
                    alert('Please select an affected category or enter complaint details.'); 
                    return; 
                }

                btn.disabled = true;
                btn.innerText = 'Analyzing Syndromic Indicators...';
                resultsCard.classList.add('d-none');

                try {
                    const resp = await fetch('/investigate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: text || category,
                            category: category || null,
                            restaurant_name: restaurant_name || null,
                            location: location || null
                        })
                    });
                    const data = await resp.json();

                    if (!resp.ok) {
                        alert(data.detail || 'Error processing request.');
                        return;
                    }

                    document.getElementById('severityScore').innerText = data.syndromic_score;
                    document.getElementById('severityLabel').innerText = data.hazard_flag;
                    document.getElementById('severityLabel').className = 'fw-bold mb-0 ' + (data.hazard_flag.includes('CRITICAL') || data.hazard_flag.includes('MODERATE') ? 'text-danger' : 'text-success');

                    const badge = document.getElementById('triageBadge');
                    badge.innerText = data.triage_level;
                    badge.className = 'badge fs-6 px-3 py-2 badge-' + data.triage_level;

                    // Clean assessment fields
                    document.getElementById('displayEstablishment').innerText = data.establishment;
                    document.getElementById('displayCategory').innerText = data.category;
                    document.getElementById('displayAnalysis').innerText = '"' + data.complaint_summary + '"';
                    document.getElementById('displayRiskScore').innerText = data.syndromic_score.toFixed(3) + ' (' + data.hazard_flag + ')';
                    document.getElementById('displaySurveillance').innerText = data.surveillance_framework;
                    document.getElementById('displayTier').innerText = data.triage_level;
                    document.getElementById('displayAction').innerText = data.action_directive;

                    // Set highlight box style
                    const actionBox = document.getElementById('actionBox');
                    actionBox.className = 'highlight-box ' + (data.triage_level.toLowerCase());

                    resultsCard.classList.remove('d-none');
                } catch (e) {
                    alert('Request failed: ' + e);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Execute Digital Health Triage';
                }
            }
        </script>
    </body>
    </html>
    """

# ASGI Handler for Vercel Serverless Function execution
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = app
