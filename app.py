"""
Sepsis Early Warning System — Flask application
==================================================

Loads the trained XGBoost model + its companion artifacts (feature column
order, MNAR lab columns, alert threshold) and serves the dashboard,
risk calculator, result page, and a downloadable PDF report.

Routes
------
GET  /                          dashboard / risk calculator form   (endpoint: "dashboard")
POST /predict                   run a prediction, render result.html
GET  /download_report/<id>      generate & stream a PDF for a past prediction
GET  /healthz                   plain-text health check (handy for Render)
"""

import io
import os
import pickle
import uuid
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from fpdf import FPDF

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Load model + artifacts once at startup (not per-request)
# ---------------------------------------------------------------------------
booster = xgb.Booster()
booster.load_model(os.path.join(MODEL_DIR, "sepsis_xgb_model.json"))

with open(os.path.join(MODEL_DIR, "sepsis_feature_cols.pkl"), "rb") as f:
    FEATURE_ORDER = pickle.load(f)          # exact 98-feature order the model expects

with open(os.path.join(MODEL_DIR, "sepsis_mnar_cols.pkl"), "rb") as f:
    MNAR_COLS = pickle.load(f)              # labs that get an "is_missing" indicator

with open(os.path.join(MODEL_DIR, "sepsis_threshold.pkl"), "rb") as f:
    THRESHOLD = float(pickle.load(f))       # F2-optimal alert cutoff (~0.1183)

# Derive the delta/acceleration column lists directly from FEATURE_ORDER
# instead of hand-duplicating them, so this can never drift from the model.
DELTA_COLS = [name[:-7] for name in FEATURE_ORDER if name.endswith("_delta1")]
ACCEL_COLS = [name[:-6] for name in FEATURE_ORDER if name.endswith("_accel")]

MODEL_METRICS = {"auroc": 0.9358, "auprc": 0.4250, "f2": 0.5556}

# Population-typical fallback values used when a lab/vital is left blank.
LAB_DEFAULTS = {
    "BaseExcess": 0, "HCO3": 24, "FiO2": 0.21, "pH": 7.40, "PaCO2": 40, "SaO2": 97,
    "AST": 25, "BUN": 14, "Alkalinephos": 80, "Calcium": 9.0, "Creatinine": 0.9,
    "Glucose": 100, "Lactate": 1.0, "Magnesium": 2.0, "Phosphate": 3.5, "Potassium": 4.0,
    "Bilirubin_total": 0.7, "TroponinI": 0.01, "Hct": 42, "Hgb": 14, "PTT": 30,
    "WBC": 7.5, "Fibrinogen": 250, "Platelets": 250,
}
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
VITAL_DEFAULTS = {"HR": 80, "O2Sat": 98, "Temp": 36.8, "SBP": 120, "MAP": 85, "DBP": 75, "Resp": 16, "EtCO2": 35}

# Friendlier labels for the SHAP chart (falls back to the raw column name).
FEATURE_LABELS = {
    "Unnamed: 0": "Row index (dataset artifact)", "Patient_ID": "Patient ID (dataset artifact)",
    "HR": "Heart rate", "O2Sat": "SpO\u2082", "Temp": "Temperature", "SBP": "Systolic BP",
    "MAP": "MAP", "DBP": "Diastolic BP", "Resp": "Respiratory rate", "Lactate": "Lactate",
    "Creatinine": "Creatinine", "WBC": "WBC", "Platelets": "Platelets",
    "Bilirubin_total": "Total bilirubin", "sofa_score": "SOFA score", "news_score": "NEWS score",
    "hours_in_icu": "Hours in ICU", "total_alarm_flags": "Active alarm flags",
    "HR_delta1": "Heart rate trend (1h)", "SBP_delta1": "Systolic BP trend (1h)",
    "Lactate_delta1": "Lactate trend (1h)", "HR_accel": "Heart rate acceleration (2h)",
    "Lactate_accel": "Lactate acceleration (2h)", "sofa_delta_3h": "SOFA change (3h)",
    "sofa_delta_6h": "SOFA change (6h)", "news_delta_3h": "NEWS change (3h)",
}

# In-memory store of recent predictions, keyed by report_id, used for PDF export.
# NOTE: this resets on every restart/deploy and isn't shared across multiple
# server processes/workers. Fine for a demo; swap in a real DB or cache
# (Redis, SQLite, Postgres) before relying on this in a multi-worker deployment.
REPORTS: dict = {}


# ---------------------------------------------------------------------------
# Form parsing helpers
# ---------------------------------------------------------------------------
def form_float(form, key, default=None):
    """Parse a form field as float; returns `default` if blank/missing/invalid."""
    raw = form.get(key, "")
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def compute_features(form):
    """
    Rebuild the exact 98-feature vector the model was trained on, from raw
    form input. Mirrors the notebook's feature-engineering pipeline:
    raw vitals/labs/demographics -> missing flags -> 1h deltas -> 2h
    accelerations -> alarm flags -> partial SOFA -> NEWS -> 3h/6h score deltas.
    """
    f = {}
    missing_flags = {}

    f["Unnamed: 0"] = form_float(form, "rowIndex", 0)
    f["Patient_ID"] = form_float(form, "patientId", 0)
    f["Age"] = form_float(form, "Age", 0)
    f["Gender"] = int(form.get("Gender", "1"))

    icu_unit = form.get("IcuUnit", "none")
    f["Unit1"] = 1 if icu_unit == "micu" else 0
    f["Unit2"] = 1 if icu_unit == "sicu" else 0

    f["HospAdmTime"] = form_float(form, "HospAdmTime", 0)
    f["ICULOS"] = form_float(form, "ICULOS", 0)
    f["hours_in_icu"] = f["ICULOS"]

    for v in VITALS:
        f[v] = form_float(form, v, VITAL_DEFAULTS[v])

    for lab in MNAR_COLS:
        raw = form_float(form, lab, None)
        is_missing = raw is None
        missing_flags[lab] = is_missing
        f[lab + "_is_missing"] = 1 if is_missing else 0
        f[lab] = LAB_DEFAULTS[lab] if is_missing else raw

    for col in DELTA_COLS:
        prev = form_float(form, "p1_" + col, None)
        f[col + "_delta1"] = 0 if prev is None else f[col] - prev

    for col in ACCEL_COLS:
        prev2 = form_float(form, "p2_" + col, None)
        f[col + "_accel"] = 0 if prev2 is None else f[col] - prev2

    f["hypoxia_flag"] = 1 if f["O2Sat"] < 94 else 0
    f["tachycardia_flag"] = 1 if f["HR"] > 100 else 0
    f["hypotension_flag"] = 1 if f["SBP"] < 90 else 0
    f["fever_flag"] = 1 if (f["Temp"] > 38.3 or f["Temp"] < 36.0) else 0
    f["tachypnea_flag"] = 1 if f["Resp"] > 22 else 0
    f["lactate_hi_flag"] = 1 if f["Lactate"] > 2.0 else 0
    f["total_alarm_flags"] = sum(
        f[k] for k in ["hypoxia_flag", "tachycardia_flag", "hypotension_flag",
                        "fever_flag", "tachypnea_flag", "lactate_hi_flag"]
    )

    sofa_resp = 2 if f["O2Sat"] < 90 else (1 if f["O2Sat"] < 94 else 0)
    sofa_coag = 3 if f["Platelets"] < 50 else (2 if f["Platelets"] < 100 else (1 if f["Platelets"] < 150 else 0))
    sofa_renal = 3 if f["Creatinine"] > 3.5 else (2 if f["Creatinine"] > 2.0 else (1 if f["Creatinine"] > 1.2 else 0))
    sofa_liver = 2 if f["Bilirubin_total"] > 2.0 else (1 if f["Bilirubin_total"] > 1.2 else 0)
    f["sofa_resp"], f["sofa_coag"], f["sofa_renal"], f["sofa_liver"] = sofa_resp, sofa_coag, sofa_renal, sofa_liver
    f["sofa_score"] = sofa_resp + sofa_coag + sofa_renal + sofa_liver

    resp = f["Resp"]
    news_resp = 3 if (resp >= 25 or resp <= 8) else (2 if 21 <= resp <= 24 else (1 if 9 <= resp <= 11 else 0))
    o2 = f["O2Sat"]
    news_o2 = 3 if o2 <= 91 else (2 if 92 <= o2 <= 93 else (1 if 94 <= o2 <= 95 else 0))
    sbp = f["SBP"]
    news_sbp = 3 if sbp <= 90 else (2 if 91 <= sbp <= 100 else (1 if 101 <= sbp <= 110 else 0))
    hr = f["HR"]
    news_hr = 3 if (hr > 130 or hr <= 40) else (2 if 111 <= hr <= 130 else (1 if 91 <= hr <= 110 else 0))
    f["news_resp"], f["news_o2"], f["news_sbp"], f["news_hr"] = news_resp, news_o2, news_sbp, news_hr
    f["news_score"] = news_resp + news_o2 + news_sbp + news_hr

    sofa3 = form_float(form, "h_sofa3", None)
    sofa6 = form_float(form, "h_sofa6", None)
    news3 = form_float(form, "h_news3", None)
    f["sofa_delta_3h"] = 0 if sofa3 is None else f["sofa_score"] - sofa3
    f["sofa_delta_6h"] = 0 if sofa6 is None else f["sofa_score"] - sofa6
    f["news_delta_3h"] = 0 if news3 is None else f["news_score"] - news3

    return f, missing_flags


def to_vector(features):
    return [float(features.get(name, 0) or 0) for name in FEATURE_ORDER]


def predict_proba(vector):
    dmatrix = xgb.DMatrix(np.array([vector], dtype=float))
    return float(booster.predict(dmatrix)[0])


def shap_contributions(vector, top_n=12):
    """XGBoost's native pred_contribs = exact SHAP values for tree models —
    no extra `shap` dependency needed."""
    dmatrix = xgb.DMatrix(np.array([vector], dtype=float))
    contribs = booster.predict(dmatrix, pred_contribs=True)[0][:-1]  # drop bias term
    pairs = sorted(zip(FEATURE_ORDER, contribs), key=lambda p: abs(p[1]), reverse=True)[:top_n]
    return [{"feature": FEATURE_LABELS.get(name, name), "value": round(float(val), 4)} for name, val in pairs]


def categorize_risk(prob, threshold):
    if prob >= threshold * 2:
        return "Critical"
    if prob >= threshold:
        return "High"
    if prob >= threshold * 0.5:
        return "Moderate"
    return "Low"


def recommendation_lines(category):
    return {
        "Critical": [
            "Activate sepsis bundle: blood cultures, broad-spectrum antibiotics, "
            "and 30 mL/kg crystalloid for hypotension/lactate \u2265 4 mmol/L.",
            "Obtain repeat lactate within 2\u20134 hours.",
            "Consider ICU/rapid response evaluation if not already escalated.",
        ],
        "High": [
            "Repeat vital signs and labs within 1 hour.",
            "Notify attending/charge nurse of elevated risk score.",
            "Review fluid status and antibiotic timing if infection is suspected.",
        ],
        "Moderate": [
            "Increase vital sign monitoring frequency.",
            "Re-screen for infection source if not already identified.",
            "Reassess risk score in 2\u20134 hours or with any clinical change.",
        ],
        "Low": [
            "Continue routine vital sign monitoring.",
            "No immediate sepsis-specific action indicated by this score alone.",
        ],
    }[category]


def pdf_safe(text):
    """fpdf2's built-in core fonts (Helvetica) only support latin-1. Swap the
    handful of typographic characters used elsewhere in the app for plain
    ASCII so PDF generation never throws a UnicodeEncodeError."""
    replacements = {
        "\u2212": "-", "\u2013": "-", "\u2014": "-",
        "\u2265": ">=", "\u2264": "<=",
        "\u2082": "2", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", endpoint="dashboard")
def dashboard():
    return render_template("index.html", metrics=MODEL_METRICS, active_page="dashboard")


@app.route("/predict", methods=["POST"], endpoint="predict")
def predict():
    features, missing_flags = compute_features(request.form)
    vector = to_vector(features)

    prob = predict_proba(vector)
    category = categorize_risk(prob, THRESHOLD)
    shap_features = shap_contributions(vector)

    report_id = uuid.uuid4().hex[:10]
    REPORTS[report_id] = {
        "risk_score": prob,
        "threshold": THRESHOLD,
        "risk_category": category,
        "sofa_score": features["sofa_score"],
        "news_score": features["news_score"],
        "sofa_components": {
            "Respiratory": features["sofa_resp"], "Coagulation": features["sofa_coag"],
            "Renal": features["sofa_renal"], "Liver": features["sofa_liver"],
        },
        "flags": {
            "Hypoxia": bool(features["hypoxia_flag"]), "Tachycardia": bool(features["tachycardia_flag"]),
            "Hypotension": bool(features["hypotension_flag"]), "Fever/Hypothermia": bool(features["fever_flag"]),
            "Tachypnea": bool(features["tachypnea_flag"]), "High lactate": bool(features["lactate_hi_flag"]),
        },
        "shap_features": shap_features,
        "assessed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    return render_template(
        "result.html",
        report_id=report_id,
        risk_score=prob,
        threshold=THRESHOLD,
        risk_category=category,
        sofa_score=features["sofa_score"],
        news_score=features["news_score"],
        shap_features=shap_features,
        assessed_at=REPORTS[report_id]["assessed_at"],
        recommendations=recommendation_lines(category),
        active_page="calculator",
    )


@app.route("/download_report/<report_id>", endpoint="download_report")
def download_report(report_id):
    data = REPORTS.get(report_id)
    if data is None:
        flash("That report has expired or could not be found. Please run a new assessment.", "error")
        return redirect(url_for("dashboard"))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Sepsis Risk Assessment Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, pdf_safe(f"Assessed: {data['assessed_at']}"), ln=True)
    pdf.cell(0, 7, f"Report ID: {report_id}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Predicted Risk: {data['risk_score']*100:.1f}%  ({data['risk_category']})", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Alert threshold: {data['threshold']*100:.1f}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"SOFA (partial): {data['sofa_score']} / 11", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for organ, val in data["sofa_components"].items():
        pdf.cell(0, 6, pdf_safe(f"  {organ}: {val}"), ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"NEWS: {data['news_score']} / 12", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Active alarm flags", ln=True)
    pdf.set_font("Helvetica", "", 10)
    active = [name for name, on in data["flags"].items() if on]
    pdf.multi_cell(0, 6, pdf_safe(", ".join(active) if active else "None"))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top feature contributions (SHAP)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for item in data["shap_features"][:8]:
        sign = "+" if item["value"] >= 0 else "-"
        pdf.cell(0, 6, pdf_safe(f"  {item['feature']}: {sign}{abs(item['value']):.4f}"), ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0, 5,
        pdf_safe(
            "Decision-support prototype for clinical education / portfolio use only. "
            "Not a certified medical device \u2014 must never replace clinical judgment "
            "or your institution's sepsis protocol."
        )
    )

    buffer = io.BytesIO(pdf.output(dest="S"))
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"sepsis_report_{report_id}.pdf",
    )


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
