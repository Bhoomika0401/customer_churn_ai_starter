import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from alerting import (
    AlertDeliveryError,
    configured_channels,
    dataset_fingerprint,
    env_flag,
    init_alert_tables,
    read_delivery_history,
    record_delivery,
    reserve_new_high_risk_events,
    send_alert_batch,
)
from retention_playbooks import build_retention_playbook

st.set_page_config(
    page_title="ChurnAI | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --ink: #172B4D;
        --muted: #63758B;
        --canvas: #F4F7FB;
        --surface: #FFFFFF;
        --line: #E2E9F2;
        --navy: #1B385C;
        --navy-surface: #24476F;
        --navy-hover: #2D5683;
        --navy-selected: #DCEBFF;
        --blue: #315AE7;
        --blue-hover: #2448C7;
        --teal: #10B981;
        --amber: #F59E0B;
        --red: #EF4444;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.92);
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #365A80;
    }

    [data-testid="stSidebar"] > div {
        background: var(--navy);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #F1F6FC;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #D3E1F1;
    }

    [data-testid="stSidebar"] hr {
        border-color: #49698B;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #F1F6FC;
        font-weight: 600;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: var(--navy-surface);
        border-color: #6584A5;
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] svg,
    [data-testid="stSidebar"] [data-baseweb="input"] svg {
        fill: #E5EFFA;
    }

    [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:first-child {
        border-color: #C6D7EA;
    }

    [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
        background: #FFFFFF;
        border-color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stTickBar"] {
        background: #A8C4E0;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: #FFFFFF;
        color: #172B4D;
    }

    [role="option"] {
        color: #172B4D;
    }

    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background: #E7F0FF;
        color: #173D76;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 750;
    }

    h2, h3 {
        font-weight: 680;
    }

    p, label {
        line-height: 1.55;
    }

    [data-testid="stMetric"] {
        background: transparent;
        border: 0;
        border-radius: 0;
        padding: 0.2rem 0 0.1rem;
        box-shadow: none;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.86rem;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 2rem;
        font-weight: 760;
    }

    .st-key-kpi-card-blue,
    .st-key-kpi-card-red,
    .st-key-kpi-card-amber,
    .st-key-kpi-card-teal,
    .st-key-risk-distribution-card,
    .st-key-customers-attention-card {
        background: linear-gradient(145deg, #FFFFFF 0%, #FCFDFF 100%);
        border: 1px solid #E2E9F2;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(19, 38, 64, 0.055);
        transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }

    .st-key-kpi-card-blue:hover,
    .st-key-kpi-card-red:hover,
    .st-key-kpi-card-amber:hover,
    .st-key-kpi-card-teal:hover,
    .st-key-risk-distribution-card:hover,
    .st-key-customers-attention-card:hover {
        border-color: #D3DEED;
        box-shadow: 0 12px 30px rgba(19, 38, 64, 0.08);
    }

    .kpi-card-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        min-height: 42px;
        margin-bottom: 0.25rem;
    }

    .kpi-icon {
        display: grid;
        width: 42px;
        height: 42px;
        place-items: center;
        border-radius: 14px;
    }

    .kpi-icon svg {
        width: 21px;
        height: 21px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.8;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    .kpi-icon-blue {
        color: #315AE7;
        background: #EEF2FF;
    }

    .kpi-icon-red {
        color: #DC3F4F;
        background: #FFF0F1;
    }

    .kpi-icon-amber {
        color: #C77709;
        background: #FFF7E8;
    }

    .kpi-icon-teal {
        color: #07866D;
        background: #E9FAF4;
    }

    .kpi-badge {
        display: inline-flex;
        align-items: center;
        min-height: 25px;
        padding: 0.25rem 0.6rem;
        border: 1px solid #E4EAF2;
        border-radius: 999px;
        background: #F6F8FB;
        color: #52657C;
        font-size: 0.7rem;
        font-weight: 650;
        line-height: 1.2;
        text-align: right;
        white-space: nowrap;
    }

    .kpi-badge-red {
        border-color: #FBD5D9;
        background: #FFF1F2;
        color: #B42332;
    }

    .kpi-badge-teal {
        border-color: #C7F0E2;
        background: #ECFBF5;
        color: #08785F;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 13px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] [role="row"]:hover {
        background: #F3F7FE !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFileUploader"] button {
        background: var(--blue);
        color: #FFFFFF;
        border: 1px solid var(--blue);
        border-radius: 9px;
        font-weight: 650;
        transition: background 150ms ease, border-color 150ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFileUploader"] button:hover {
        background: var(--blue-hover);
        border-color: var(--blue-hover);
        color: #FFFFFF;
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {
        outline: 3px solid #9BB5FF;
        outline-offset: 2px;
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background: #FFFFFF;
        border-color: #D3DDE9;
        border-radius: 9px;
    }

    [data-testid="stFileUploader"] section {
        background: #F7F9FC;
        border: 1px dashed #A8B8CC;
        border-radius: 9px;
    }

    [data-testid="stAlert"] {
        border-radius: 11px;
    }

    hr {
        border-color: var(--line);
        margin: 1.4rem 0;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        [data-testid="stMetric"] {
            padding: 0.8rem;
        }
    }

    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 calc(50% - 0.5rem) !important;
            min-width: min(100%, 220px);
        }
    }
</style>
""", unsafe_allow_html=True)

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("APP_DATA_DIR", str(APP_ROOT))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DATA_DIR / "interventions.db")

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 24px 0;">
        <h1 style="color:#FFFFFF; margin-bottom:0; font-size:1.65rem;">ChurnAI</h1>
        <p style="color:#B8C7DD; font-size:13px; margin-top:4px;">
            Customer intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("PREDICT  ·  PREVENT  ·  RETAIN")

@st.cache_data
def make_demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly_charges = np.round(rng.normal(70, 20, n).clip(18, 130), 2)
    usage_previous = rng.integers(5, 31, n)
    usage_recent = np.maximum(0, np.round(usage_previous * rng.normal(0.92, 0.35, n))).astype(int)
    complaints = rng.poisson(0.5, n)
    missed_payments = rng.binomial(3, 0.15, n)
    engagement = rng.integers(1, 11, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    logit = (
        -1.5 + 0.035 * (monthly_charges - 70) - 0.035 * tenure
        + 0.38 * complaints + 0.52 * missed_payments
        - 0.13 * engagement + 0.045 * (usage_previous - usage_recent)
        + np.where(contract == "Month-to-month", 0.9, 0)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "usage_previous": usage_previous,
        "usage_recent": usage_recent,
        "complaints_last_90_days": complaints,
        "missed_payments_last_90_days": missed_payments,
        "engagement_score": engagement,
        "contract_type": contract,
        "Churn": churn,
    })

def init_db():
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as conn, conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            action TEXT NOT NULL,
            owner TEXT,
            status TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL
        )
        """)
    init_alert_tables(DB_PATH)

def add_intervention(customer_id, action, owner, status, notes):
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as conn, conn:
        conn.execute(
            "INSERT INTO interventions(customer_id, action, owner, status, notes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (customer_id, action, owner, status, notes, datetime.now().isoformat(timespec="seconds"))
        )

def read_interventions():
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as conn:
        return pd.read_sql_query("SELECT * FROM interventions ORDER BY created_at DESC", conn)

init_db()

with st.sidebar:
    st.subheader("Customer data")
    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data (not real customers)", value=uploaded is None)
    st.divider()
    st.subheader("Risk thresholds")
    high_threshold = st.slider("High risk starts at (%)", 50, 90, 75)
    medium_threshold = st.slider("Medium risk starts at (%)", 20, 70, 40)
    st.caption("Thresholds affect displayed risk labels, not the trained model.")
    st.divider()
    st.subheader("Behaviour alert")
    usage_decline_threshold = st.slider("Flag usage decline greater than (%)", 10, 90, 40)
    st.divider()
    st.subheader("External notifications")

    alert_channel_config = {}
    alert_config_error = None
    try:
        alert_channel_config = configured_channels()
    except ValueError as exc:
        alert_config_error = str(exc)

    automatic_alerts_enabled = st.checkbox(
        "Automatically notify on new high-risk customers",
        value=env_flag("AUTO_ALERTS_ENABLED")
    )
    if alert_channel_config:
        st.caption(
            "Configured channels: "
            + ", ".join(alert_channel_config)
            + ". Secrets are read from environment variables."
        )
    else:
        st.caption("Add Slack or SMTP settings to enable delivery.")
    if alert_config_error:
        st.error(alert_config_error)

st.title("Customer churn early warning")
st.caption("Monitor customer health, identify churn risk, and focus retention efforts where they matter most.")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        data_source = "Uploaded CSV"
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif use_demo:
    df = make_demo_data()
    data_source = "Demo data (illustration only)"
else:
    st.info("Upload a CSV or select **Use demo data** in the sidebar to begin.")
    st.stop()

if df.empty:
    st.error("The dataset has no rows.")
    st.stop()

target_candidates = [c for c in df.columns if c.lower() in {"churn", "churned", "exited", "target", "label"}]
default_target = target_candidates[0] if target_candidates else df.columns[-1]
id_candidates = [c for c in df.columns if c.lower() in {"customer_id", "customerid", "customer", "id"}]
with st.sidebar:
    st.caption(f"{data_source} · {len(df):,} rows · {df.shape[1]} columns")
    st.subheader("Model setup")
    target = st.selectbox(
        "Churn outcome column",
        df.columns,
        index=df.columns.get_loc(default_target)
    )
    id_col = st.selectbox(
        "Customer ID column (optional)",
        ["(row number)"] + list(df.columns),
        index=(1 + df.columns.get_loc(id_candidates[0])) if id_candidates else 0
    )

if df[target].nunique(dropna=True) < 2:
    st.error("The target column must contain at least two classes (for example, Yes and No).")
    st.stop()

# Treat the last sorted target class as the positive class, except common churn labels.
target_values = df[target].dropna().astype(str).str.strip()
lower_to_original = {v.lower(): v for v in target_values.unique()}
positive = next((lower_to_original[k] for k in ["yes", "1", "true", "churn", "churned"] if k in lower_to_original), sorted(target_values.unique())[-1])
y = (df[target].astype(str).str.strip().str.lower() == positive.lower()).astype(int)
feature_cols = [c for c in df.columns if c not in {target, id_col if id_col != "(row number)" else "__no_id__"}]
X = df[feature_cols].copy()
# Remove completely empty columns and identifier-like columns to avoid leakage.
X = X.dropna(axis=1, how="all")
id_like = [c for c in X.columns if c.lower() in {"customer_id", "customerid", "id", "customer"}]
X = X.drop(columns=id_like, errors="ignore")
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]
if X.shape[1] == 0:
    st.error("No usable feature columns remain after excluding the target and IDs.")
    st.stop()

if min(np.bincount(y)) < 2:
    st.error("Each target class needs at least two records for a train/test split.")
    st.stop()

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
preprocess = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])
test_size = 0.25 if len(df) >= 20 else 0.4
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
except Exception as exc:
    st.error(f"Model training failed. Check that the target and feature columns are valid. Details: {exc}")
    st.stop()

probabilities = model.predict_proba(X)[:, list(model.named_steps["classifier"].classes_).index(1)]
pred_test = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, list(model.named_steps["classifier"].classes_).index(1)]
metrics = {
    "Accuracy": accuracy_score(y_test, pred_test),
    "Precision": precision_score(y_test, pred_test, zero_division=0),
    "Recall": recall_score(y_test, pred_test, zero_division=0),
}
try:
    metrics["ROC-AUC"] = roc_auc_score(y_test, test_prob)
except Exception:
    metrics["ROC-AUC"] = np.nan

results = df.copy()
results["churn_probability"] = probabilities
results["risk_score"] = (results["churn_probability"] * 100).round().astype(int)
def risk_label(score):
    if score >= high_threshold:
        return "High"
    if score >= medium_threshold:
        return "Medium"
    return "Low"
results["risk_level"] = results["risk_score"].apply(risk_label)

high_count = int((results["risk_level"] == "High").sum())
medium_count = int((results["risk_level"] == "Medium").sum())
low_count = int((results["risk_level"] == "Low").sum())
total_customers = len(results)
average_churn_probability = results["churn_probability"].mean() * 100
predicted_retention_rate = (results["churn_probability"] < 0.5).mean() * 100

st.markdown("## Portfolio overview")
st.caption("A live snapshot of customer exposure, risk, and predicted retention.")

kpi_cards = [
    (
        "Customers",
        f"{total_customers:,}",
        "Customers included in the current prediction results.",
        "Current cohort",
        "blue",
        '<circle cx="9" cy="8" r="3"/><path d="M3 20v-1a6 6 0 0 1 12 0v1"/><path d="M16 4.5a3 3 0 0 1 0 7"/><path d="M18 14a5 5 0 0 1 3 4.5V20"/>'
    ),
    (
        "High risk",
        f"{high_count:,}",
        "Customers above the configured high-risk threshold.",
        f"Cutoff ≥{high_threshold}%",
        "red",
        '<path d="M12 3 21 19H3L12 3Z"/><path d="M12 9v4"/><path d="M12 16h.01"/>'
    ),
    (
        "Avg. churn risk",
        f"{average_churn_probability:.1f}%",
        "Average predicted churn probability across the current dataset.",
        "Portfolio avg.",
        "amber",
        '<path d="M4 19V5"/><path d="M4 19h17"/><path d="m7 14 4-4 3 2 6-7"/><path d="M16 5h4v4"/>'
    ),
    (
        "Retention rate",
        f"{predicted_retention_rate:.1f}%",
        "Share of customers below 50% predicted churn probability.",
        "Below 50% churn",
        "teal",
        '<path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/>'
    ),
]

for col, (label, value, help_text, badge, tone, icon_svg) in zip(
    st.columns(4), kpi_cards
):
    with col:
        with st.container(border=True, key=f"kpi-card-{tone}"):
            st.markdown(
                f"""
                <div class="kpi-card-topline">
                    <span class="kpi-icon kpi-icon-{tone}" aria-hidden="true">
                        <svg viewBox="0 0 24 24">{icon_svg}</svg>
                    </span>
                    <span class="kpi-badge {'kpi-badge-red' if tone == 'red' else 'kpi-badge-teal' if tone == 'teal' else ''}">
                        {badge}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.metric(label, value, help=help_text, border=False)

overview_left, overview_right = st.columns([0.9, 1.1])
with overview_left:
    with st.container(border=True, key="risk-distribution-card"):
        st.subheader("Risk distribution")
        st.caption("Share of customers by predicted churn risk tier.")
        risk_counts = pd.DataFrame({
            "Risk level": ["Low", "Medium", "High"],
            "Customers": [low_count, medium_count, high_count]
        })
        risk_colors = {
            "Low": "#10B981",
            "Medium": "#F59E0B",
            "High": "#EF4444"
        }
        fig_risk = px.pie(
            risk_counts,
            names="Risk level",
            values="Customers",
            hole=0.68,
            color="Risk level",
            color_discrete_map=risk_colors,
            category_orders={"Risk level": ["Low", "Medium", "High"]}
        )
        fig_risk.update_traces(
            textposition="inside",
            texttemplate="%{percent:.0%}",
            insidetextorientation="horizontal",
            marker=dict(line=dict(color="#FFFFFF", width=4)),
            sort=False,
            hovertemplate="<b>%{label} risk</b><br>%{value:,} customers · %{percent:.1%}<extra></extra>"
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#63758B", size=12),
            legend=dict(
                title_text="",
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.04,
                yanchor="top",
                font=dict(size=12)
            ),
            margin=dict(l=8, r=8, t=8, b=34),
            height=330,
            annotations=[
                dict(
                    text=f"<b style='font-size:30px;color:#172B4D'>{total_customers:,}</b>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    align="center"
                )
            ]
        )
        st.plotly_chart(fig_risk, width="stretch")

with overview_right:
    with st.container(border=True, key="customers-attention-card"):
        st.subheader("Customers needing attention")
        st.caption("Highest predicted churn scores in the current portfolio.")
        attention_columns = [
            column for column in [
                id_col if id_col != "(row number)" else None,
                "risk_score",
                "risk_level"
            ]
            if column in results.columns
        ]
        attention_df = (
            results.sort_values("risk_score", ascending=False)
            .loc[:, attention_columns]
            .head(10)
            .rename(columns={
                "risk_score": "Risk score",
                "risk_level": "Risk level"
            })
        )
        attention_df["Risk level"] = attention_df["Risk level"].map(
            lambda level: [level]
        )
        styled_attention_df = (
            attention_df.style
            .set_properties(**{"padding": "10px 12px"})
            .set_table_styles([
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#F6F8FC"),
                        ("color", "#53657B"),
                        ("font-weight", "700"),
                        ("border-bottom", "1px solid #E5EAF2"),
                        ("padding", "11px 12px")
                    ]
                },
                {
                    "selector": "tbody tr:hover",
                    "props": [("background-color", "#F3F7FE")]
                }
            ])
        )
        customer_id_column = id_col if id_col != "(row number)" else None
        attention_column_config = {
            "Risk score": st.column_config.ProgressColumn(
                "Risk score",
                help="Predicted churn risk from 0 to 100.",
                min_value=0,
                max_value=100,
                format="%d",
                color="blue"
            ),
        }
        attention_column_config["Risk level"] = (
            st.column_config.MultiselectColumn(
                "Risk level",
                options=["Low", "Medium", "High"],
                color=["green", "orange", "red"]
            )
        )
        if customer_id_column:
            attention_column_config[customer_id_column] = (
                st.column_config.TextColumn("Customer")
            )
        st.dataframe(
            styled_attention_df,
            width="stretch",
            hide_index=True,
            height=385,
            column_config=attention_column_config
        )

with st.expander("Preview customer data"):
    st.dataframe(df.head(8), width="stretch")

# Add transparent signal descriptions for the selected customer's row.
def reasons_for_row(row):
    reasons = []
    for col in row.index:
        val = row[col]
        col_lower = col.lower()
        if "complaint" in col_lower and pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0] > 0:
            reasons.append(f"{col}: {val}")
        elif ("missed_payment" in col_lower or "overdue" in col_lower) and pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0] > 0:
            reasons.append(f"{col}: {val}")
        elif "usage_previous" in col_lower and "usage_recent" in row.index:
            prev = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            recent = pd.to_numeric(pd.Series([row["usage_recent"]]), errors="coerce").iloc[0]
            if pd.notna(prev) and prev > 0 and pd.notna(recent):
                decline = max(0, (prev - recent) / prev * 100)
                if decline >= usage_decline_threshold:
                    reasons.append(f"Usage declined {decline:.0f}% compared with previous period")
        elif "engagement" in col_lower:
            score = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            if pd.notna(score) and score <= 3:
                reasons.append(f"Low engagement ({val})")
    return reasons or ["No configured warning signal detected; risk score reflects the overall model pattern."]

st.divider()
st.subheader("Model evaluation")
st.caption("Metrics are calculated on a held-out test split. Demo-data metrics are only for testing the app workflow.")
metric_cols = st.columns(len(metrics))
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, "N/A" if pd.isna(value) else f"{value:.1%}")


st.subheader("Model comparison")

try:
    comparison_models = {
        "Random Forest": clone(model),
        "Logistic Regression": Pipeline([
            ("preprocess", clone(preprocess)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
    }

    comparison_results = []

    for model_name, candidate in comparison_models.items():
        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_test)
        probabilities_comparison = candidate.predict_proba(X_test)[:, 1]

        comparison_results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities_comparison)
        })

    comparison_df = pd.DataFrame(comparison_results)

    st.dataframe(
        comparison_df.round(3),
        width="stretch",
        hide_index=True
    )

    st.download_button(
        "📥 Download Model Comparison Report",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_report.csv",
        mime="text/csv"
    )

    st.write("### Compare model performance")

    chart_metric = st.selectbox(
        "Choose a metric to compare",
        ["Accuracy", "Precision", "Recall", "ROC-AUC"],
        key="model_comparison_metric"
    )

    st.bar_chart(
        comparison_df.set_index("Model")[chart_metric]
    )

    st.caption(
        "Both models use the same held-out test set. "
        "Compare recall and ROC-AUC alongside accuracy when "
        "evaluating churn detection."
    )

except Exception as e:
    st.warning(f"Model comparison could not be completed: {e}")

high_count = int((results["risk_level"] == "High").sum())
medium_count = int((results["risk_level"] == "Medium").sum())
low_count = int((results["risk_level"] == "Low").sum())

st.subheader("Customer risk explorer")

feature_names = model.named_steps["preprocess"].get_feature_names_out()
feature_importances = model.named_steps["classifier"].feature_importances_
importance_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

customer_options = results.index.tolist()
selected_index = st.selectbox("Choose a customer row", customer_options,
    format_func=lambda i: str(results.loc[i, id_col]) if id_col != "(row number)" else f"Row {i + 1}")
selected = results.loc[selected_index]
score = int(selected["risk_score"])
st.markdown(f"### Risk score: {score}/100 — **{selected['risk_level']} risk**")
if "usage_previous" in results.columns and "usage_recent" in results.columns:
    prev = pd.to_numeric(pd.Series([selected["usage_previous"]]), errors="coerce").iloc[0]
    recent = pd.to_numeric(pd.Series([selected["usage_recent"]]), errors="coerce").iloc[0]
    if pd.notna(prev) and prev > 0 and pd.notna(recent):
        decline = (prev - recent) / prev * 100
        if decline >= usage_decline_threshold:
            st.warning(f"Behaviour alert: usage is down {decline:.1f}% compared with the previous period.")
        else:
            st.success(f"No usage-decline alert: observed decline is {max(0, decline):.1f}%.")
st.write("**Potential contributing signals to review:**")
for reason in reasons_for_row(selected[feature_cols]):
    st.write(f"- {reason}")
    
st.write("### Why is this customer at risk?")

customer_reasons = reasons_for_row(selected[feature_cols])

for reason in customer_reasons:
    st.write(f"- {reason}")

selected_playbook = build_retention_playbook(
    selected[feature_cols],
    str(selected["risk_level"]),
    usage_decline_threshold
)
st.subheader("Recommended retention playbook")
st.caption(
    "Actions are mapped to observable input signals. They are practical "
    "recommendations, not confirmed causes of churn."
)
for step_number, action in enumerate(selected_playbook, start=1):
    with st.container(border=True):
        st.markdown(
            f"**{step_number}. {action['driver']}** · "
            f"{action['priority']} priority · {action['timing']}"
        )
        st.caption(f"Observed signal: {action['evidence']}")
        st.write(
            f"**Next action:** {action['action']}  \n"
            f"**Suggested owner:** {action['owner']}"
        )

st.write("### Important model features")

top_features = importance_df.head(5)

for _, feature in top_features.iterrows():
    st.write(
        f"- **{feature['Feature']}** "
        f"(importance: {feature['Importance']:.3f})"
    )

st.caption(
    "These are overall model feature importances, not "
    "customer-specific causal explanations."
)

st.caption("These rule-based signals are transparent hints, not causal explanations of the model. Add SHAP in a later iteration for model-specific explanations.")


st.divider()
st.subheader("Customer Behaviour Change Monitor")

activity_file = str(APP_ROOT / "customer_activity_history.csv")

# ---------- CUSTOMER ACTIVITY TREND ----------
st.markdown("## Customer Behaviour Trends")

activity_columns = {
    "customerID",
    "period",
    "login_count",
    "feature_usage",
    "complaints",
    "missed_payments"
}
activity_df = pd.read_csv(activity_file) if os.path.exists(activity_file) else pd.DataFrame()
missing_activity_columns = activity_columns.difference(activity_df.columns)
activity_ready = not activity_df.empty and not missing_activity_columns

if activity_ready:
    activity_df["period"] = pd.to_datetime(
        activity_df["period"], format="%Y-%m", errors="coerce"
    )
    activity_df = activity_df.dropna(subset=["period"]).sort_values(
        ["customerID", "period"]
    )
    activity_ready = not activity_df.empty

if activity_ready:
    trend_df = activity_df.copy()
    monthly_activity = (
        trend_df.groupby(trend_df["period"].dt.to_period("M"), as_index=False)
        .agg(
            Total_Logins=("login_count", "sum"),
            Average_Feature_Usage=("feature_usage", "mean")
        )
        .rename(columns={"period": "Reporting month"})
        .sort_values("Reporting month")
    )
    monthly_activity["Reporting month"] = monthly_activity[
        "Reporting month"
    ].astype(str)

    fig_trend = px.line(
        monthly_activity,
        x="Reporting month",
        y=["Total_Logins", "Average_Feature_Usage"],
        markers=True,
        title="Customer Activity Over Time",
        labels={
            "period": "Reporting Period",
            "value": "Activity",
            "variable": "Metric"
        },
        color_discrete_sequence=["#245FE5", "#14B8A6"]
    )

    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial", color="#475569"),
        title_font=dict(size=18, color="#14243A"),
        margin=dict(l=15, r=15, t=55, b=15),
        height=380,
        legend_title_text=""
    )

    fig_trend.update_xaxes(showgrid=False)
    fig_trend.update_yaxes(
        gridcolor="#E8EEF6",
        zeroline=False
    )

    st.plotly_chart(fig_trend, width="stretch")
elif missing_activity_columns and os.path.exists(activity_file):
    st.error(
        "Customer activity history is missing required columns: "
        + ", ".join(sorted(missing_activity_columns))
    )
elif os.path.exists(activity_file):
    st.info("No customer activity records with valid reporting periods were found.")
else:
    st.info("No customer activity history is available.")


if activity_ready:
    activity = activity_df

    customer_ids = activity["customerID"].unique().tolist()

    def format_activity_customer(customer_id):
        history = activity.loc[
            activity["customerID"] == customer_id
        ].sort_values("period")
        if len(history) < 2:
            behavior = "Activity trend unavailable"
        else:
            previous_logins = pd.to_numeric(
                history.iloc[-2]["login_count"], errors="coerce"
            )
            recent_logins = pd.to_numeric(
                history.iloc[-1]["login_count"], errors="coerce"
            )
            if pd.isna(previous_logins) or pd.isna(recent_logins):
                behavior = "Activity trend unavailable"
            elif previous_logins == 0:
                behavior = "Improving activity" if recent_logins > 0 else "Stable activity"
            else:
                change = (recent_logins - previous_logins) / previous_logins * 100
                if change <= -usage_decline_threshold:
                    behavior = "Declining activity"
                elif change >= usage_decline_threshold:
                    behavior = "Improving activity"
                else:
                    behavior = "Stable activity"
        return f"{customer_id} · {behavior}"

    selected_activity_id = st.selectbox(
        "Select customer for behaviour monitoring",
        customer_ids,
        format_func=format_activity_customer,
        key="activity_customer"
    )

    customer_history = activity[
        activity["customerID"] == selected_activity_id
    ].sort_values("period")

    st.write("### Activity history")
    st.line_chart(
        customer_history.set_index("period")[
            ["login_count", "feature_usage"]
        ]
    )

    if len(customer_history) >= 2:
        previous = customer_history.iloc[-2]
        recent = customer_history.iloc[-1]

        previous_logins = previous["login_count"]
        recent_logins = recent["login_count"]

        if previous_logins > 0:
            decline = (
                (previous_logins - recent_logins)
                / previous_logins
            ) * 100
        else:
            decline = 0

        st.metric(
            "Login decline",
            f"{decline:.1f}%",
            delta=f"{recent_logins - previous_logins} logins"
        )

        if decline >= usage_decline_threshold:
            st.warning(
                "Behaviour alert: recent login activity has "
                "declined beyond the configured threshold."
            )
        else:
            st.success(
                "No login decline alert for the selected period."
            )

        st.write("### Latest customer signals")
        st.write(f"- Feature usage: {recent['feature_usage']}")
        st.write(f"- Complaints: {recent['complaints']}")
        st.write(f"- Missed payments: {recent['missed_payments']}")
    else:
        st.info("At least two activity periods are required.")

st.divider()
st.subheader("🚨 Customers Requiring Attention")

# ---------- HIGH-RISK CUSTOMER TABLE ----------
st.markdown("## Customers Requiring Attention")
st.caption("Customers with a high predicted risk of churn.")

high_risk_df = results[
    results["risk_level"].str.lower() == "high"
].copy()

if not high_risk_df.empty:
    # Format the probability as a percentage for display
    if "churn_probability" in high_risk_df.columns:
        high_risk_df["Churn Probability"] = (
            high_risk_df["churn_probability"] * 100
        ).map(lambda value: f"{value:.1f}%")

    # Display only useful business-facing columns
    preferred_columns = [
        "customerID",
        "Churn Probability",
        "risk_level"
    ]

    display_columns = [
        col for col in preferred_columns
        if col in high_risk_df.columns
    ]

    st.dataframe(
        high_risk_df[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "customerID": st.column_config.TextColumn(
                "Customer ID"
            ),
            "Churn Probability": st.column_config.TextColumn(
                "Churn Probability"
            ),
            "risk_level": st.column_config.TextColumn(
                "Risk Level"
            )
        }
    )
else:
    st.success("No high-risk customers were found in the current results.")


# Identify customers classified as high risk
high_risk_customers = results[
    results["risk_level"].astype(str).str.lower() == "high"
].copy()

notification_source = (
    f"{data_source}:{uploaded.name}" if uploaded is not None else data_source
)
source_fingerprint = dataset_fingerprint(
    df,
    namespace=f"{notification_source}|{target}|{id_col}"
)
alert_records = []
for row_index, customer in results.iterrows():
    customer_label = (
        str(customer[id_col])
        if id_col != "(row number)" and pd.notna(customer[id_col])
        else f"Row {row_index + 1}"
    )
    customer_playbook = build_retention_playbook(
        customer[feature_cols],
        str(customer["risk_level"]),
        usage_decline_threshold
    )
    alert_records.append({
        "customer_label": customer_label,
        "source_label": notification_source,
        "source_fingerprint": source_fingerprint,
        "risk_score": int(customer["risk_score"]),
        "risk_level": str(customer["risk_level"]),
        "driver": customer_playbook[0]["driver"],
        "next_action": customer_playbook[0]["action"],
    })

if uploaded is None:
    if automatic_alerts_enabled:
        st.info("External alerts are suppressed while using demo data.")
else:
    delivery_channels = (
        list(alert_channel_config)
        if automatic_alerts_enabled and not alert_config_error
        else []
    )
    try:
        pending_by_channel = reserve_new_high_risk_events(
            DB_PATH,
            alert_records,
            delivery_channels
        )
    except sqlite3.Error:
        st.error("Alert state could not be read or saved. No external notifications were sent.")
    else:
        if automatic_alerts_enabled and alert_config_error:
            st.error("Automatic alerts are paused until the channel configuration is corrected.")
        elif automatic_alerts_enabled and not alert_channel_config:
            st.warning("Automatic alerts are enabled, but no Slack or Email channel is configured.")
        elif automatic_alerts_enabled:
            delivery_failures = []
            for channel, events in pending_by_channel.items():
                if not events:
                    continue
                event_ids = [str(event["id"]) for event in events]
                try:
                    send_alert_batch(channel, alert_channel_config[channel], events)
                except AlertDeliveryError as exc:
                    record_delivery(
                        DB_PATH,
                        event_ids,
                        channel,
                        success=False,
                        detail=type(exc).__name__
                    )
                    delivery_failures.append(channel)
                    st.error(str(exc))
                except sqlite3.Error:
                    st.error(
                        f"{channel} delivery outcome could not be saved. "
                        "Check the notification history before retrying."
                    )
                else:
                    try:
                        record_delivery(DB_PATH, event_ids, channel, success=True)
                    except sqlite3.Error:
                        st.error(
                            f"{channel} acknowledged the alert, but its delivery state "
                            "could not be saved. Check the destination before retrying."
                        )
                    else:
                        st.success(
                            f"{channel}: sent a digest covering {len(events)} new "
                            "high-risk customer alert(s)."
                        )
            failed_channels = ", ".join(delivery_failures)
            if failed_channels:
                st.warning(
                    f"Some alerts were not delivered through {failed_channels}. "
                    "Failed deliveries are retried no more than once every 15 minutes."
                )

if not high_risk_customers.empty:
    st.warning(
        f"{len(high_risk_customers)} customer(s) are currently "
        "classified as high risk."
    )

    display_columns = [id_col, "risk_score", "risk_level"]

    # Show only columns that exist in the results
    display_columns = [
        col for col in display_columns
        if col in high_risk_customers.columns
    ]

    st.dataframe(
        high_risk_customers[display_columns]
        .sort_values("risk_score", ascending=False),
        width="stretch"
    )
else:
    st.success("No high-risk customers were identified.")
    
st.divider()
st.subheader("📋 High-Risk Alert Log")

alert_log_file = DATA_DIR / "customer_alert_history.csv"

if not alert_log_file.exists():
    pd.DataFrame(columns=[
        "timestamp", "customer_id", "risk_score", "risk_level"
    ]).to_csv(alert_log_file, index=False)

if not high_risk_customers.empty:
    if st.button("Record current high-risk alerts"):
        alert_history = pd.read_csv(alert_log_file)

        new_alerts = pd.DataFrame({
            "timestamp": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ] * len(high_risk_customers),
            "customer_id": [
                str(row[id_col])
                if id_col != "(row number)" and pd.notna(row[id_col])
                else f"Row {index + 1}"
                for index, row in high_risk_customers.iterrows()
            ],
            "risk_score": high_risk_customers["risk_score"].values,
            "risk_level": high_risk_customers["risk_level"].values
        })

        alert_history = pd.concat(
            [alert_history, new_alerts],
            ignore_index=True
        )

        alert_history.to_csv(alert_log_file, index=False)
        st.success("Current high-risk alerts recorded.")
        st.rerun()

alert_history = pd.read_csv(alert_log_file)

if not alert_history.empty:
    st.dataframe(alert_history, width="stretch")

    csv_data = alert_history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download alert history",
        data=csv_data,
        file_name="customer_alert_history.csv",
        mime="text/csv"
    )
else:
    st.info("No alerts have been recorded yet.")

with st.expander("External notification delivery history"):
    delivery_history = read_delivery_history(DB_PATH)
    if delivery_history.empty:
        st.info("No external notifications have been sent yet.")
    else:
        st.dataframe(delivery_history, width="stretch", hide_index=True)
        st.caption(
            "Delivery state is stored in the local SQLite database. "
            "Recipient secrets are never included in this history."
        )

with st.expander("View all customer risk scores"):
    
    st.dataframe(results.sort_values("risk_score", ascending=False), width="stretch", hide_index=True)
    st.download_button("Download risk report CSV", results.to_csv(index=False).encode("utf-8"), "customer_risk_report.csv", "text/csv")

st.divider()
st.subheader("Intervention tracking")
with st.form("intervention_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    customer_id = c1.text_input("Customer ID", value=str(selected[id_col]) if id_col != "(row number)" else f"Row {selected_index + 1}")
    action = c2.selectbox("Intervention", ["Call customer", "Resolve complaint", "Offer retention plan", "Payment reminder", "Schedule follow-up", "Other"])
    c3, c4 = st.columns(2)
    owner = c3.text_input("Owner / team member")
    status = c4.selectbox("Status", ["Planned", "In progress", "Completed", "Unsuccessful"])
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Save intervention")
    if submitted:
        if customer_id.strip():
            add_intervention(customer_id.strip(), action, owner.strip(), status, notes.strip())
            st.success("Intervention saved.")
        else:
            st.error("Enter a customer ID.")
interventions = read_interventions()
if not interventions.empty:
    st.dataframe(interventions, width="stretch", hide_index=True)
    st.download_button("Download intervention log CSV", interventions.to_csv(index=False).encode("utf-8"), "intervention_log.csv", "text/csv")
else:
    st.info("No interventions recorded yet. Use the form above to add the first one.")
