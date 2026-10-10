

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Customer Churn Early Warning", page_icon="📊", layout="wide")
DB_PATH = "interventions.db"

st.title("📊 Customer Churn Early Warning System")
st.caption("AI-assisted churn risk monitoring • Explainable signals • Intervention tracking")

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
    with sqlite3.connect(DB_PATH) as conn:
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

def add_intervention(customer_id, action, owner, status, notes):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO interventions(customer_id, action, owner, status, notes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (customer_id, action, owner, status, notes, datetime.now().isoformat(timespec="seconds"))
        )

def read_interventions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM interventions ORDER BY created_at DESC", conn)

init_db()

# --- CORRECTED SIDEBAR STRUCTURE ---
with st.sidebar:
    st.markdown("### AI-POWERED CUSTOMER INTELLIGENCE")
    st.divider()
    
    st.header("1. Customer data")
    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data (not real customers)", value=uploaded is None)
    
    st.divider()
    st.header("2. Risk thresholds")
    high_threshold = st.slider("High risk starts at (%)", 50, 90, 75)
    medium_threshold = st.slider("Medium risk starts at (%)", 20, 70, 40)
    st.caption("Thresholds affect displayed risk labels, not the trained model.")
    
    st.divider()
    st.header("3. Behaviour alert")
    usage_decline_threshold = st.slider("Flag usage decline greater than (%)", 10, 90, 40)

# --- DATA HANDLING ---
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

st.info(f"Data source: **{data_source}** · {len(df):,} rows · {df.shape[1]} columns")
st.subheader("Preview")
st.dataframe(df.head(8), use_container_width=True)

target_candidates = [c for c in df.columns if c.lower() in {"churn", "churned", "exited", "target", "label"}]
default_target = target_candidates[0] if target_candidates else df.columns[-1]
target = st.selectbox("Select the column that tells whether the customer churned", df.columns, index=df.columns.get_loc(default_target))
id_candidates = [c for c in df.columns if c.lower() in {"customer_id", "customerid", "customer", "id"}]
id_col = st.selectbox("Customer ID column (optional)", ["(row number)"] + list(df.columns),
                    index=(1 + df.columns.get_loc(id_candidates[0])) if id_candidates else 0)

if df[target].nunique(dropna=True) < 2:
    st.error("The target column must contain at least two classes (for example, Yes and No).")
    st.stop()

target_values = df[target].dropna().astype(str).str.strip()
lower_to_original = {v.lower(): v for v in target_values.unique()}
positive = next((lower_to_original[k] for k in ["yes", "1", "true", "churn", "churned"] if k in lower_to_original), sorted(target_values.unique())[-1])
y = (df[target].astype(str).str.strip().str.lower() == positive.lower()).astype(int)
feature_cols = [c for c in df.columns if c not in {target, id_col if id_col != "(row number)" else "__no_id__"}]
X = df[feature_cols].copy()
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

high_count = int((results["risk_level"] == "High").sum())
medium_count = int((results["risk_level"] == "Medium").sum())
low_count = int((results["risk_level"] == "Low").sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers monitored", f"{len(results):,}")
k2.metric("High risk", high_count)
k3.metric("Medium risk", medium_count)
k4.metric("Low risk", low_count)

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Customer risk distribution")
    dist = results["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0).rename_axis("Risk level").reset_index(name="Customers")
    st.plotly_chart(px.bar(dist, x="Risk level", y="Customers", color="Risk level"), use_container_width=True)
with right:
    st.subheader("Customers needing attention")
    st.dataframe(results.sort_values("risk_score", ascending=False).head(10), use_container_width=True, hide_index=True)

st.subheader("Global feature importance")
try:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed_data = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)
    st.caption("These features were most influential across the model's predictions.")
    st.bar_chart(importance_df.set_index("Feature")["Importance"])
except Exception as e:
    st.warning(f"Feature importance could not be displayed: {e}")

st.subheader("Customer risk explorer")
customer_options = results.index.tolist()
selected_index = st.selectbox("Choose a customer row", customer_options,
    format_func=lambda i: str(results.loc[i, id_col]) if id_col != "(row number)" else f"Row {i + 1}")
selected = results.loc[selected_index]
score = int(selected["risk_score"])
st.markdown(f"### Risk score: {score}/100 — **{selected['risk_level']} risk**")

st.divider()
st.subheader("🔎 SHAP Explanation for Selected Customer")
try:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    selected_customer = X.loc[[selected_index]]
    transformed_customer = preprocessor.transform(selected_customer)
    if hasattr(transformed_customer, "toarray"):
        transformed_customer = transformed_customer.toarray()
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_customer)
    
    if isinstance(shap_values, list):
        customer_shap = np.asarray(shap_values[1])[0]
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            customer_shap = shap_array[0, :, 1]
        elif shap_array.ndim == 2:
            customer_shap = shap_array[0]
        else:
            raise ValueError("Unexpected SHAP output format.")

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP value": customer_shap
    })
    shap_df["Absolute impact"] = shap_df["SHAP value"].abs()
    shap_df = shap_df.sort_values("Absolute impact", ascending=False).head(10)
    shap_df["Direction"] = np.where(shap_df["SHAP value"] > 0, "Pushes toward churn", "Pushes away from churn")

    st.bar_chart(shap_df.set_index("Feature")["SHAP value"])
    st.dataframe(shap_df[["Feature", "SHAP value", "Direction"]], use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"SHAP explanation could not be displayed: {e}")

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
    st.dataframe(interventions, use_container_width=True, hide_index=True)
else:
    st.info("No interventions recorded yet.")
    