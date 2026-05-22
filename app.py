"""
Stroke prediction + market basket analysis (Streamlit).
Run: streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import wittgenstein as lw
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Paths
BASE_DIR = Path(__file__).resolve().parent
STROKE_CSV = BASE_DIR / "healthcare-dataset-stroke-data.csv"
GROCERIES_CSV = BASE_DIR / "groceries.csv"

GROCERIES_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
)


# Page config
st.set_page_config(
    page_title="Stroke prediction & baskets",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(160deg, #f6f8fc 0%, #eef3fb 45%, #f8fafc 100%);
    }
    /* Main content: dark default text */
    section.main {
        color: #0f172a;
    }
    section.main [data-testid="stWidgetLabel"] p,
    section.main label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    section.main h1, section.main h2, section.main h3 {
        color: #0f172a !important;
    }
    section.main .stCaption, section.main [data-testid="stCaption"] {
        color: #475569 !important;
    }
    /* Sidebar — only inside sidebar */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e3a5f 0%, #152238 100%);
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e8eef7 !important;
    }
    /* Section cards */
    div.card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.75rem;
    }
    div.card h3 { color: #0f172a !important; }
    /* Market basket — header strip */
    div.mba-hero {
        background: linear-gradient(125deg, #0f172a 0%, #1e3a5f 55%, #1d4ed8 100%);
        border-radius: 14px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
    }
    div.mba-hero h2 { color: #f8fafc !important; margin: 0 0 0.4rem 0; font-size: 1.35rem; }
    div.mba-hero p { color: #cbd5e1 !important; margin: 0; font-size: 0.95rem; }
    /* Colored "Compare all models" button — uses a sibling-selector anchor */
    div.cmp-btn-anchor { display: none; }
    div.element-container:has(div.cmp-btn-anchor) + div.element-container button {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%) !important;
        color: #ffffff !important;
        border: 0 !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35) !important;
    }
    div.element-container:has(div.cmp-btn-anchor) + div.element-container button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }
    /* Panel that wraps the full-width comparison block */
    div.cmp-panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.25rem 0.5rem 1.25rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
        margin-top: 0.5rem;
    }
    div.cmp-panel h3 { color: #0f172a !important; margin: 0 0 0.25rem 0; }
    div.cmp-panel p  { color: #475569 !important; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def card(title: str, body: str) -> None:
    """Small HTML card for section headers."""
    st.markdown(
        f'<div class="card"><h3 style="margin:0 0 0.35rem 0;">{title}</h3>'
        f'<p style="margin:0;color:#475569;">{body}</p></div>',
        unsafe_allow_html=True,
    )


def polish_fig(fig, height: int = 360):
    """Shared chart styling (clean, presentation-friendly)."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI, Tahoma, sans-serif", size=12, color="#0f172a"),
        title_font_size=15,
        title_font_color="#0f172a",
        height=height,
        margin=dict(l=48, r=24, t=56, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
    )
    return fig


# 1) Stroke pipeline 
NUM_COLS = ["age", "avg_glucose_level", "bmi"]


@st.cache_data(show_spinner=False)
def load_stroke_raw() -> pd.DataFrame:
    if not STROKE_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {STROKE_CSV}")
    return pd.read_csv(STROKE_CSV)


@st.cache_resource(show_spinner=True)
def train_stroke_models():
    """
    Same preprocessing + same four classifiers as the Jupyter notebook:
    Decision Tree, RIPPER, SVM (RBF), Naive Bayes — trained AFTER preprocessing.
    """
    df_proc = load_stroke_raw().copy()
    df_proc = df_proc.drop(columns=["id"])

    df_proc["bmi"] = pd.to_numeric(df_proc["bmi"], errors="coerce")
    bmi_mean = float(df_proc["bmi"].mean())
    df_proc["bmi"] = df_proc["bmi"].fillna(bmi_mean)

    df_proc = df_proc[df_proc["gender"] != "Other"].reset_index(drop=True)

    # Label encoding
    encoders: dict[str, LabelEncoder] = {}
    df_enc = df_proc.copy()
    cat_cols = df_enc.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    # Binning
    df_enc["age_bin"] = pd.cut(
        df_enc["age"], bins=[0, 40, 60, 100], labels=[0, 1, 2]
    ).astype(int)
    df_enc["glucose_bin"] = pd.cut(
        df_enc["avg_glucose_level"], bins=[0, 100, 126, 300], labels=[0, 1, 2]
    ).astype(int)
    df_enc["bmi_bin"] = pd.cut(
        df_enc["bmi"], bins=[0, 24.9, 29.9, 100], labels=[0, 1, 2]
    ).astype(int)

    scaler = MinMaxScaler()
    df_enc[NUM_COLS] = scaler.fit_transform(df_enc[NUM_COLS])

    X = df_enc.drop("stroke", axis=1)
    y = df_enc["stroke"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Models
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    dt_model.fit(X_train, y_train)

    ripper_model = lw.RIPPER(random_state=42, max_rules=10)
    ripper_model.fit(X_train, y_train.astype(str), pos_class="1")

    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    )
    svm_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    models = {
        "Decision Tree": dt_model,
        "Rule-Based (RIPPER)": ripper_model,
        "SVM": svm_model,
        "Naive Bayes": nb_model,
    }

    # Test-set metrics + predictions
    metrics = {}
    preds_store: dict[str, np.ndarray] = {}
    for name, model in models.items():
        if name == "Rule-Based (RIPPER)":
            y_pred = np.array([int(p) for p in model.predict(X_test)])
        else:
            y_pred = model.predict(X_test)
        preds_store[name] = np.asarray(y_pred)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

    return {
        "models": models,
        "metrics": metrics,
        "preds": preds_store,
        "y_test": np.asarray(y_test),
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": feature_names,
        "bmi_mean": bmi_mean,
        "cat_cols": cat_cols,
    }


def encode_stroke_row(raw: dict, bundle: dict) -> pd.DataFrame:
    """Turn sidebar inputs into one preprocessed row (same order as training)."""
    encoders: dict[str, LabelEncoder] = bundle["encoders"]
    scaler: MinMaxScaler = bundle["scaler"]
    bmi_mean: float = bundle["bmi_mean"]

    bmi_val = raw["bmi"]
    row = {
        "gender": raw["gender"],
        "age": float(raw["age"]),
        "hypertension": int(raw["hypertension"]),
        "heart_disease": int(raw["heart_disease"]),
        "ever_married": raw["ever_married"],
        "work_type": raw["work_type"],
        "Residence_type": raw["Residence_type"],
        "avg_glucose_level": float(raw["avg_glucose_level"]),
        "bmi": float(bmi_val) if bmi_val is not None else np.nan,
        "smoking_status": raw["smoking_status"],
    }

    df1 = pd.DataFrame([row])

    # BMI fill with training-time mean
    df1["bmi"] = pd.to_numeric(df1["bmi"], errors="coerce")
    df1["bmi"] = df1["bmi"].fillna(bmi_mean)

    # Label encode
    for col, le in encoders.items():
        val = str(df1.loc[0, col])
        if val not in le.classes_:
            raise ValueError(
                f"The value '{val}' for '{col}' was never seen while training. "
                "Pick another category from the lists."
            )
        df1[col] = le.transform([val])[0]

    # Bins
    df1["age_bin"] = (
        pd.cut(df1["age"], bins=[0, 40, 60, 100], labels=[0, 1, 2]).astype(int)
    )
    df1["glucose_bin"] = (
        pd.cut(
            df1["avg_glucose_level"], bins=[0, 100, 126, 300], labels=[0, 1, 2]
        ).astype(int)
    )
    df1["bmi_bin"] = (
        pd.cut(df1["bmi"], bins=[0, 24.9, 29.9, 100], labels=[0, 1, 2]).astype(int)
    )

    # Scale numeric columns
    df1[NUM_COLS] = scaler.transform(df1[NUM_COLS])

    # Final column order
    X1 = df1[bundle["feature_names"]]
    return X1


# 2) Groceries + MBA helpers
@st.cache_data(show_spinner=False)
def load_groceries_list() -> list[list[str]]:
    """Each inner list is one shopping basket (list of item names)."""
    path = GROCERIES_CSV
    if not path.exists():
        import urllib.request

        urllib.request.urlretrieve(GROCERIES_URL, path)

    text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    baskets: list[list[str]] = []
    for line in text:
        items = [x.strip() for x in line.split(",") if x.strip()]
        if items:
            baskets.append(items)
    return baskets


@st.cache_data(show_spinner=False)
def groceries_one_hot() -> pd.DataFrame:
    """One-hot matrix used by Apriori / FP-Growth (same as mlxtend tutorials)."""
    dataset = load_groceries_list()
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    return pd.DataFrame(te_ary, columns=te.columns_)


@st.cache_data(show_spinner=False)
def groceries_dataset_stats() -> dict:
    """Summary numbers for the basket dataset (for dashboard headers)."""
    baskets = load_groceries_list()
    sizes = [len(b) for b in baskets]
    te = TransactionEncoder()
    te.fit(baskets)
    return {
        "n_transactions": len(baskets),
        "n_unique_items": len(te.columns_),
        "avg_items_per_basket": round(sum(sizes) / max(len(sizes), 1), 2),
        "basket_sizes": sizes,
    }


@st.cache_data(show_spinner=False)
def groceries_top_items(top_n: int = 18) -> pd.DataFrame:
    """Most frequent single products across all baskets."""
    from collections import Counter

    c: Counter = Counter()
    for basket in load_groceries_list():
        for it in basket:
            c[it] += 1
    rows = c.most_common(top_n)
    return pd.DataFrame(rows, columns=["Item", "Transactions"])


def frozenset_to_str(fs) -> str:
    return ", ".join(sorted(list(fs)))


def run_mba(algorithm: str, min_sup: float, min_conf: float) -> dict:
    """Run chosen MBA algorithm; return tables + timing."""
    df_ohe = groceries_one_hot()

    t0 = time.perf_counter()
    if algorithm == "Apriori":
        freq = apriori(df_ohe, min_support=min_sup, use_colnames=True)
    else:
        freq = fpgrowth(df_ohe, min_support=min_sup, use_colnames=True)
    t_mine = time.perf_counter() - t0

    rules = pd.DataFrame()
    t_rules = 0.0
    if len(freq) > 0:
        t1 = time.perf_counter()
        rules = association_rules(
            freq, metric="confidence", min_threshold=min_conf
        )
        t_rules = time.perf_counter() - t1
        if len(rules) > 0:
            rules = rules.sort_values("lift", ascending=False)

    runtime = t_mine + t_rules

    # Friendly tables for Streamlit
    freq_display = freq.copy()
    if len(freq_display) > 0:
        freq_display["itemsets"] = freq_display["itemsets"].apply(
            lambda x: ", ".join(sorted(list(x)))
        )

    rules_display = rules.copy()
    if len(rules_display) > 0:
        rules_display["antecedents"] = rules_display["antecedents"].apply(
            frozenset_to_str
        )
        rules_display["consequents"] = rules_display["consequents"].apply(
            frozenset_to_str
        )

    top10 = rules_display.head(10) if len(rules_display) > 0 else rules_display

    n_itemsets_table = int(len(freq_display))

    return {
        "algorithm": algorithm,
        "min_support": min_sup,
        "min_confidence": min_conf,
        "frequent_itemsets": freq_display,
        "rules": rules_display,
        "top10": top10,
        "runtime_sec": runtime,
        "runtime_mine": float(t_mine),
        "runtime_rules": float(t_rules),
        "n_itemsets": n_itemsets_table,
        "n_rules": int(len(rules)),
    }

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    st.caption("Stroke prediction · Shopping baskets")
    st.markdown("---")
    section = st.radio(
        "Go to section",
        ("🧠 Stroke prediction", "🛒 Market basket analysis"),
        label_visibility="collapsed",
    )
    st.markdown("---")

# Main layout
st.title("Stroke prediction & market baskets")
st.caption("Compare stroke classifiers on patient inputs, then mine basket rules with Apriori or FP-Growth.")

if section.startswith("🧠"):
    card(
        "Section 1 — Stroke prediction",
        "Same preprocessing and four classifiers as in the original stroke project.",
    )

    bundle = train_stroke_models()
    df_raw = load_stroke_raw()
    df_plot = df_raw[df_raw["gender"] != "Other"].copy()

    # (dataset + model comparison)
    st.subheader("Visual overview")
    st.caption("Quick charts from the stroke CSV and from the trained models (test set).")
    g1, g2, g3 = st.columns(3)
    vc = df_plot["stroke"].value_counts().sort_index()
    with g1:
        fig_counts = px.bar(
            x=["No stroke (0)", "Stroke (1)"],
            y=[int(vc.get(0, 0)), int(vc.get(1, 0))],
            labels={"x": "Class", "y": "Patients"},
            title="Patients in dataset",
            color_discrete_sequence=["#3b82f6", "#f97316"],
        )
        fig_counts.update_layout(height=320, showlegend=False)
        st.plotly_chart(polish_fig(fig_counts, height=320), use_container_width=True)
    with g2:
        df_viz = df_plot.copy()
        df_viz["stroke_label"] = df_viz["stroke"].map({0: "No stroke", 1: "Stroke"})
        fig_age = px.histogram(
            df_viz,
            x="age",
            color="stroke_label",
            barmode="overlay",
            opacity=0.55,
            nbins=30,
            title="Age distribution",
            color_discrete_map={"No stroke": "#3b82f6", "Stroke": "#ea580c"},
        )
        fig_age.update_layout(height=320, legend_title_text="")
        st.plotly_chart(polish_fig(fig_age, height=320), use_container_width=True)
    with g3:
        comp_rows = []
        for name, met in bundle["metrics"].items():
            for k in ("accuracy", "precision", "recall", "f1"):
                comp_rows.append(
                    {"Model": name.replace("Rule-Based (RIPPER)", "RIPPER"), "Metric": k.title(), "Score": met[k]}
                )
        comp_df = pd.DataFrame(comp_rows)
        fig_comp = px.bar(
            comp_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="All models — test scores",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_comp.update_layout(height=320, yaxis_range=[0, 1.05], legend_orientation="h", legend_y=-0.35)
        st.plotly_chart(polish_fig(fig_comp, height=320), use_container_width=True)

    st.subheader("Risk factors in the dataset")
    st.caption("Glucose levels and hypertension counts split by stroke outcome (raw CSV, same filter as above).")
    g4, g5 = st.columns(2)
    df_plot2 = df_plot.copy()
    df_plot2["stroke_lab"] = df_plot2["stroke"].map({0: "No stroke", 1: "Stroke"})
    with g4:
        fig_glc = px.box(
            df_plot2,
            x="stroke_lab",
            y="avg_glucose_level",
            color="stroke_lab",
            title="Average glucose by stroke outcome",
            labels={"stroke_lab": "", "avg_glucose_level": "Average glucose"},
            color_discrete_map={"No stroke": "#3b82f6", "Stroke": "#ea580c"},
        )
        fig_glc.update_layout(showlegend=False)
        st.plotly_chart(polish_fig(fig_glc, height=340), use_container_width=True)
    with g5:
        agg_ht = (
            df_plot.groupby(["hypertension", "stroke"])
            .size()
            .reset_index(name="Patients")
        )
        agg_ht["Hypertension"] = agg_ht["hypertension"].map({0: "No", 1: "Yes"})
        agg_ht["Stroke"] = agg_ht["stroke"].map({0: "No stroke", 1: "Stroke"})
        fig_ht = px.bar(
            agg_ht,
            x="Hypertension",
            y="Patients",
            color="Stroke",
            barmode="group",
            title="Hypertension vs stroke (patient counts)",
            color_discrete_map={"No stroke": "#3b82f6", "Stroke": "#ea580c"},
        )
        fig_ht.update_layout(legend_title_text="")
        st.plotly_chart(polish_fig(fig_ht, height=340), use_container_width=True)

    st.divider()

    col_left, col_right = st.columns((1.05, 1.15), gap="large")

    with col_left:
        st.subheader("Patient form")
        st.caption("Fill each field below, then press **Predict stroke risk**.")
        genders = [g for g in sorted(df_raw["gender"].dropna().unique()) if g != "Other"]
        st.markdown("**1. Gender**")
        gender = st.selectbox("Gender", genders, label_visibility="collapsed")
        st.markdown("**2. Age (years)**")
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=52.0, step=1.0, label_visibility="collapsed")
        st.markdown("**3. Hypertension** — `0` = No, `1` = Yes")
        hypertension = st.selectbox("Hypertension", [0, 1], label_visibility="collapsed")
        st.markdown("**4. Heart disease** — `0` = No, `1` = Yes")
        heart_disease = st.selectbox("Heart disease", [0, 1], label_visibility="collapsed")
        ever_opts = sorted(df_raw["ever_married"].dropna().unique())
        st.markdown("**5. Ever married**")
        ever_married = st.selectbox("Ever married", ever_opts, label_visibility="collapsed")
        work_opts = sorted(df_raw["work_type"].dropna().unique())
        st.markdown("**6. Work type**")
        work_type = st.selectbox("Work type", work_opts, label_visibility="collapsed")
        res_opts = sorted(df_raw["Residence_type"].dropna().unique())
        st.markdown("**7. Residence type**")
        Residence_type = st.selectbox("Residence type", res_opts, label_visibility="collapsed")
        st.markdown("**8. Average glucose level**")
        glucose = st.number_input(
            "Glucose", min_value=0.0, max_value=400.0, value=100.0, step=1.0, label_visibility="collapsed"
        )
        st.markdown("**9. BMI** — use `0` to auto-fill with training mean")
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=80.0,
            value=28.0,
            step=0.1,
            label_visibility="collapsed",
        )
        smoke_opts = sorted(df_raw["smoking_status"].dropna().unique())
        st.markdown("**10. Smoking status**")
        smoking_status = st.selectbox("Smoking status", smoke_opts, label_visibility="collapsed")

        st.markdown("**11. Classifier**")
        model_name = st.selectbox(
            "Classifier",
            list(bundle["models"].keys()),
            label_visibility="collapsed",
        )

        predict_clicked = st.button("Predict stroke risk", type="primary", use_container_width=True)

        # Colored "Compare all models" button
        st.markdown('<div class="cmp-btn-anchor"></div>', unsafe_allow_html=True)
        compare_clicked = st.button(
            "📊  Compare all models",
            use_container_width=True,
            key="cmp_models_btn",
            help="Show test-set scores for all four classifiers side-by-side.",
        )
        if compare_clicked:
            st.session_state["show_model_cmp"] = not st.session_state.get(
                "show_model_cmp", False
            )

    with col_right:
        st.subheader("Model performance (test set)")
        m = bundle["metrics"][model_name]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{m['accuracy']:.3f}")
        m2.metric("Precision", f"{m['precision']:.3f}")
        m3.metric("Recall", f"{m['recall']:.3f}")
        m4.metric("F1 score", f"{m['f1']:.3f}")
        st.caption(
            "Scores use the same 80/20 split (`random_state=42`) as in the training script."
        )

        # Confusion matrix for the selected model
        y_t = bundle["y_test"]
        pred = bundle["preds"][model_name]
        cm = confusion_matrix(y_t, pred, labels=[0, 1])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["No stroke (0)", "Stroke (1)"],
            y=["No stroke (0)", "Stroke (1)"],
            color_continuous_scale="Blues",
            title=f"Confusion matrix — {model_name}",
        )
        fig_cm.update_layout(height=340)
        st.plotly_chart(polish_fig(fig_cm, height=340), use_container_width=True)

        st.subheader("Prediction")
        if predict_clicked:
            try:
                raw = {
                    "gender": gender,
                    "age": age,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "ever_married": ever_married,
                    "work_type": work_type,
                    "Residence_type": Residence_type,
                    "avg_glucose_level": glucose,
                    "bmi": None if bmi == 0 else bmi,
                    "smoking_status": smoking_status,
                }
                X_row = encode_stroke_row(raw, bundle)
                model = bundle["models"][model_name]
                if model_name == "Rule-Based (RIPPER)":
                    pred_out = int(model.predict(X_row)[0])
                else:
                    pred_out = int(model.predict(X_row)[0])

                if pred_out == 1:
                    st.error("Predicted class: **Stroke (1)** — higher risk flag.")
                else:
                    st.success("Predicted class: **No stroke (0)** — model says low risk.")
                st.write("Raw model output:", pred_out)
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error("Something went wrong while predicting.")
                st.exception(e)

    # Full-width "Compare all models" panel
    if st.session_state.get("show_model_cmp"):
        st.markdown(
            '<div class="cmp-panel"><h3>Model comparison — all classifiers</h3>'
            '<p>Same 80/20 test split for every model. Click the comparison button again to hide.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        all_metrics = bundle["metrics"]
        short_name = {
            "Decision Tree": "Decision Tree",
            "Rule-Based (RIPPER)": "RIPPER",
            "SVM": "SVM",
            "Naive Bayes": "Naive Bayes",
        }

        # Long-format dataframe for the grouped bar chart
        cmp_rows = []
        for name, met in all_metrics.items():
            for k in ("accuracy", "precision", "recall", "f1"):
                cmp_rows.append(
                    {
                        "Model": short_name.get(name, name),
                        "Metric": k.title(),
                        "Score": met[k],
                    }
                )
        cmp_long = pd.DataFrame(cmp_rows)

        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            fig_all = px.bar(
                cmp_long,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="All metrics by model",
                color_discrete_sequence=["#2563eb", "#ea580c", "#059669", "#7c3aed"],
                text=cmp_long["Score"].round(2),
            )
            fig_all.update_traces(
                textposition="outside",
                textfont_size=11,
                cliponaxis=False,
            )
            fig_all.update_layout(
                yaxis_range=[0, 1.12],
                legend_orientation="h",
                legend_y=-0.22,
                legend_title_text="",
                bargap=0.25,
                bargroupgap=0.05,
            )
            fig_all.update_xaxes(tickangle=0)
            st.plotly_chart(polish_fig(fig_all, height=460), use_container_width=True)

        with cc2:
            fig_radar = go.Figure()
            metric_order = ["Accuracy", "Precision", "Recall", "F1"]
            palette = ["#2563eb", "#ea580c", "#059669", "#7c3aed"]
            for i, (name, met) in enumerate(all_metrics.items()):
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[met["accuracy"], met["precision"], met["recall"], met["f1"]],
                        theta=metric_order,
                        fill="toself",
                        name=short_name.get(name, name),
                        line=dict(color=palette[i % len(palette)], width=2),
                        opacity=0.45,
                    )
                )
            fig_radar.update_layout(
                title="Model profile (radar)",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=10, color="#475569"),
                    ),
                    angularaxis=dict(tickfont=dict(size=12, color="#0f172a")),
                    bgcolor="#f8fafc",
                ),
                legend_orientation="h",
                legend_y=-0.18,
                legend_title_text="",
            )
            st.plotly_chart(polish_fig(fig_radar, height=460), use_container_width=True)

        # Wide table of exact values for every model
        table_df = (
            pd.DataFrame(all_metrics)
            .T.rename_axis("Model")
            .reset_index()
            .rename(
                columns={
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                }
            )
        )
        table_df["Model"] = table_df["Model"].map(lambda n: short_name.get(n, n))
        st.markdown("**Exact metric values (all models)**")
        st.dataframe(
            table_df.style.format(
                {"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
            ).background_gradient(
                cmap="Blues",
                subset=["Accuracy", "Precision", "Recall", "F1"],
                vmin=0,
                vmax=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

else:
    card(
        "Section 2 — Market basket analysis",
        "Discover frequent itemsets and association rules with Apriori or FP-Growth (mlxtend).",
    )

    stats = groceries_dataset_stats()
    st.markdown(
        '<div class="mba-hero">'
        "<h2>Market basket analysis</h2>"
        "<p>Same support & confidence for both algorithms — compare speed and how many patterns each finds.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Transactions", f"{stats['n_transactions']:,}")
    h2.metric("Unique products", f"{stats['n_unique_items']:,}")
    h3.metric("Avg. items / basket", f"{stats['avg_items_per_basket']}")
    h4.metric("Source file", "groceries.csv")

    st.subheader("Dataset profile")
    prof_l, prof_r = st.columns(2)
    with prof_l:
        fig_bs = px.histogram(
            pd.DataFrame({"Items per basket": stats["basket_sizes"]}),
            x="Items per basket",
            nbins=24,
            title="How many products shoppers buy per trip",
            color_discrete_sequence=["#4f46e5"],
        )
        st.plotly_chart(polish_fig(fig_bs, height=300), use_container_width=True)
    with prof_r:
        topi = groceries_top_items(16)
        fig_top = px.bar(
            topi,
            x="Transactions",
            y="Item",
            orientation="h",
            title="Most popular single products (frequency)",
            color_discrete_sequence=["#0d9488"],
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(polish_fig(fig_top, height=300), use_container_width=True)

    st.divider()
    st.subheader("Rule mining settings")
    c1, c2 = st.columns([1, 1])
    with c1:
        min_sup = st.slider("Minimum support", 0.005, 0.15, 0.03, 0.005)
    with c2:
        min_conf = st.slider("Minimum confidence", 0.10, 0.90, 0.40, 0.05)
    st.caption(
        "**Important:** after you move the sliders, click **Run Apriori** / **Run FP-Growth** / **Compare** again — "
        "numbers only refresh when you run."
    )
    b_ap, b_fp = st.columns(2)
    with b_ap:
        b_run_ap = st.button("Run Apriori", type="primary", use_container_width=True)
    with b_fp:
        b_run_fp = st.button("Run FP-Growth", type="primary", use_container_width=True)

    b_cmp = st.button("Compare Apriori & FP-Growth (same thresholds)", use_container_width=True)

    if b_run_ap:
        with st.spinner("Running Apriori…"):
            res = run_mba("Apriori", min_sup, min_conf)
        st.session_state["mba_last"] = res
    if b_run_fp:
        with st.spinner("Running FP-Growth…"):
            res = run_mba("FP-Growth", min_sup, min_conf)
        st.session_state["mba_last"] = res

    if b_cmp:
        with st.spinner("Running both algorithms…"):
            ra = run_mba("Apriori", min_sup, min_conf)
            rf = run_mba("FP-Growth", min_sup, min_conf)
        st.session_state["mba_cmp"] = {"Apriori": ra, "FP-Growth": rf}

    if "mba_last" not in st.session_state:
        st.info("Adjust sliders if you like, then run **Apriori** or **FP-Growth** to see charts and tables.")

    if "mba_last" in st.session_state:
        res = st.session_state["mba_last"]
        show_cols = [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
        ]
        st.markdown("---")
        st.subheader(f"Last run — {res['algorithm']}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total runtime (s)", f"{res['runtime_sec']:.4f}")
        k2.metric("Mining phase (s)", f"{res['runtime_mine']:.4f}")
        k3.metric("Rules phase (s)", f"{res['runtime_rules']:.4f}")
        k4.metric("Frequent itemsets", f"{res['n_itemsets']}")
        k5.metric("Association rules", f"{res['n_rules']}")
        n_rows_fi = len(res["frequent_itemsets"])
        st.caption(
            f"These results use **support ≥ {res['min_support']}** and **confidence ≥ {res['min_confidence']}**. "
            f"The **Frequent itemsets** number is the full table size (**{n_rows_fi}** rows). "
            "If the table looks short, **scroll inside the grid** — only part of the rows fit on screen."
        )

        tab_vis, tab_tbl = st.tabs(["Charts", "Tables"])

        with tab_vis:
            if res["n_itemsets"] == 0:
                st.warning(
                    "No frequent itemsets at this support level. Lower **minimum support** and try again."
                )
            else:
                c_a, c_b = st.columns(2)
                with c_a:
                    top_plot = (
                        res["frequent_itemsets"]
                        .sort_values("support", ascending=False)
                        .head(15)
                    )
                    fig_item = px.bar(
                        top_plot,
                        x="support",
                        y="itemsets",
                        orientation="h",
                        title="Top frequent itemsets (support)",
                        labels={"support": "Support", "itemsets": "Itemset"},
                        color_discrete_sequence=["#2563eb"],
                    )
                    fig_item.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(polish_fig(fig_item, height=420), use_container_width=True)
                with c_b:
                    fp_sz = res["frequent_itemsets"].copy()
                    fp_sz["Items in set"] = fp_sz["itemsets"].str.split(",").str.len()
                    sz_counts = (
                        fp_sz.groupby("Items in set").size().reset_index(name="Count")
                    )
                    fig_sz = px.bar(
                        sz_counts,
                        x="Items in set",
                        y="Count",
                        title="How large are the frequent patterns?",
                        labels={"Items in set": "Number of products in itemset"},
                        color_discrete_sequence=["#7c3aed"],
                    )
                    st.plotly_chart(polish_fig(fig_sz, height=420), use_container_width=True)
                    st.caption(
                        "This chart groups patterns by **how many products** are in the itemset. "
                        "The **heights** add up to the total frequent itemset count — it is **not** one bar per itemset."
                    )

            if res["n_rules"] == 0:
                st.warning(
                    "No rules passed the confidence threshold. Try a **lower minimum confidence** "
                    "or a **lower minimum support**."
                )
            else:
                rsub = res["rules"].head(50)
                fig_sc = px.scatter(
                    rsub,
                    x="confidence",
                    y="lift",
                    size="support",
                    hover_data=["antecedents", "consequents"],
                    title="Rules landscape — confidence vs lift (up to 50 rules)",
                    color_discrete_sequence=["#db2777"],
                )
                fig_sc.update_traces(marker=dict(line=dict(width=0.5, color="white")))
                st.plotly_chart(polish_fig(fig_sc, height=400), use_container_width=True)

        with tab_tbl:
            if res["n_itemsets"] > 0:
                st.markdown("**Frequent itemsets**")
                st.caption(
                    f"**{len(res['frequent_itemsets'])}** rows — scroll the grid or download to count in Excel/Sheets."
                )
                st.download_button(
                    label="Download frequent itemsets (CSV)",
                    data=res["frequent_itemsets"]
                    .sort_values("support", ascending=False)
                    .to_csv(index=False)
                    .encode("utf-8"),
                    file_name="frequent_itemsets.csv",
                    mime="text/csv",
                )
                st.dataframe(
                    res["frequent_itemsets"].sort_values("support", ascending=False),
                    use_container_width=True,
                    height=420,
                )
            if res["n_rules"] > 0:
                st.markdown("**Top 10 rules (by lift)**")
                st.dataframe(res["top10"][show_cols], use_container_width=True, height=360)
                st.markdown("**All association rules**")
                st.dataframe(
                    res["rules"][show_cols], use_container_width=True, height=320
                )

    if "mba_cmp" in st.session_state:
        st.markdown("---")
        st.subheader("Apriori vs FP-Growth — comparison")
        cmp = st.session_state["mba_cmp"]
        ra, rf = cmp["Apriori"], cmp["FP-Growth"]

        c4, c5, c6, c7, c8, c9 = st.columns(6)
        c4.metric("Apriori total (s)", f"{ra['runtime_sec']:.4f}")
        c5.metric("FP-Growth total (s)", f"{rf['runtime_sec']:.4f}")
        c6.metric("Apriori rules", f"{ra['n_rules']}")
        c7.metric("FP rules", f"{rf['n_rules']}")
        c8.metric("Apriori itemsets", f"{ra['n_itemsets']}")
        c9.metric("FP itemsets", f"{rf['n_itemsets']}")

        st.info(
            "For the **same** minimum support and confidence, **Apriori** and **FP-Growth** must find the "
            "**same** frequent itemsets and therefore generate the **same** association rules. "
            "Only **runtime** (especially the mining step) should differ. "
            "That is why the **rule / itemset count** bars often look identical — it is correct, not a bug."
        )

        cmp_time = pd.DataFrame(
            [
                {"Algorithm": "Apriori", "Step": "Mine itemsets", "Seconds": ra["runtime_mine"]},
                {"Algorithm": "Apriori", "Step": "Build rules", "Seconds": ra["runtime_rules"]},
                {"Algorithm": "FP-Growth", "Step": "Mine itemsets", "Seconds": rf["runtime_mine"]},
                {"Algorithm": "FP-Growth", "Step": "Build rules", "Seconds": rf["runtime_rules"]},
            ]
        )
        cmp_out = pd.DataFrame(
            [
                {
                    "Algorithm": "Apriori",
                    "Output": "Frequent itemsets",
                    "Count": ra["n_itemsets"],
                },
                {"Algorithm": "Apriori", "Output": "Association rules", "Count": ra["n_rules"]},
                {
                    "Algorithm": "FP-Growth",
                    "Output": "Frequent itemsets",
                    "Count": rf["n_itemsets"],
                },
                {"Algorithm": "FP-Growth", "Output": "Association rules", "Count": rf["n_rules"]},
            ]
        )

        cmp_mine_only = pd.DataFrame(
            [
                {"Algorithm": "Apriori", "Phase": "Mining itemsets", "Seconds": ra["runtime_mine"]},
                {"Algorithm": "FP-Growth", "Phase": "Mining itemsets", "Seconds": rf["runtime_mine"]},
            ]
        )

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            fig_time = px.bar(
                cmp_time,
                x="Algorithm",
                y="Seconds",
                color="Step",
                barmode="stack",
                title="Runtime breakdown (stacked)",
                color_discrete_sequence=["#3b82f6", "#f59e0b"],
            )
            fig_time.update_layout(legend_orientation="h", legend_y=-0.22)
            st.plotly_chart(polish_fig(fig_time, height=400), use_container_width=True)
        with cmp_col2:
            fig_mine = px.bar(
                cmp_mine_only,
                x="Algorithm",
                y="Seconds",
                title="Mining step only (this should usually differ)",
                labels={"Seconds": "Seconds"},
                color="Algorithm",
                color_discrete_map={"Apriori": "#6366f1", "FP-Growth": "#14b8a6"},
            )
            fig_mine.update_layout(showlegend=False)
            st.plotly_chart(polish_fig(fig_mine, height=400), use_container_width=True)

        with st.expander("Itemset & rule counts (both algorithms)", expanded=False):
            st.caption("These numbers should match pairwise — see the blue info box above.")
            st.dataframe(
                cmp_out.rename(columns={"Output": "Measure", "Count": "Rows"}),
                use_container_width=True,
                hide_index=True,
            )

        if ra["n_rules"] > 0 and rf["n_rules"] > 0:
            st.caption("Rule strength — same axes for both algorithms (up to 40 rules each).")
            s1, s2 = st.columns(2)
            with s1:
                rra = ra["rules"].head(40)
                fig_sa = px.scatter(
                    rra,
                    x="confidence",
                    y="lift",
                    size="support",
                    hover_data=["antecedents", "consequents"],
                    title="Apriori — confidence vs lift",
                    color_discrete_sequence=["#2563eb"],
                )
                st.plotly_chart(polish_fig(fig_sa, height=360), use_container_width=True)
            with s2:
                rrf = rf["rules"].head(40)
                fig_sf = px.scatter(
                    rrf,
                    x="confidence",
                    y="lift",
                    size="support",
                    hover_data=["antecedents", "consequents"],
                    title="FP-Growth — confidence vs lift",
                    color_discrete_sequence=["#059669"],
                )
                st.plotly_chart(polish_fig(fig_sf, height=360), use_container_width=True)
