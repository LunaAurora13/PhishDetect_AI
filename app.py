import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="PhishDetect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.1rem; color: #4B5563; }
    .metric-box { font-size: 1.5rem !important; font-weight: 700; color: #1F2937; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached_assets():
    loaded_model = joblib.load("models/phishing_model.pkl")
    loaded_scaler = joblib.load("models/scaler.pkl")
    loaded_df = pd.read_csv("dataset/phishing_dataset.csv")
    return loaded_model, loaded_scaler, loaded_df


try:
    ml_model, data_scaler, dataset_df = load_cached_assets()
except Exception as err:
    st.error(
        f"Error loading model assets. Please run `train_model.py` first! Details:"
        f" {err}"
    )
    st.stop()

if not isinstance(dataset_df, pd.DataFrame):
    dataset_df = pd.DataFrame(dataset_df)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/domain--v1.png", width=70)
st.sidebar.markdown("### PhishDetect AI")
st.sidebar.markdown("Cybersecurity Intelligence Platform")
nav_option = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔍 Website Analyzer",
        "📈 Model Evaluation",
        "📜 Prediction History",
    ],
)

if "history" not in st.session_state:
    st.session_state.history = []

# ==================== 1. DASHBOARD ====================
if nav_option == "📊 Dashboard":
    st.markdown(
        '<p class="main-header">🛡️ PhishDetect AI Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Machine Learning-Based Suspicious Website'
        " Detection Using Logistic Regression</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Dataset Records**")
        st.markdown(
            '<p class="metric-box">10,000</p>', unsafe_allow_html=True
        )
    with col2:
        st.markdown("**Input Features**")
        st.markdown(
            '<p class="metric-box">8 Numerical</p>', unsafe_allow_html=True
        )
    with col3:
        st.markdown("**Target Classes**")
        st.markdown(
            '<p class="metric-box" style="font-size: 1.1rem !important;">2'
            " (Legitimate/Suspicious)</p>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown("**Algorithm**")
        st.markdown(
            '<p class="metric-box" style="font-size: 1.1rem !important;">Logistic'
            " Regression</p>",
            unsafe_allow_html=True,
        )

    st.markdown("### 📌 Project Overview")
    st.write(
        "Phishing websites deceive users into sharing sensitive information"
        " like passwords and financial details. This educational detection"
        " prototype analyzes 8 numerical URL characteristics using **Logistic"
        " Regression** to classify whether a website is legitimate or"
        " suspicious."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            "**Input Features Used:**\n"
            "1. `https_usage` (1=Yes, 0=No)\n"
            "2. `url_length` (Character count)\n"
            "3. `special_char_count` (Special symbols)\n"
            "4. `subdomain_count` (Subdomain levels)\n"
            "5. `ip_address_usage` (1=Yes, 0=No)\n"
            "6. `hyphen_count` (Hyphens in URL)\n"
            "7. `at_symbol_usage` (Contains '@')\n"
            "8. `suspicious_keyword_count` (Phishing keywords)"
        )
    with col_b:
        st.success(
            "**Class Definitions:**\n"
            "- **Class 0 (Legitimate):** Safe URL indicators detected.\n"
            "- **Class 1 (Suspicious):** Potential phishing indicators"
            " detected.\n\n"
            "*Note: This is an educational prototype and does not guarantee"
            " absolute real-world threat verification.*"
        )

# ==================== 2. WEBSITE ANALYZER ====================
elif nav_option == "🔍 Website Analyzer":
    st.markdown(
        '<p class="main-header">🔍 Website Security Analyzer</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Enter a website URL or manually tune feature values to analyze"
        " potential risk."
    )
    st.markdown("---")

    url_input = st.text_input(
        "Website URL", "https://secure-login-account-verify.com/update"
    )

    st.markdown("### Feature Extraction Preview")
    col1, col2 = st.columns(2)

    with col1:
        url_str = url_input if isinstance(url_input, str) else ""
        has_https = 1 if url_str.startswith("https") else 0
        url_len = len(url_str)
        spec_chars = sum(
            1 for c in url_str if c in ["@", "-", "_", "?", "=", "&", "%", "."]
        )
        subdomains = url_str.count(".") - 1 if url_str.count(".") > 1 else 0

        https_usage = st.selectbox(
            "HTTPS Usage",
            [0, 1],
            index=has_https,
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )
        url_length = st.number_input(
            "URL Length", min_value=5, max_value=300, value=url_len
        )
        special_char_count = st.number_input(
            "Special Character Count", min_value=0, max_value=50, value=spec_chars
        )
        subdomain_count = st.number_input(
            "Subdomain Count",
            min_value=0,
            max_value=10,
            value=max(0, subdomains),
        )

    with col2:
        domain_part = (
            url_str.split("//")[-1].split("/")[0]
            if "//" in url_str
            else url_str
        )
        ip_usage = 1 if any(char.isdigit() for char in domain_part) else 0
        hyphens = url_str.count("-")
        at_sym = 1 if "@" in url_str else 0
        kw_count = sum(
            1
            for kw in [
                "login",
                "verify",
                "update",
                "secure",
                "account",
                "banking",
                "signin",
            ]
            if kw in url_str.lower()
        )

        ip_address_usage = st.selectbox(
            "IP Address in URL",
            [0, 1],
            index=ip_usage,
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )
        hyphen_count = st.number_input(
            "Hyphen Count", min_value=0, max_value=20, value=hyphens
        )
        at_symbol_usage = st.selectbox(
            "@ Symbol Usage",
            [0, 1],
            index=at_sym,
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )
        suspicious_keyword_count = st.number_input(
            "Suspicious Keyword Count", min_value=0, max_value=10, value=kw_count
        )

    if st.button("Analyze URL Risk", type="primary"):
        input_data = np.array(
            [[
                https_usage,
                url_length,
                special_char_count,
                subdomain_count,
                ip_address_usage,
                hyphen_count,
                at_symbol_usage,
                suspicious_keyword_count,
            ]]
        )

        input_scaled = data_scaler.transform(input_data)
        prediction = ml_model.predict(input_scaled)[0]
        probabilities = ml_model.predict_proba(input_scaled)[0]

        legit_prob = float(probabilities[0]) * 100
        phish_prob = float(probabilities[1]) * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1:
                st.error(
                    f"### 🚨 Classification: SUSPICIOUS\n**Model Confidence:**"
                    f" {phish_prob:.2f}% risk indicators detected."
                )
            else:
                st.success(
                    f"### ✅ Classification: LEGITIMATE\n**Model Confidence:**"
                    f" {legit_prob:.2f}% safe indicators detected."
                )

        with res_col2:
            st.write("**Probability Breakdown:**")
            st.progress(int(phish_prob))
            st.write(
                f"- Legitimate Probability: **{legit_prob:.2f}%**\n- Suspicious"
                f" Probability: **{phish_prob:.2f}%**"
            )

        st.session_state.history.append({
            "URL": url_str,
            "Result": "Suspicious" if prediction == 1 else "Legitimate",
            "Confidence": f"{max(legit_prob, phish_prob):.2f}%",
        })

# ==================== 3. MODEL EVALUATION ====================
elif nav_option == "📈 Model Evaluation":
    st.markdown(
        '<p class="main-header">📈 Model Performance & Evaluation</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Complete analytical breakdown, evaluation metrics, and visual graphs"
        " tested on the exact 2K test dataset."
    )
    st.markdown("---")

    X = dataset_df.drop(columns=["website_status"])
    y = dataset_df["website_status"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=2000, random_state=42, stratify=y
    )
    X_test_scaled = data_scaler.transform(X_test)
    y_pred = ml_model.predict(X_test_scaled)
    y_prob = ml_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Accuracy", f"{acc * 100:.2f}%")
    with mcol2:
        st.metric("Precision", f"{precision * 100:.2f}%")
    with mcol3:
        st.metric("Recall", f"{recall * 100:.2f}%")
    with mcol4:
        st.metric("F1-Score", f"{f1 * 100:.2f}%")

    st.markdown("---")
    st.markdown(
        "### 📊 Comprehensive Project Graphs (Graphs 1 to 7 Overview)"
    )

    # --- GRAPH 1 & GRAPH 2 (Side by side) ---
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown(
            "#### 1. Website Classification Distribution (Bar Chart)"
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            x="website_status",
            data=dataset_df,
            palette=["#3B82F6", "#EF4444"],
            ax=ax,
        )
        ax.set_title(
            "Dataset Class Balance (Legitimate vs Suspicious)", fontsize=11
        )
        ax.set_xlabel("Website Status (0 = Legitimate, 1 = Suspicious)")
        ax.set_ylabel("Record Count")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Legitimate (0)", "Suspicious (1)"])
        st.pyplot(fig)
        plt.close(fig)

    with g_col2:
        st.markdown("#### 2. HTTPS Usage Comparison (Grouped Bar Chart)")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_bar = sns.countplot(
            x="https_usage",
            hue="website_status",
            data=dataset_df,
            palette=["#10B981", "#F59E0B"],
            ax=ax,
        )
        ax.set_title("HTTPS Usage by Website Status", fontsize=11)
        ax.set_xlabel("HTTPS Usage (0 = No, 1 = Yes)")
        ax.set_ylabel("Count")
        ax.legend(title="Status", labels=["Legitimate", "Suspicious"])

        # Add exact count numbers on top of each bar
        for p in plot_bar.patches:
            height = p.get_height()
            if height > 0:
                plot_bar.annotate(
                    f"{int(height):,}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    xytext=(0, 2),
                    textcoords="offset points",
                )

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # --- GRAPH 3: URL Feature Distribution ---
    st.markdown("#### 3. URL Feature Distribution (Histograms)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        dataset_df["url_length"],
        bins=30,
        kde=True,
        color="#2563EB",
        ax=ax,
    )
    ax.set_title("Distribution of URL Length Across Dataset", fontsize=11)
    ax.set_xlabel("URL Length (Character Count)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # --- GRAPH 4 & GRAPH 5 (Side by side) ---
    g_col3, g_col4 = st.columns(2)

    with g_col3:
        st.markdown(
            "#### 4. Feature Correlation Heatmap (With Numbers Included)"
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        corr_matrix = dataset_df.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax,
            cbar=True,
            annot_kws={"size": 7},
        )
        ax.set_title("Correlation Matrix of Features", fontsize=11)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=8
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    with g_col4:
        st.markdown(
            "#### 5. Confusion Matrix Heatmap (Exact 2K Test Records)"
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=["Legitimate (0)", "Suspicious (1)"],
            yticklabels=["Legitimate (0)", "Suspicious (1)"],
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(
            f"Confusion Matrix (Test Records: {len(y_test)})", fontsize=11
        )
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # --- GRAPH 6 & GRAPH 7 (Side by side) ---
    g_col5, g_col6 = st.columns(2)

    with g_col5:
        st.markdown(
            "#### 6. Model Performance Comparison (Accuracy, Precision, Recall,"
            " F1)"
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        metrics_vals = [acc * 100, precision * 100, recall * 100, f1 * 100]
        bars = ax.bar(
            metrics_names,
            metrics_vals,
            color=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Evaluation Metrics Comparison", fontsize=11)
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 2,
                f"{yval:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )
        st.pyplot(fig)
        plt.close(fig)

    with g_col6:
        st.markdown("#### 7. ROC Curve (Receiver Operating Characteristic)")
        fig, ax = plt.subplots(figsize=(6, 4))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color="#2563EB",
            lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - PhishDetect AI", fontsize=11)
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

# ==================== 4. PREDICTION HISTORY ====================
elif nav_option == "📜 Prediction History":
    st.markdown(
        '<p class="main-header">📜 Prediction History</p>',
        unsafe_allow_html=True,
    )
    st.markdown("List of URLs analyzed during this session.")
    st.markdown("---")

    if len(st.session_state.history) == 0:
        st.info(
            "No URLs analyzed yet. Head over to the **Website Analyzer** tab"
            " to test URLs!"
        )
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)