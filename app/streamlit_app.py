import sys
import os

# Get the project root directory regardless of where the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import roc_curve

from src.features import compute_woe_iv, compute_all_iv
from src.cleaning import report

# -------------------- Page Config --------------------
st.set_page_config(page_title="DataQuest 2026", layout="wide")
st.title("DataQuest 2026 - Credit Risk EDA")

# -------------------- Load Data --------------------
@st.cache_data
# this decorator tells Streamlit to only load the CSV once, and cache the result
def load_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "outputs", "loan_book_clean.csv")
    df = pd.read_csv(path)
    return df

df = load_data()
train = df[df["set"] == "train"].copy()
test = df[df["set"] == "test"].copy()

@st.cache_data
def load_models():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "outputs", "model_objects.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

models = load_models()

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["EDA", "Model", "Dashboard"])

with tab1:
    st.header("Exploratory Data Analysis")

    # -------------------- Section 1: Data Quality Report --------------------
    st.subheader("Data Quality Report")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Default Rate", f"{df['default_flag'].mean():.1%}")

    st.markdown("#### Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing>0].reset_index()
    missing.columns = ["Feature", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)

    if len(missing) > 0:
        st.dataframe(missing, use_container_width=True)
    else:
        st.success("No missing values")

    st.markdown("#### Data Types")
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ["Feature", "Type"]
    st.dataframe(dtypes_df, use_container_width=True)

    # -------------------- Section 2: Univariate Explorer --------------------
    st.subheader("Univariate Explorer")

    exclude_cols = ["applicant_id_hash", "default_flag", "set", "application_date"]
    feature_options = [c for c in df.columns if c not in exclude_cols]

    selected_feature = st.selectbox("Select a feature to explore", feature_options)

    woe_df, iv = compute_woe_iv(train, selected_feature, "default_flag")

    st.metric("Information Value (IV)", f"{iv:.4f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Feature Distribution")
        fig, ax = plt.subplots(figsize=(6,4))

        if pd.api.types.is_numeric_dtype(train[selected_feature]):
            ax.hist(train[selected_feature].dropna(), bins=30, color="steelblue", edgecolor="white")
        else:
            counts = train[selected_feature].value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="steelblue")
            plt.xticks(rotation=45, ha="right")

        ax.set_title(f"Distribution of {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Count")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("### Default Rate by Bin")

        # Calculate default rate per bin directly
        temp = train[[selected_feature, "default_flag"]].copy()
        if pd.api.types.is_numeric_dtype(train[selected_feature]):
            temp["bin"] = pd.qcut(temp[selected_feature], q=10, duplicates="drop")
        else:
            temp["bin"] = temp[selected_feature]

        grouped = temp.groupby("bin", observed=True)["default_flag"].mean().reset_index()
        grouped.columns = ["bin", "default_rate"]

        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(range(len(grouped)), grouped["default_rate"], color="coral")
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped["bin"].astype(str), rotation=45, ha="right")
        ax.set_title(f"Default Rate by Bin - {selected_feature}")
        ax.set_ylabel("Default Rate")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # WoE Table
    st.markdown("#### WoE / IV Table")
    display_cols = ["bin", "events", "non_events", "woe", "iv_bin"]
    woe_display = woe_df[display_cols].copy()
    woe_display["bin"] = woe_display["bin"].astype(str)
    st.dataframe(woe_display.round(4), use_container_width=True)

    # -------------------- Section 3: Bivariate Explorer --------------------
    st.subheader("Bivariate Explorer")

    col_a, col_b = st.columns(2)
    with col_a:
        feature_x = st.selectbox("Select first feature", feature_options, key="bivar_x")
    with col_b:
        feature_y = st.selectbox("Select second feature", feature_options, key="bivar_y")

    if feature_x == feature_y:
        st.warning("Please select two different features")
    else:
        def bin_feature(series, n_bins=5):
            if pd.api.types.is_numeric_dtype(series):
                binned = pd.qcut(series, q=n_bins, duplicates="drop")
                # Sort categories by their left edge so they can be displayed in order
                sorted_cats = sorted(binned.cat.categories, key=lambda x: x.left)
                binned = binned.cat.reorder_categories(sorted_cats)
                return binned.astype(str)
            else:
                return series.astype(str)


        temp = train[[feature_x, feature_y, "default_flag"]].copy()
        temp["x_bin"] = bin_feature(temp[feature_x])
        temp["y_bin"] = bin_feature(temp[feature_y])

        # Sort bin labels numerically
        x_order = sorted(temp["x_bin"].unique(), key=lambda s: float(s.split(",")[0].strip("([ ")))
        y_order = sorted(temp["y_bin"].unique(), key=lambda s: float(s.split(",")[0].strip("([ ")))

        # Cross table of default rates
        cross = temp.groupby(["x_bin", "y_bin"], observed=True)["default_flag"].mean().round(3)
        cross_table = cross.unstack(fill_value=0)

        # Reorder rows and columns
        cross_table = cross_table.reindex(index=x_order, columns=y_order)
        # Reverse rows so that lowest values are at the bottom
        cross_table = cross_table.iloc[::-1]

        # Heatmap
        st.markdown("#### Default Rate Heatmap")
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.heatmap(data=cross_table, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, linewidths=0.5)
        ax.set_title(f"Default Rate Heatmap: {feature_x} vs. {feature_y}")
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Cross table
        st.markdown("#### Default Rate Table")
        st.dataframe(cross_table, use_container_width=True)


with tab2:
    st.header("Model Evaluation")

    # -------------------- Key Metrics --------------------
    st.subheader("AUC Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline AUC", f"{models['baseline_auc']:.4f}")
    col2.metric("Improved AUC", f"{models['final_auc']:.4f}")
    col3.metric("Improvement", f"{models['final_auc'] - models['baseline_auc']:.4f}")

    # -------------------- ROC Curve --------------------
    st.subheader("ROC Curve Comparison")
    baseline_scaler = models["baseline_scaler"]
    final_scaler = models["final_scaler"]
    baseline_model = models["baseline_model"]
    final_model = models["final_model"]
    X_test_base = baseline_scaler.transform(test[models["baseline_features"]])
    y_pred_base = baseline_model.predict_proba(X_test_base)[:, 1]
    # Recreating engineered features for the improved model
    test_eng = test.copy()
    test_eng["income_to_loan_ratio"] = test["annual_income"] / (test["loan_amount"] + 1)
    test_eng["delinquency_severity"] = test["num_delinquencies_2yr"] / (test["months_since_last_delinquency"])
    test_eng["revolving_to_income"] = test["total_revolving_balance"] / (test["annual_income"] + 1)
    test_eng["loan_income_x_dti"] = (test["loan_amount"] / (test["annual_income"] + 1)) * test["dti_ratio"]

    X_test_imp = final_scaler.transform(test_eng[models["final_features"]])
    y_pred_imp = final_model.predict_proba(X_test_imp)[:, 1]

    fpr_base, tpr_base, _ = roc_curve(test["default_flag"], y_pred_base)
    fpr_imp, tpr_imp, _ = roc_curve(test["default_flag"], y_pred_imp)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_base, tpr_base, color="gray", label=f"Baseline (AUC = {models['baseline_auc']:.3f})")
    ax.plot(fpr_imp, tpr_imp, color="steelblue", label=f"Improved (AUC = {models['final_auc']:.3f}")
    ax.plot([0,1], [0,1], color="lightgray", linestyle="--", label="Random (AUC = 0.500")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Baseline vs Improved")
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -------------------- Feature Importance --------------------
    st.subheader("Feature Coefficients (Improved Model)")

    coef_df = pd.DataFrame({
        "Feature": models["final_features"],
        "Coefficient:": final_model.coef_[0]
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient:"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["coral" if c > 0 else "steelblue" for c in coef_df["Coefficient:"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient:"], color=colors)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Feature Coefficients for Improved Model")
    ax.axvline(x=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Red = increases default risk and Blue = decreases default risk")

    # -------------------- Model Equation --------------------
    st.subheader("Model Equation")
    st.markdown("**P(default) = 1 / (1 + e(^-n))**")
    st.markdown("where n=")

    equation_df = pd.DataFrame({
        "Feature": ["intercept"] + models["final_features"],
        "Coefficient": [final_model.intercept_[0]] + list(final_model.coef_[0])
    }).round(4)
    st.dataframe(equation_df, use_container_width=True)

with tab3:
    st.header("Business Dashboard")

    # -------------------- Get predictions on test set --------------------
    test_eng = test.copy()
    test_eng["income_to_loan_ratio"] = test["annual_income"] / (test["loan_amount"] + 1)
    test_eng["delinquency_severity"] = test["num_delinquencies_2yr"] / (test["months_since_last_delinquency"] + 1)
    test_eng["revolving_to_income"] = test["total_revolving_balance"] / (test["annual_income"] + 1)
    test_eng["loan_income_x_dti"] = (test["loan_amount"] / (test["annual_income"] + 1)) * test["dti_ratio"]

    final_scaler = models["final_scaler"]
    final_model = models["final_model"]

    X_test_dash = final_scaler.transform(test_eng[models["final_features"]])
    test_eng["pred_default_prob"] = final_model.predict_proba(X_test_dash)[:, 1]

