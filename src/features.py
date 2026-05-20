import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_woe_iv(df, feature, target, bins=10):
    """
    Compute Weight of Evidence (WoE) and Information Value (IV)
    for a single feature against a binary target.

    Works for both numeric (auto-binned) and categorical features.
    Returns a summary dataframe and the IV score.
    """
    df = df[[feature, target]].copy()

    # Bin numeric features, leave categories as-is
    if pd.api.types.is_numeric_dtype(df[feature]):
        df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
    else:
        df["bin"] = df[feature]

    # Count events (=1) and non-events (=0) per bin
    grouped = df.groupby("bin", observed=True)[target].agg(
        events="sum",
        total="count"
    ).reset_index()

    grouped["non_events"] = grouped["total"] - grouped["events"]

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    grouped["pct_events"] = grouped["events"] / (total_events + 1e-6)
    grouped["pct_non_events"] = grouped["non_events"] / (total_non_events + 1e-6)

    grouped["woe"] = np.log((grouped["pct_non_events"] + 1e-6) / (grouped["pct_events"] + 1e-6))
    grouped["iv_bin"] = (grouped["pct_non_events"] - grouped["pct_events"]) * grouped["woe"]

    iv = grouped["iv_bin"].sum()
    grouped["feature"] = feature
    grouped["iv"] = iv

    return grouped, iv

def compute_all_iv(df, target, exclude_cols=None):
    """
    Compute IV for every feature in the dataframe.
    Returns a sorted summary dataframe.
    """
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [c for c in df.columns if c not in exclude_cols + [target]]

    results = []
    for col in feature_cols:
        try:
            _, iv = compute_woe_iv(df, col, target)
            results.append({"feature": col, "iv": round(iv, 4)})
        except Exception as e:
            results.append({"feature": col, "iv": None, "error": str(e)})

    summary = pd.DataFrame(results).sort_values("iv", ascending=False)
    return summary

def plot_default_rate_by_bin(df, feature, target="default_flag", bins=10):
    """Plot default rate across bins of a feature."""
    temp = df[[feature, target]].copy()

    if pd.api.types.is_numeric_dtype(temp[feature]):
        temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")
    else:
        temp["bin"] = temp[feature]

    grouped = temp.groupby("bin", observed=True)[target].agg(
        default_rate="mean",
        count="count"
    ).reset_index()

    fig,ax = plt.subplots(figsize=(10,5))
    ax.bar(range(len(grouped)), grouped["default_rate"], color="steelblue")
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped["bin"].astype(str), rotation=45, ha="right")
    ax.set_title(f"Default Rate for {feature}")
    ax.set_ylabel("Default Rate")
    ax.set_xlabel(feature)

    plt.tight_layout()
    plt.show()

    return grouped