import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

def woe_encode(train_series, test_series, target, bins=10):
    """
        Compute WoE bins on train, then apply the same mapping to test.
        Returns transformed train and test series.
    """
    temp = pd.DataFrame({"feature": train_series, "target": target})

    if pd.api.types.is_numeric_dtype(train_series):
        temp["bin"], bin_edges = pd.qcut(train_series, q=bins, duplicates="drop", retbins=True)
    else:
        temp["bin"] = train_series
        bin_edges = None

    grouped = temp.groupby("bin", observed=True)["target"].agg(
        events="sum",
        total="count",
    ).reset_index()
    grouped["non_events"] = grouped["total"] - grouped["events"]

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()
    eps = 1e-6

    grouped["woe"] = np.log(
        (grouped["non_events"] / (total_non_events + eps) + eps) /
        (grouped["events"] / (total_events + eps) + eps)
    )

    woe_map = dict(zip(grouped["bin"], grouped["woe"]))
    train_woe = temp["bin"].map(woe_map)

    if bin_edges is not None:
        test_binned = pd.cut(test_series, bins=bin_edges, duplicates="drop")
        test_woe = test_binned.map(woe_map)
    else:
        test_woe = test_series.map(woe_map)

    return train_woe, test_woe