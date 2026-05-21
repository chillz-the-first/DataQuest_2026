import sys
import os

# Get the project root directory regardless of where the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["EDA", "Model", "Dashboard"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.write("Coming soon")

with tab2:
    st.header("Model Evaluation")
    st.write("Coming soon")

with tab3:
    st.header("Business Dashboard")
    st.write("Coming soon")

