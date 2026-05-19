import pandas as pd
import numpy as np


def report(df):
    """Print a quick summary of the dataframe."""
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    null_vals = df.isnull().sum()
    null_vals = null_vals[null_vals > 0]
    if len(null_vals) > 0:
        print("\nMissing values:")
        for col, val in null_vals.items():
            print(f"    {col}: {val} missing values")
    else:
        print("\nNo missing values")

    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def normalise_categories(df):
    """Standardise inconsistent categorical encodings."""
    # home_ownership: collapse case variants and synonyms
    df["home_ownership"] = df["home_ownership"].str.lower().replace({
        "renting": "rent",
        "owner": "own",   # NOTE: you had this backwards — see below
    })

    # loan_purpose: lowercase and replace spaces with underscores
    df["loan_purpose"] = (
        df["loan_purpose"]
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def parse_dates(df):
    """Parse mixed-format application_date into datetime."""
    df["application_date"] = pd.to_datetime(df["application_date"], format="mixed", dayfirst=False)
    return df


def get_iqr_bounds(series):
    """Return (lower_bound, upper_bound) using 1.5*IQR rule."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def cap_outliers(df, column):
    """
    Cap outliers at IQR bounds (Winsorisation) rather than removing rows.
    Better for credit modelling — preserves sample size.
    Only applies to numeric columns.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df
    lower, upper = get_iqr_bounds(df[column])
    df[column] = df[column].clip(lower=lower, upper=upper)      # .clip() is a pandas method that caps values
    return df

def find_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"Mean: {df[column].mean()}")
    print(f"Median: {df[column].median()}")
    print(f"Outliers found: {len(outliers)}")

    return outliers, lower_bound, upper_bound