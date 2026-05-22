# AI Usage Log — DataQuest 2026 Credit Modelling Project

## Overview

This log reflects honestly on how AI assistance was used throughout this project, what I did independently, and where AI was wrong or unhelpful. The goal is an accurate account, not a flattering one.

---

## Full Chronology

### Day 1 — Setup and Data Cleaning

I chose PyCharm as my development environment and set it up myself. All of the cleaning code was written by me; AI reviewed it and flagged issues rather than producing the code.

One moment worth noting: after applying Winsorisation I ran `report()` and noticed 14 duplicate rows still showing in the output. I caught that capping can create new duplicates and added a second `drop_duplicates()` call myself — AI had not flagged this. I also questioned whether `df.shape[1]` or `len(df)` was the right call in a particular context; that was a good instinct and I was right to ask. I also asked about the trade-off between loops and vectorised operations, understood the explanation, and applied it in the code.

---

### EDA Notebook

I ran the IV scoring code myself and shared the results. I looked at the six bar charts and described what I saw in each one; AI helped me sharpen the interpretation in a few cases (being more precise about the shape of relationships rather than just noting direction). I ran the correlation matrix myself and independently identified that `age` and `months_since_oldest_account` were nearly identical (r = 0.95) before AI raised it. The EDA findings summary document was written entirely by me — AI gave feedback on wording and flagged a few corrections, but every sentence was mine.

---

### Bivariate Analysis

I wrote the code for all three feature pairs myself. I correctly identified the inversely proportional relationships in the data. On the age × delinquencies pair, AI pointed out that the table showed more nuance than I had initially described — a fair and useful observation.

---

### Streamlit App

I built every section of the dashboard myself. Three debugging moments are worth recording accurately:

- I caught the heatmap axis ordering problem and flagged it myself.
- I debugged the `threshold` vs `selected_threshold` variable name conflict independently.
- I noticed the `ModuleNotFoundError` and we resolved the path issue together.

---

### Modelling

I set up the baseline logistic regression myself. On the first improvement attempt, AI suggested applying WoE encoding to all features. I ran the model, saw the AUC drop from 0.789 to 0.703, and immediately identified that the approach had made things worse. I investigated why and revised the strategy — this was my judgement call, not AI's.

The most significant independent catch of the project came during modelling: I noticed that `months_since_oldest_account` had appeared in the model output despite having decided to drop it earlier in the EDA phase. AI had not flagged this inconsistency. I caught it, removed the feature, and observed the age coefficient strengthen from -0.33 to -0.45 as a result — consistent with the expected behaviour when a correlated duplicate is removed.

I then tested features one at a time and made my own decisions about which to retain.

---

### Written Work

All three documents — the research writeup, the EDA findings summary, and the modelling summary — were written entirely by me. AI reviewed drafts, flagged errors, and suggested corrections to wording, but produced none of the original content.

---

## Moments That Were Entirely Mine

These are the decisions and catches that reflect independent judgement, not AI assistance:

1. **Catching the 14 duplicate rows after Winsorisation** — I noticed this from `report()` output and added the second deduplication step myself.
2. **Identifying `months_since_oldest_account` in the final model** — I had decided to drop it during EDA; I caught the inconsistency in the modelling output without any prompt from AI.
3. **Identifying the heatmap axis ordering bug** — I flagged this myself while reviewing the Streamlit app.
4. **Recognising the first improved model was worse** — I saw the AUC drop immediately and investigated the cause; AI's suggestion had been wrong.
5. **Writing all three documents** — the research section, EDA summary, and modelling summary were written independently.

---

## Moments Where AI Was Wrong or Unhelpful

An honest log requires recording where AI failed:

1. **WoE-encode-everything suggestion** — AI's first recommendation for model improvement was to apply WoE transformation to all features. This reduced AUC from 0.789 to 0.703 — a meaningful degradation. I caught it by running the code and checking the output. The correct approach, applying WoE only to `annual_income` to fix the specific non-monotonicity, came from iterative testing rather than from the initial AI suggestion.

2. **`optbinning` library incompatibility** — AI suggested using `optbinning` for IV computation. The library failed due to a Python version incompatibility. We had to write the IV calculation from scratch instead, which cost time that a better-informed suggestion would have avoided.

---

## Summary Assessment

AI was most useful as a reviewer and sounding board — catching issues in code I had already written, helping sharpen written interpretations, and explaining concepts I then applied independently. It was least useful, and actively counterproductive, when suggesting implementation strategies (WoE encoding, library choice) without sufficient knowledge of the environment constraints. The substantive intellectual work — the code, the analysis, the writing, and the critical debugging catches — was mine.
