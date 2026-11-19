#!/usr/bin/env python
# coding: utf-8

# # Component C: Data cleaning pipeline
# 
# Reproducible pandas pipeline. Reads `data/raw/immo_data.csv` (never modified), writes
# `data/cleaned/immo_data_clean.csv` and `outputs/cleaning/audit_log.csv` (before/after statistics per step).
# Every step cites the Component B issue ID it answers (see `outputs/profiling/issue_register.csv`).
# Run All from top to bottom. The last cell is a check on postcodes.

# In[ ]:


"""
clean_pipeline.py  -  Component C: reproducible data cleaning pipeline
-----------------------------------------------------------------------
Input : data/raw/immo_data.csv                (never modified)
Output: data/cleaned/immo_data_clean.csv      cleaned dataset
        outputs/cleaning/audit_log.csv        before/after statistics per step

Run from anywhere:
    python src/C_clean_pipeline.py
or inside a notebook:
    open src/C_cleaning.ipynb and Run All

Every step responds to an issue ID from the Component B issue register.
All thresholds are defined once, at the top, so decisions are visible and easy to change.

NOTE for anyone reading the cleaned CSV: load it with dtype={"geo_plz": str},
otherwise pandas turns postcodes back into numbers and drops the leading zeros.
"""



# In[ ]:


import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (work from the terminal, %run, or a notebook in the project)
# ---------------------------------------------------------------------------
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside the notebook: find the folder that contains data/raw
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").is_dir())

RAW = ROOT / "data" / "raw" / "immo_data.csv"
CLEAN = ROOT / "data" / "cleaned" / "immo_data_clean.csv"
AUDIT_FILE = ROOT / "outputs" / "cleaning" / "audit_log.csv"



# In[ ]:


# ---------------------------------------------------------------------------
# Cleaning decisions (one place, documented)
# ---------------------------------------------------------------------------
DATE_MAP = {"Sep18": "2018-09", "May19": "2019-05", "Oct19": "2019-10", "Feb20": "2020-02"}
PLACEHOLDERS = ["no_information"]
ZERO_STATES = {"Sachsen", "Sachsen_Anhalt", "Thüringen", "Brandenburg"}  # postcodes 01xxx-09xxx

MISSING_THRESHOLD = 75.0  # drop columns with more than 75% missing (after placeholders -> NaN)
REDUNDANT_COLUMNS = [
    "geo_bln", "geo_krs",                   # identical to regio1 / regio2 (K3)
    "street",                               # HTML-encoded duplicate of streetPlain (K3)
    "baseRentRange", "livingSpaceRange",    # derived bins; baseRentRange leaks the target (K3)
    "noRoomsRange", "yearConstructedRange",
]

# Row is removed if outside these ranges: target and core feature must be valid (V1)
HARD_RULES = {"baseRent": (50, 20_000), "livingSpace": (10, 1_000)}

# Value is set to missing if outside these ranges; the row is kept (V1, V2)
SOFT_RULES = {
    "totalRent": (50, 25_000), "serviceCharge": (0, 2_000), "heatingCosts": (0, 2_000),
    "noRooms": (1, 20), "floor": (-1, 100), "numberOfFloors": (1, 100),
    "yearConstructed": (1500, 2021), "lastRefurbish": (1500, 2020),
    "noParkSpaces": (0, 20), "thermalChar": (1, 1_000),
}

# Near-duplicate key: the same flat scraped again under a new scoutId (U1)
DEDUP_KEY = ["geo_plz", "livingSpace", "baseRent", "noRooms", "floor", "yearConstructed"]

# Categorical gaps become an explicit "unknown" category (C3)
CATEGORICAL_UNKNOWN = ["condition", "interiorQual", "petsAllowed", "typeOfFlat",
                       "heatingType", "firingTypes", "telekomTvOffer"]

# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
AUDIT = []


def snapshot(df):
    """Statistics recorded before and after every step."""
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "mean_baseRent": round(float(df["baseRent"].mean()), 1),
        "std_baseRent": round(float(df["baseRent"].std()), 1),
    }


def text_columns(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Cleaning steps: each returns (df, values_changed, detail)
# ---------------------------------------------------------------------------
def fix_types(df):
    """F1 - postcodes as valid 5-character text, dates in ISO 8601.

    Only the eastern states use postcodes starting with 0 (01xxx-09xxx). A 4-digit code from
    these states has lost its leading zero and is restored. Any other short code is truncated
    (a digit is missing at the end) and cannot be repaired, so it becomes missing.
    """
    df = df.copy()
    plz = df["geo_plz"]
    restorable = (plz.str.len() == 4) & ~plz.str.startswith("0") & df["regio1"].isin(ZERO_STATES)
    df.loc[restorable, "geo_plz"] = "0" + plz[restorable]
    invalid = (df["geo_plz"].str.len() != 5) | ~df["geo_plz"].str.isdigit()
    df.loc[invalid, "geo_plz"] = np.nan
    df["date"] = df["date"].map(DATE_MAP)
    unmapped = int(df["date"].isna().sum())
    detail = (f"{int(restorable.sum())} postcodes restored with leading zero; "
              f"{int(invalid.sum())} truncated postcodes set to missing; "
              f"date -> ISO 8601 (unmapped: {unmapped})")
    return df, int(restorable.sum()) + int(invalid.sum()) + len(df), detail


def replace_placeholders(df):
    """C2 - disguised missing values ('no_information') become real missing values."""
    df = df.copy()
    cols = text_columns(df)
    mask = df[cols].apply(lambda s: s.astype(str).str.strip().str.lower().isin(PLACEHOLDERS))
    per_col = mask.sum()
    df[cols] = df[cols].mask(mask)
    detail = "; ".join(f"{c}: {n}" for c, n in per_col[per_col > 0].items())
    return df, int(mask.values.sum()), detail


def drop_columns(df):
    """C1 + K3 - near-empty, constant, redundant and derived columns."""
    miss = df.isna().mean() * 100
    sparse = miss[miss > MISSING_THRESHOLD].index.tolist()
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    drop = sorted(set(REDUNDANT_COLUMNS + sparse + constant) & set(df.columns))
    detail = (f"sparse (>{MISSING_THRESHOLD:.0f}% missing): {sorted(sparse)}; "
              f"constant: {sorted(constant)}; redundant/derived: {sorted(set(REDUNDANT_COLUMNS) & set(df.columns))}")
    return df.drop(columns=drop), len(drop), detail


def remove_invalid_rows(df):
    """V1 - rows with an impossible base rent or living space are removed."""
    keep = pd.Series(True, index=df.index)
    parts = []
    for col, (lo, hi) in HARD_RULES.items():
        ok = df[col].between(lo, hi)
        parts.append(f"{col} outside {lo}-{hi}: {int((~ok).sum())}")
        keep &= ok
    return df[keep].copy(), int((~keep).sum()), "; ".join(parts)


def null_implausible_values(df):
    """V1 + V2 - implausible values in other columns become missing; rows are kept."""
    df = df.copy()
    parts, total = [], 0
    for col, (lo, hi) in SOFT_RULES.items():
        if col in df.columns:
            bad = df[col].notna() & ~df[col].between(lo, hi)
            df.loc[bad, col] = np.nan
            total += int(bad.sum())
            if bad.any():
                parts.append(f"{col}: {int(bad.sum())}")
    if {"lastRefurbish", "yearConstructed"} <= set(df.columns):
        bad = df["lastRefurbish"] < df["yearConstructed"]
        df.loc[bad, "lastRefurbish"] = np.nan
        total += int(bad.sum())
        parts.append(f"refurbished before built: {int(bad.sum())}")
    return df, total, "; ".join(parts)


def flag_inconsistencies(df):
    """K1 + K2 - contradictions are flagged, not guessed: we cannot know which value is wrong."""
    df = df.copy()
    calc = df["baseRent"] + df["serviceCharge"].fillna(0) + df.get("heatingCosts", 0).fillna(0)
    consistent = (df["totalRent"] - calc).abs() <= 1
    df["totalRent_consistent"] = consistent.where(df["totalRent"].notna()).astype("boolean")
    df["floor_above_building"] = (df["floor"] > df["numberOfFloors"]).astype("boolean")
    n_rent = int((df["totalRent_consistent"] == False).sum())  # noqa: E712
    n_floor = int(df["floor_above_building"].sum())
    detail = f"totalRent inconsistent: {n_rent}; floor above building: {n_floor} (2 flag columns added)"
    return df, n_rent + n_floor, detail


def remove_near_duplicates(df):
    """U1 - keep only the most recent scrape of each flat."""
    df = df.sort_values("date", kind="stable")
    known = df["streetPlain"].notna()
    dup_known = known & df.duplicated(DEDUP_KEY + ["streetPlain"], keep="last")
    # hidden address: only treat as duplicate if the full listing text is identical too
    has_text = df["description"].notna()
    dup_hidden = ~known & has_text & df.duplicated(DEDUP_KEY + ["regio3", "description"], keep="last")
    drop = dup_known | dup_hidden
    detail = (f"removed {int(dup_known.sum())} with known address, "
              f"{int(dup_hidden.sum())} with hidden address (matched on description); kept latest scrape")
    return df[~drop].sort_index(), int(drop.sum()), detail


def fill_unknown_categories(df):
    """C3 - categorical gaps become an explicit 'unknown' category; numeric gaps are left for modelling."""
    df = df.copy()
    parts, total = [], 0
    for col in CATEGORICAL_UNKNOWN:
        if col in df.columns:
            n = int(df[col].isna().sum())
            df[col] = df[col].fillna("unknown")
            total += n
            parts.append(f"{col}: {n}")
    return df, total, "; ".join(parts)


STEPS = [
    ("1", "F1", "Fix data types and formats", fix_types),
    ("2", "C2", "Convert hidden placeholders to missing", replace_placeholders),
    ("3", "C1, K3", "Drop sparse, constant, redundant and derived columns", drop_columns),
    ("4", "V1", "Remove rows with impossible rent or living space", remove_invalid_rows),
    ("5", "V1, V2", "Set implausible values to missing", null_implausible_values),
    ("6", "K1, K2", "Flag cross-column inconsistencies", flag_inconsistencies),
    ("7", "U1", "Remove near-duplicate listings", remove_near_duplicates),
    ("8", "C3", "Fill categorical gaps with 'unknown'", fill_unknown_categories),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df = pd.read_csv(RAW, low_memory=False, dtype={"geo_plz": str})
    raw_stats = snapshot(df)
    print(f"Loaded {RAW.name}: {raw_stats['rows']:,} rows x {raw_stats['cols']} columns\n")

    for step, issue, action, func in STEPS:
        before = snapshot(df)
        df, changed, detail = func(df)
        after = snapshot(df)
        AUDIT.append({
            "step": step, "issue": issue, "action": action,
            **{f"{k}_before": v for k, v in before.items()},
            **{f"{k}_after": v for k, v in after.items()},
            "values_changed": changed, "detail": detail,
        })
        print(f"Step {step} [{issue}] {action}: rows {before['rows']:,} -> {after['rows']:,}, "
              f"cols {before['cols']} -> {after['cols']}, changed {changed:,}")

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    pd.DataFrame(AUDIT).to_csv(AUDIT_FILE, index=False)

    corr_before = pd.read_csv(RAW, usecols=["baseRent", "livingSpace"]).corr().iloc[0, 1]
    corr_after = df["baseRent"].corr(df["livingSpace"])
    print("\n--- Summary ---")
    print(f"Rows: {raw_stats['rows']:,} -> {len(df):,} | Columns: {raw_stats['cols']} -> {df.shape[1]}")
    print(f"Mean base rent: €{raw_stats['mean_baseRent']:,} -> €{df['baseRent'].mean():,.1f}")
    print(f"Std base rent:  €{raw_stats['std_baseRent']:,} -> €{df['baseRent'].std():,.1f}")
    print(f"corr(baseRent, livingSpace): {corr_before:.2f} -> {corr_after:.2f}")
    print(f"Input  SHA-256: {sha256(RAW)}")
    print(f"Output SHA-256: {sha256(CLEAN)}")
    print(f"\nWrote {CLEAN}\nWrote {AUDIT_FILE}")


if __name__ == "__main__":
    main()



# In[ ]:


raw = pd.read_csv(RAW,
                  usecols=["geo_plz", "regio1", "regio2"], dtype={"geo_plz": str})
print("Postcodes starting with 0:", raw["geo_plz"].str.startswith("0").sum())
print(raw[raw["geo_plz"].str.len() < 5].value_counts().head(25))

