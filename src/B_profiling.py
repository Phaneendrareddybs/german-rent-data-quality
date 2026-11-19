#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% [markdown]
# # Component B: Data quality and profiling
# 
# **Student B: Phaneendra.** Profiles `data/raw/immo_data.csv` across five quality dimensions (completeness,
# uniqueness, consistency, validity, timeliness), produces three charts and an issue table with likely causes,
# downstream impact and suggested cleaning. Outputs go to `outputs/profiling/`.

# %%
from pathlib import Path
import pandas as pd

# Search the notebook's working directory and its parents for the project CSV.
relative_csv = Path("data/raw/immo_data.csv")
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / relative_csv).is_file()), None)
if ROOT is None:
    raise FileNotFoundError(
        f"Could not find {relative_csv} from {Path.cwd()}. "
        "Run this notebook from the project folder or one of its subfolders."
    )
RAW = ROOT / relative_csv
OUT = ROOT / "outputs" / "profiling"
OUT.mkdir(parents=True, exist_ok=True)
print(RAW.exists(), OUT)

# %% [markdown]
# ## Step 1: Load and look
# 
# Read the CSV with pandas (`low_memory=False`). Print shape and the count of each dtype.

# %%
df = pd.read_csv(RAW, low_memory=False)
print(df.shape)
print(df.dtypes.value_counts())

# %% [markdown]
# ## Step 2: Completeness
# 
# Missing count and percentage per column, sorted. Save as `missing_profile.csv` in OUT.

# %%
missing = df.isna().sum()
pct = (missing / len(df) * 100).round(2)

missing_profile = pd.DataFrame({"missing": missing, "pct": pct})
missing_profile = missing_profile.sort_values("pct", ascending=False)

missing_profile.to_csv(OUT / "missing_profile.csv")
missing_profile.head(15)

# %%
cols = ["electricityBasePrice", "telekomHybridUploadSpeed", "energyEfficiencyClass"]
df[cols].isna().groupby(df["date"]).mean().round(3) * 100

# %% [markdown]
# ## Step 3: Hidden missingness
# 
# Find placeholder text that `isna()` cannot see, for example `no_information` in `street`. Count it.

# %%
print((df["street"] == "no_information").sum())
print(((df["street"] == "no_information") & df["streetPlain"].isna()).sum())

# %% [markdown]
# ## Step 4: Uniqueness: exact
# 
# Count exact duplicate rows and duplicate `scoutId` values.

# %%
print("exact duplicate rows:", df.duplicated().sum())
print("duplicate scoutId:", df["scoutId"].duplicated().sum())

# %% [markdown]
# ## Step 5: Uniqueness: repeat listings

# %%
flat_key = ["geo_plz", "street", "baseRent", "livingSpace", "noRooms", "floor"]

repeats = df.duplicated(subset=flat_key)
print("repeat listings:", repeats.sum())

# how many of those groups appear in more than one snapshot?
groups = df[df.duplicated(subset=flat_key, keep=False)]
print("groups across snapshots:", (groups.groupby(flat_key)["date"].nunique() > 1).sum())

# %% [markdown]
# ## Step 6: Format consistency
# 
# Postcodes: convert `geo_plz` to text and count values shorter than 5 digits.

# %%
plz_text = df["geo_plz"].astype(str)
print("postcodes shorter than 5 digits:", (plz_text.str.len() < 5).sum())
plz_text[plz_text.str.len() < 5].head()

# %% [markdown]
# ## Step 7: Encoding consistency
# 
# Count street names that contain HTML codes such as `&szlig;`.

# %%
has_code = df["street"].str.contains("&[a-z]+;", regex=True, na=False)
print("streets with HTML codes:", has_code.sum())
df.loc[has_code, "street"].head()

# %% [markdown]
# ## Step 8: Validity
# 
# Nonpositive rent, living space and rooms; construction years in the future or implausibly old. Write down your thresholds.

# %%
MIN_YEAR = 1500   # your choice: oldest believable building
MAX_YEAR = 2020   # your choice: last snapshot is Feb 2020

print("baseRent <= 0:", (df["baseRent"] <= 0).sum())
print("livingSpace <= 0:", (df["livingSpace"] <= 0).sum())
print("noRooms <= 0:", (df["noRooms"] <= 0).sum())
print("year before MIN:", (df["yearConstructed"] < MIN_YEAR).sum())
print("year after MAX:", (df["yearConstructed"] > MAX_YEAR).sum())
df[["baseRent", "totalRent", "livingSpace", "noRooms", "yearConstructed"]].describe().round(1)

# %% [markdown]
# ## Step 9: Cross-field rules
# 
# Rows where floor > numberOfFloors, and where totalRent < baseRent.

# %%
print("floor above building height:", (df["floor"] > df["numberOfFloors"]).sum())
print("totalRent below baseRent:", (df["totalRent"] < df["baseRent"]).sum())

# %% [markdown]
# ## Step 10: Timeliness
# 
# Rows per snapshot (`date`). How old is the newest snapshot today?

# %%
print(df["date"].value_counts())
print("newest snapshot: Feb 2020, about", 2026 - 2020, "years old")

# %% [markdown]
# ## Step 11: Chart 1: completeness
# 
# Bar chart of the top 15 columns by missing percentage.

# %%
import matplotlib.pyplot as plt

missing_profile["pct"].head(15).sort_values().plot(kind="barh", figsize=(8, 6), color="#2a7ab0")
plt.title("Top 15 columns by missing values")
plt.xlabel("Missing (%)")
plt.tight_layout()
plt.savefig(OUT / "chart1_missing.png", dpi=200)
plt.show()

# %% [markdown]
# ## Step 12: Chart 2: distribution
# 
# Histogram of rent per m2 (baseRent / livingSpace). Drop infinite values, log scale on x.

# %%
import numpy as np

per_m2 = (df["baseRent"] / df["livingSpace"]).replace([np.inf, -np.inf], np.nan).dropna()
per_m2 = per_m2[per_m2 > 0]

plt.figure(figsize=(8, 4))
plt.hist(per_m2, bins=np.logspace(np.log10(per_m2.min()), np.log10(per_m2.max()), 80), color="#2a7ab0")
plt.xscale("log")
plt.title("Base rent per m² (log scale)")
plt.xlabel("EUR per m²")
plt.ylabel("Listings")
plt.tight_layout()
plt.savefig(OUT / "chart2_rent_per_m2_log.png", dpi=200)
plt.show()
print(per_m2.describe(percentiles=[0.01, 0.5, 0.99]).round(2))

# %%
low, high = 1, 100   # believable EUR per m²
outside = ((per_m2 < low) | (per_m2 > high)).sum()

plt.figure(figsize=(8, 4))
plt.hist(per_m2[(per_m2 >= low) & (per_m2 <= high)], bins=80, color="#2a7ab0")
plt.title(f"Base rent per m² ({outside} listings outside {low} to {high} EUR not shown)")
plt.xlabel("EUR per m²")
plt.ylabel("Listings")
plt.tight_layout()
plt.savefig(OUT / "chart2_rent_per_m2.png", dpi=200)
plt.show()
print(per_m2.describe(percentiles=[0.01, 0.5, 0.99]).round(2))

# %% [markdown]
# ## Step 13: Issue table
# 
# Build the issue table from the profiling checks, add notes, and save it before plotting.
# Percentages use all rows in the CSV as the denominator. Repeat listings count rows after the first occurrence of each matching key. The rent-per-m² range check uses positive, finite ratios, as in Step 12.

# %%
issue_counts = [
    ("Completeness", "Electricity price only collected in Sep18", missing["electricityBasePrice"]),
    ("Completeness", "Placeholder 'no_information' for street", df["street"].eq("no_information").sum()),
    ("Completeness", "Total rent missing", missing["totalRent"]),
    ("Uniqueness", "Same flat listed more than once", repeats.sum()),
    ("Consistency", "Postcode lost its leading zero", (plz_text.str.len() < 5).sum()),
    ("Consistency", "HTML codes in street names", has_code.sum()),
    ("Validity", "Living space zero or negative", df["livingSpace"].le(0).sum()),
    ("Validity", "Construction year impossible",
     (df["yearConstructed"].lt(MIN_YEAR) | df["yearConstructed"].gt(MAX_YEAR)).sum()),
    ("Consistency (cross-field)", "Floor above building height", (df["floor"] > df["numberOfFloors"]).sum()),
    ("Consistency (cross-field)", "Total rent below base rent", (df["totalRent"] < df["baseRent"]).sum()),
    ("Validity", "Rent per m2 outside believable range", outside),
    ("Timeliness", "Newest data is from Feb 2020", len(df)),
]
issues = pd.DataFrame(issue_counts, columns=["dimension", "issue", "affected_rows"])
issues["pct"] = (issues["affected_rows"] / len(df) * 100).round(2)

notes = {
    0:  ["Field only collected in the Sep18 snapshot", "Models can only learn from Sep18 listings", "Drop the column, too sparse to fill"],
    1:  ["Site writes text instead of leaving street empty", "isna() misses it; fake street in groupings", "Replace 'no_information' with a real missing value"],
    2:  ["Optional field, many sellers only give base rent", "Total cost comparisons lose 15% of listings", "Keep missing; do not fill from base rent"],
    3:  ["Same flat re-posted in later snapshots or by two agents", "Repeated flats over-weight their district median", "Keep one copy per flat, the newest snapshot"],
    4:  ["Postcode read as a number, leading zero dropped", "Eastern postcodes fail to match maps or other data", "Read geo_plz as text and pad to 5 digits"],
    5:  ["Scraper saved HTML codes instead of real letters", "Same street appears under two spellings", "Decode HTML codes with Python's html module"],
    6:  ["Form default or typing error", "Rent per m2 becomes infinite or meaningless", "Set to missing and flag, keep the listing"],
    7:  ["Typing errors or placeholder years", "Building age analysis gets distorted", "Set years outside 1500 to 2020 to missing"],
    8:  ["Typo, or ground floor counted differently", "Floor-based features become unreliable", "Flag only; definition is ambiguous"],
    9:  ["Fields swapped or utilities entered as negative", "Total cost per m2 is understated", "Flag and exclude from total rent analysis"],
    10: ["Placeholder values like 9,999,999 or tiny areas", "Means explode; medians mostly survive", "Flag outside 1 to 100 EUR per m2, use medians"],
    11: ["Data scraped 2018 to 2020, before COVID and inflation", "Rents too low for today's market", "Label results as 2018 to 2020; do not use as current rents"],
}
for i, (cause, impact, cleaning) in notes.items():
    issues.loc[i, ["likely_cause", "impact", "suggested_cleaning"]] = [cause, impact, cleaning]

issues.to_csv(OUT / "quality_issues.csv", index=False)
issues

# %% [markdown]
# ## Step 14: Chart 3: issues by dimension
# 
# Plot percentages and affected-row counts from the issue table. Both views omit timeliness because it applies to the entire historical dataset.

# %%
plot_df = issues[issues["dimension"] != "Timeliness"].sort_values("pct")
dims = plot_df["dimension"].unique()
colors = dict(zip(dims, plt.cm.tab10.colors))

plt.figure(figsize=(9, 5))
plt.barh(plot_df["issue"], plot_df["pct"], color=plot_df["dimension"].map(colors))
for d, c in colors.items():
    plt.bar(0, 0, color=c, label=d)
plt.legend(title="Dimension", fontsize=8)
plt.title("Share of listings affected by each quality issue")
plt.xlabel("Listings affected (%)")
plt.tight_layout()
plt.savefig(OUT / "chart3_issues_pct.png", dpi=200)
plt.show()

# %%
plot_df = issues[issues["dimension"] != "Timeliness"].sort_values("affected_rows").copy()
plot_df["main_dim"] = plot_df["dimension"].str.split(" ").str[0]
colors = dict(zip(plot_df["main_dim"].unique(), plt.cm.tab10.colors))

plt.figure(figsize=(10, 5))
bars = plt.barh(plot_df["issue"], plot_df["affected_rows"], color=plot_df["main_dim"].map(colors))
plt.bar_label(bars, labels=[f"{v:,}" for v in plot_df["affected_rows"]], padding=3, fontsize=8)
count_scale = "symlog" if plot_df["affected_rows"].eq(0).any() else "log"
plt.xscale(count_scale)
plt.margins(x=0.15)
for d, c in colors.items():
    plt.bar(0, 0, color=c, label=d)
plt.legend(title="Dimension", loc="lower right", fontsize=8)
plt.title(f"Listings affected by each quality issue (timeliness affects all {len(df):,})")
plt.xlabel(f"Listings affected ({count_scale} scale)")
plt.tight_layout()
plt.savefig(OUT / "chart3_issues.png", dpi=200)
plt.show()

# %% [markdown]
# ## Step 15: Issue IDs used by the cleaning pipeline (link to Component C)
# 
# Component C cites an issue ID in every audit-log row. This table gives each B issue that ID, so every cleaning
# step can be traced back to the problem it fixes. Issues with no cleaning step are kept on purpose (stated in
# `c_handling`).

# %%
issue_ids = {
    "Electricity price only collected in Sep18": ("C1", "Column dropped (more than 75% missing)"),
    "Placeholder 'no_information' for street": ("C2", "Placeholder turned into a real missing value"),
    "Total rent missing": ("C4", "Kept as missing on purpose; not filled from base rent"),
    "Same flat listed more than once": ("U1", "Near-duplicates removed, newest snapshot kept"),
    "Postcode lost its leading zero": ("F1", "Postcode read as text, leading zero restored"),
    "HTML codes in street names": ("K3", "HTML-coded street column dropped; clean streetPlain kept"),
    "Living space zero or negative": ("V1", "Rows with living space outside 10 to 1,000 m2 removed"),
    "Construction year impossible": ("V2", "Years outside 1500 to 2021 set to missing"),
    "Floor above building height": ("K2", "Flag column floor_above_building added"),
    "Total rent below base rent": ("K1", "Flag column totalRent_consistent added"),
    "Rent per m2 outside believable range": ("V1", "Rows with impossible base rent or living space removed"),
    "Newest data is from Feb 2020": ("T1", "No cleaning possible; results labelled 2018 to 2020"),
}
register = issues.copy()
register.insert(0, "issue_id", register["issue"].map(lambda x: issue_ids.get(x, ("?", ""))[0]))
register["c_handling"] = register["issue"].map(lambda x: issue_ids.get(x, ("?", ""))[1])
print("issues without an ID:", (register["issue_id"] == "?").sum())
register.to_csv(OUT / "issue_register.csv", index=False)
register[["issue_id", "issue", "dimension", "affected_rows", "c_handling"]]
