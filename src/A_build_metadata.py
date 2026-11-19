#!/usr/bin/env python
# coding: utf-8

# """
# build_metadata.py  -  Component A: FAIR Assessment & Metadata Design
# -----------------------------------------------------------------------
# Generates two files from the raw dataset:
#  
#   outputs/fair/data_dictionary.csv     human-readable variable documentation
#   outputs/fair/metadata_record.jsonld  schema.org JSON-LD record (DataCite-aligned)
#  
# Run from the project root:
#     python src/build_metadata.py
#  
# Descriptions and units are written by hand (VARIABLES below).
# Types, missing %, ranges and category values are computed from the data,
# so the documentation always matches the actual file.
# """

# In[1]:


import json
from pathlib import Path

import pandas as pd

# Project root = the folder that contains data/raw, wherever this runs from (VS Code, Jupyter, terminal)
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").is_dir())
RAW = ROOT / "data" / "raw" / "immo_data.csv"
OUT = ROOT / "outputs" / "fair"
OUT.mkdir(parents=True, exist_ok=True)

ZENODO_DOI = "10.5281/zenodo.23494643"         # version DOI (this exact version)
ZENODO_CONCEPT_DOI = "10.5281/zenodo.23494642"  # concept DOI (all versions)
KAGGLE_URL = "https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany"
FILE_URL = "https://zenodo.org/api/records/23494643/files/immo_data.csv/content"



# # ---------------------------------------------------------------------------
# # 1. Hand-written variable documentation
# #    name: (group, description, unit, intended_type, notes)
# # ---------------------------------------------------------------------------

# In[2]:


VARIABLES = {
    # --- identifiers & time ---
    "scoutId": ("identifier", "Unique listing ID on ImmoScout24.", None, "string", ""),
    "date": ("time", "Scrape period in which the listing was collected (Sep18, May19, Oct19, Feb20).", None, "date (ISO 8601 recommended)", "Non-standard format, e.g. 'Feb20'; should be converted to ISO 8601 (2020-02)."),
    # --- location ---
    "regio1": ("location", "Federal state (Bundesland), words joined by underscores, e.g. Nordrhein_Westfalen.", None, "category", "Duplicates geo_bln."),
    "geo_bln": ("location", "Federal state (Bundesland); same information as regio1.", None, "category", "Redundant with regio1."),
    "regio2": ("location", "District or district-free city (Kreis / kreisfreie Stadt).", None, "category", "Duplicates geo_krs."),
    "geo_krs": ("location", "District (Kreis); same information as regio2.", None, "category", "Redundant with regio2."),
    "regio3": ("location", "Town or city quarter (Ortsteil / Stadtteil).", None, "category", ""),
    "geo_plz": ("location", "German postal code (Postleitzahl).", None, "string (5 characters)", "Read as integer, so leading zeros are lost (e.g. 01067 -> 1067)."),
    "street": ("location", "Street name; may contain HTML-encoded characters.", None, "string", "'no_information' is used as a placeholder when the address is hidden (disguised missing value)."),
    "streetPlain": ("location", "Street name in plain text; empty when the landlord hides the address.", None, "string", ""),
    "houseNumber": ("location", "House number; empty when the landlord hides the address.", None, "string", ""),
    # --- price & costs ---
    "baseRent": ("price", "Monthly base rent without utilities (Kaltmiete).", "EUR per month", "float", "Contains 0 and placeholder values such as 9,999,999."),
    "totalRent": ("price", "Total monthly rent (Warmmiete): base rent plus service charge and heating costs, where given.", "EUR per month", "float", "Contains 0 and extreme outliers."),
    "serviceCharge": ("price", "Monthly ancillary costs (Nebenkosten) as stated by the landlord.", "EUR per month", "float", "Whether heating is included is not documented by the source."),
    "heatingCosts": ("price", "Monthly heating costs, if stated separately.", "EUR per month", "float", ""),
    "electricityBasePrice": ("price", "Electricity base price shown on the listing page.", "EUR (billing period undocumented, likely per year)", "float", "Nearly constant; appears to be a default tariff, not listing-specific."),
    "electricityKwhPrice": ("price", "Electricity price per kWh shown on the listing page.", "EUR per kWh", "float", "Nearly constant; appears to be a default tariff."),
    "pricetrend": ("price", "Price trend indicator calculated by ImmoScout24.", "undocumented (possibly % change)", "float", "Meaning and unit are not documented by the source."),
    "baseRentRange": ("price", "Binned base rent (code 1-9) derived from baseRent.", None, "ordinal category", "Bin boundaries not documented by the source; see inferred ranges."),
    # --- size & layout ---
    "livingSpace": ("size", "Living area of the flat.", "square metre (m2)", "float", "Contains 0 and placeholder values such as 111,111."),
    "livingSpaceRange": ("size", "Binned living space (code 1-7) derived from livingSpace.", None, "ordinal category", "Bin boundaries not documented by the source; see inferred ranges."),
    "noRooms": ("size", "Number of rooms (half rooms possible, e.g. 2.5).", "rooms", "float", "Contains placeholder values such as 999.99."),
    "noRoomsRange": ("size", "Binned number of rooms (code 1-5) derived from noRooms.", None, "ordinal category", "Bin boundaries not documented by the source; see inferred ranges."),
    "floor": ("size", "Floor of the flat (0 = ground floor, negative = basement).", "floor number", "integer", "Contains placeholder values such as 999."),
    "numberOfFloors": ("size", "Total number of floors in the building.", "floors", "integer", "Contains placeholder values such as 999."),
    # --- building & condition ---
    "yearConstructed": ("building", "Year the building was constructed.", "year", "integer", "Contains implausible values (e.g. 1000, 2090)."),
    "yearConstructedRange": ("building", "Binned construction year (code 1-9) derived from yearConstructed.", None, "ordinal category", "Bin boundaries not documented by the source; see inferred ranges."),
    "lastRefurbish": ("building", "Year of the most recent refurbishment.", "year", "integer", "Contains typos (e.g. 1015, 2919)."),
    "newlyConst": ("building", "True if the building is newly constructed.", None, "boolean", ""),
    "condition": ("building", "Condition of the flat, e.g. well_kept, refurbished, first_time_use.", None, "category", ""),
    "interiorQual": ("building", "Quality of the interior, e.g. simple, normal, sophisticated, luxury.", None, "ordinal category", ""),
    "typeOfFlat": ("building", "Type of flat, e.g. apartment, roof_storey, ground_floor, maisonette.", None, "category", ""),
    "thermalChar": ("building", "Energy consumption or demand value from the energy performance certificate.", "kWh per m2 per year", "float", "Some values look like years typed into this field (e.g. 1996)."),
    "energyEfficiencyClass": ("building", "Energy efficiency class from the energy performance certificate (A_PLUS to H).", None, "ordinal category", ""),
    "heatingType": ("building", "Type of heating system, e.g. central_heating, district_heating.", None, "category", ""),
    "firingTypes": ("building", "Energy source(s) used for heating, e.g. gas, oil; can combine several values.", None, "category (multi-valued)", ""),
    # --- features ---
    "balcony": ("features", "True if the flat has a balcony.", None, "boolean", ""),
    "hasKitchen": ("features", "True if a fitted kitchen is included.", None, "boolean", ""),
    "cellar": ("features", "True if a cellar room is included.", None, "boolean", ""),
    "lift": ("features", "True if the building has a lift.", None, "boolean", ""),
    "garden": ("features", "True if a garden can be used.", None, "boolean", ""),
    "petsAllowed": ("features", "Whether pets are allowed (e.g. yes, no, negotiable).", None, "category", ""),
    "noParkSpaces": ("features", "Number of parking spaces offered with the flat.", "count", "integer", "Contains implausible values (max 2,241)."),
    # --- internet ---
    "telekomTvOffer": ("internet", "Availability and terms of Deutsche Telekom TV at the address.", None, "category", ""),
    "telekomUploadSpeed": ("internet", "Maximum Deutsche Telekom upload speed at the address.", "Mbit/s", "float", ""),
    "telekomHybridUploadSpeed": ("internet", "Upload speed of the Telekom hybrid (DSL + LTE) offer.", "Mbit/s", "float", "Constant value (10) wherever present; carries no information."),
    # --- listing ---
    "picturecount": ("listing", "Number of photos attached to the listing.", "count", "integer", ""),
    "description": ("listing", "Free-text description of the offer written by the landlord (mostly German).", None, "text", ""),
    "facilities": ("listing", "Free-text description of facilities and fittings (mostly German).", None, "text", ""),
}

# Binned columns and the column they were derived from
RANGE_SOURCES = {
    "baseRentRange": "baseRent",
    "livingSpaceRange": "livingSpace",
    "noRoomsRange": "noRooms",
    "yearConstructedRange": "yearConstructed",
}

# UN/CEFACT unit codes for schema.org (only where an exact code exists)
UNIT_CODES = {"square metre (m2)": "MTK"}

VARIABLE_URIS = {
    "livingSpace": "https://schema.org/floorSize",
    "noRooms": "https://schema.org/numberOfRooms",
    "petsAllowed": "https://schema.org/petsAllowed",
    "yearConstructed": "https://schema.org/yearBuilt",
    "floor": "https://schema.org/floorLevel",
    "geo_plz": "https://schema.org/postalCode",
    "regio1": "https://schema.org/addressRegion",
    "regio3": "https://schema.org/addressLocality",
    "streetPlain": "https://schema.org/streetAddress",
    "scoutId": "https://schema.org/identifier",
}



# # ---------------------------------------------------------------------------
# # 2. Compute facts from the data
# # ---------------------------------------------------------------------------

# In[3]:


def infer_bins(df, range_col, source_col):
    """Approximate value range per bin code (1st-99th percentile, robust to outliers)."""
    parts = []
    for code, grp in df.groupby(range_col)[source_col]:
        if grp.notna().any():
            lo, hi = grp.quantile(0.01), grp.quantile(0.99)
            parts.append(f"{int(code)}: ~{lo:g}-{hi:g}")
    return "; ".join(parts)


def profile_column(df, col):
    s = df[col]
    row = {
        "pandas_dtype": str(s.dtype),
        "missing_pct": round(s.isna().mean() * 100, 1),
        "hidden_missing": int(s.astype(str).str.strip().str.lower().isin(["no_information", "none", ""]).sum()),
        "n_unique": int(s.nunique()),
        "min": "",
        "max": "",
        "values_or_examples": "",
    }
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        row["min"], row["max"] = s.min(), s.max()
    if row["n_unique"] <= 30:
        row["values_or_examples"] = " | ".join(map(str, s.dropna().value_counts().index[:30]))
    elif not pd.api.types.is_numeric_dtype(s):
        row["values_or_examples"] = " | ".join(map(str, s.dropna().value_counts().index[:5])) + " | ..."
    return row


# 
# # ---------------------------------------------------------------------------
# # 3. Build the data dictionary
# # ---------------------------------------------------------------------------

# In[4]:


def build_dictionary(df):
    unknown = set(df.columns) - set(VARIABLES)
    missing = set(VARIABLES) - set(df.columns)
    if unknown:
        print("WARNING - columns without documentation:", sorted(unknown))
    if missing:
        print("WARNING - documented columns not in data:", sorted(missing))

    rows = []
    for col in df.columns:
        group, desc, unit, intended, notes = VARIABLES.get(col, ("undocumented", "", None, "", ""))
        row = {"variable": col, "group": group, "description": desc, "unit": unit or "",
               "intended_type": intended}
        row.update(profile_column(df, col))
        if col in RANGE_SOURCES:
            notes = f"{notes} Inferred ranges of {RANGE_SOURCES[col]}: {infer_bins(df, col, RANGE_SOURCES[col])}"
        row["quality_notes"] = notes.strip()
        rows.append(row)
    return pd.DataFrame(rows)


# # ---------------------------------------------------------------------------
# # 4. Build the schema.org JSON-LD record (DataCite-aligned)
# # ---------------------------------------------------------------------------

# In[5]:


def build_jsonld(dictionary, n_rows):
    variables = []
    for _, r in dictionary.iterrows():
        v = {"@type": "PropertyValue", "name": r["variable"], "description": r["description"]}
        if r["variable"] in VARIABLE_URIS:
            v["propertyID"] = VARIABLE_URIS[r["variable"]]
        if r["unit"]:
            v["unitText"] = r["unit"]
            if r["unit"] in UNIT_CODES:
                v["unitCode"] = UNIT_CODES[r["unit"]]
        variables.append(v)

    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "dcterms": "http://purl.org/dc/terms/",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "Dataset",
        "@id": f"https://doi.org/{ZENODO_DOI}",
        "identifier": [
            {"@type": "PropertyValue", "propertyID": "DOI", "value": ZENODO_DOI,
             "url": f"https://doi.org/{ZENODO_DOI}", "description": "Version DOI"},
            {"@type": "PropertyValue", "propertyID": "DOI", "value": ZENODO_CONCEPT_DOI,
             "url": f"https://doi.org/{ZENODO_CONCEPT_DOI}", "description": "Concept DOI (all versions)"},
        ],
        "name": "Apartment rental offers in Germany",
        "alternateName": "immo_data",
        "description": (
            f"{n_rows:,} apartment rental listings scraped from ImmoScout24, Germany's largest real estate "
            "platform, at four points in time between September 2018 and February 2020. Each row is one "
            "listing with rent and cost figures, size, location (federal state to postcode), building "
            "characteristics, energy data, features and free-text descriptions (German). Contains rental "
            "offers only, no sales listings. A full data dictionary is provided as data_dictionary.csv."
        ),
        "keywords": ["rental market", "apartment rentals", "housing", "rent prices", "real estate",
                     "Germany", "ImmoScout24", "web scraping"],
        "about": [
            {"@type": "DefinedTerm", "name": "housing", "termCode": "1690",
             "@id": "http://eurovoc.europa.eu/1690", "inDefinedTermSet": "http://eurovoc.europa.eu/100141"},
            {"@type": "DefinedTerm", "name": "rent", "termCode": "1687",
             "@id": "http://eurovoc.europa.eu/1687", "inDefinedTermSet": "http://eurovoc.europa.eu/100141"},
            {"@type": "DefinedTerm", "name": "housing",
             "@id": "http://www.eionet.europa.eu/gemet/concept/4022",
             "inDefinedTermSet": "http://www.eionet.europa.eu/gemet/"},
        ],
        "inLanguage": ["en", "de"],
        "creator": [{"@type": "Person", "name": "CorrieBar", "url": "https://www.kaggle.com/corrieaar"}],
        "contributor": [
            {"@type": "Role", "roleName": "DataCurator",
             "contributor": {"@type": "Person", "name": name,
                             "affiliation": {"@type": "Organization", "name": "SRH University, Campus Munich"}}}
                        for name in ["Pratham Mandya Badavaraju", "Phaneendra Reddy", "Gagan"]
    ],
        "publisher": {"@type": "Organization", "name": "Zenodo", "url": "https://zenodo.org"},
        "datePublished": "2026-10-20",
        "temporalCoverage": "2018-09-22/2020-02",
        "spatialCoverage": {
            "@type": "Place", "name": "Germany",
            "sameAs": ["https://sws.geonames.org/2921044/", "https://www.wikidata.org/entity/Q183"],
        },
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "isAccessibleForFree": True,
        "dcterms:accessRights": {"@id": "http://purl.org/coar/access_right/c_abf2"},
        "conditionsOfAccess": "Open access, no registration required.",
        "isBasedOn": KAGGLE_URL,
        "prov:wasDerivedFrom": {"@id": KAGGLE_URL},
        "prov:hadPrimarySource": {"@id": "https://www.immobilienscout24.de"},
        "measurementTechnique": (
            "Web scraping of all rental listings available on ImmoScout24 at four points in time "
            "(2018-09-22, 2019-05-10, 2019-10-08 and February 2020). Minimal processing by the original author."
        ),
        "url": "https://zenodo.org/records/23494643",
        "distribution": [
            {"@type": "DataDownload", "name": "immo_data.csv", "encodingFormat": "text/csv",
             "contentUrl": FILE_URL, "contentSize": "284924672 bytes (271.73 MiB)",
             "identifier": {"@type": "PropertyValue", "propertyID": "SHA-256",
                            "value": "895b934bd08bbaa75399633b4b48b042324d57cd068a5f383e0271725a1994e9"}},
            {"@type": "DataDownload", "name": "data_dictionary.csv", "encodingFormat": "text/csv",
             "description": "Variable-level documentation for all columns."},
        ],
        "variableMeasured": variables,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW, low_memory=False)

    dictionary = build_dictionary(df)
    dictionary.to_csv(OUT / "data_dictionary.csv", index=False)

    record = build_jsonld(dictionary, len(df))
    with open(OUT / "metadata_record.jsonld", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)

    print(f"Wrote {OUT / 'data_dictionary.csv'} ({len(dictionary)} variables)")
    print(f"Wrote {OUT / 'metadata_record.jsonld'}")


if __name__ == "__main__":
    main()




# In[6]:


import hashlib
h = hashlib.sha256()
with open(RAW, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print("SHA-256:", h.hexdigest())


# In[ ]:




