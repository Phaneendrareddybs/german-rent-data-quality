# How Does Data Quality Affect Fair Rent Estimates?

A FAIR-assessed, profiled and cleaned dataset of German rental listings, with a NetworkX graph model and an
interactive Streamlit dashboard. SRH _Data Management_ group project, Winter 2026/27.

| Component                       | Student                                 | Notebook                                                |
| ------------------------------- | --------------------------------------- | ------------------------------------------------------- |
| A: FAIR assessment and metadata | Pratham Mandya Basavaraju               | `src/A_build_metadata.ipynb`                            |
| B: Data quality and profiling   | Phaneendra Reddy Bonthu Srinivasa Reddy | `src/B_profiling.ipynb`                                 |
| C: Cleaning pipeline            | Gagan                                   | `src/C_cleaning.ipynb` (also `src/C_cleaning.py`)       |
| Graph model (shared)            | Group                                   | `src/G_graph_model.ipynb` (also `src/G_graph_model.py`) |
| Dashboard                       | Group                                   | `src/dashboard.py`                                      |

## Dataset

ImmoScout24 rental listings, 268,850 rows and 49 columns, four snapshots (Sep 2018 to Feb 2020).
Zenodo DOI [10.5281/zenodo.23494643](https://doi.org/10.5281/zenodo.23494643), licence CC BY 4.0.

## Setup (once)

You need Python 3.11 or newer. Open a terminal **in this folder** (the one with this README):

```
python -m venv .venv
.venv\Scripts\activate                 (Windows)
source .venv/bin/activate              (Mac or Linux)
pip install -r requirements.txt
```

The dashboard needs Streamlit 1.46 or newer (`pip install -U streamlit` if a filter button is missing).
If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

Then put the raw file at `data/raw/immo_data.csv`.

## Run

to run dashboard.py file use this `python -m streamlit run src/dashboard.py`.

`IMPORTANT`
"and also make sure to run all files then only under outputs files will be displayed as we have limit to 100mb for uploading
if every output file is there it is taking too much and we can't submit in a single zip file"

Run the notebooks in this order. In VS Code, open each one, pick the `.venv` kernel and click **Run All**.

| Step | Notebook                 | Reads       | Writes                                                                                                                 |
| ---- | ------------------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1    | `A_build_metadata.ipynb` | raw CSV     | `outputs/fair/data_dictionary.csv`, `metadata_record.jsonld`                                                           |
| 2    | `B_profiling.ipynb`      | raw CSV     | `outputs/profiling/`: `missing_profile.csv`, `quality_issues.csv`, `issue_register.csv`, 3 charts                      |
| 3    | `C_cleaning.ipynb`       | raw CSV     | `data/cleaned/immo_data_clean.csv`, `outputs/cleaning/audit_log.csv`                                                   |
| 4    | `G_graph_model.ipynb`    | cleaned CSV | `outputs/graph/`: `graph_schema.png`, `property_graph_sample.gexf`, `district_graph.gexf/.png`, `district_metrics.csv` |

C and the graph take a few minutes each. Step 4 needs step 3 first.
Without Jupyter: `python src/C_cleaning.py` and then `python src/G_graph_model.py`.

Dashboard:

```
streamlit run src/dashboard.py
```

It opens at http://localhost:8501. The first load takes 10 to 20 seconds. Stop it with Ctrl + C.
Tabs: Overview, A · FAIR & Metadata, B · Data Quality, C · Cleaning Pipeline, Graph, Explore. A filter bar at the top
(state, scrape wave, base rent) drives the Explore and Graph tabs.
The `.gexf` files open in [Gephi](https://gephi.org/).

## Key results

- **FAIR (A):** F-UJI scored the Zenodo record and the Kaggle page 22 of 26 points (84.6%). Evidence: the F-UJI JSON
  and PDF exports in `outputs/fair/`, plus the Schema Markup Validator result.
- **Quality (B):** 12 issues across five dimensions, for example 71,013 hidden missing streets (`no_information`),
  104,441 street names with HTML codes, 75,356 postcodes without their leading zero, 11,270 repeat listings, and
  electricity prices collected in only one snapshot. Each issue has an ID that the cleaning log cites.
- **Cleaning (C):** 8 logged steps took the table from 268,850 to 259,690 rows. The mean base rent fell from 694.1 to
  650.9 EUR and its standard deviation from 19,536 to 506, because a few impossible values had inflated it.
- **Graph:** a property graph (Listing, District, PostcodeRegion, State, HeatingType, FlatType) and a district graph
  with 419 nodes and 1,910 edges, 11 communities (modularity 0.80), measured with degree and betweenness.

## Deliverables in this ZIP (brief section 6)

| #   | Deliverable                            | Where                                                                                          |
| --- | -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1   | Joint written report (max 4,000 words) | `report/`: joint report, with the A, B and C component sections                                |
| 2   | Python cleaning script                 | `src/C_cleaning.ipynb`, `src/C_cleaning.py`                                                    |
| 3   | Tableau Prep flow                      | not used                                                                                       |
| 4   | Metadata record                        | `outputs/fair/metadata_record.jsonld`, `outputs/fair/data_dictionary.csv`                      |
| 5   | FAIR assessment evidence               | `outputs/fair/`: F-UJI PDF and JSON exports (Kaggle and Zenodo), `schema_validator_result.png` |
| 6   | Quality visualisations                 | `outputs/profiling/chart1_missing.png`, `chart2_rent_per_m2.png`, `chart3_issues.png`          |
| 7   | Original and cleaned datasets          | `data/raw/immo_data.csv`, `data/cleaned/immo_data_clean.csv` (add both before submitting)      |
| 8   | Graph files                            | `src/G_graph_model.py`, `outputs/graph/*.gexf`, `*.png`                                        |
| 9   | Reflection logs                        | submitted by each student on the course portal, not in the ZIP                                 |
| –   | Group presentation                     | `report/Data_Management_Presentation.pptx`                                                     |

## Limitations

- The newest listings are from February 2020, so results describe the 2018 to 2020 market.
- Asking rents are not signed contract rents.
- Repeat listings are matched on six key columns, so some repeats may remain.
