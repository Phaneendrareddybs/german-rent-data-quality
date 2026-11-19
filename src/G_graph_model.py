#!/usr/bin/env python
# coding: utf-8

# # Shared: Graph modelling (property graph + NetworkX analytics)
# 
# Reads `data/cleaned/immo_data_clean.csv` (run Component C first). Writes the property graph model
# (`graph_schema.png`), an instance graph on a 5,000-listing sample (`property_graph_sample.gexf`), the district
# network (`district_graph.gexf/.png`) and the analytics per district (`district_metrics.csv`) to `outputs/graph/`.
# Run All from top to bottom.

# In[ ]:


"""
graph_model.py  -  Group part: property graph model, NetworkX implementation, analytics
-----------------------------------------------------------------------------------------
Input : data/cleaned/immo_data_clean.csv   (output of clean_pipeline.py)
Output: outputs/graph/graph_schema.png              the property graph model (node + edge types)
        outputs/graph/property_graph_sample.gexf    instance graph on a 5,000-listing sample (Gephi)
        outputs/graph/district_graph.gexf           district projection (Gephi)
        outputs/graph/district_graph.png            district projection, coloured by community
        outputs/graph/district_metrics.csv          analytics per district

Run:  python src/G_graph_model.py
  or  open src/G_graph_model.ipynb and Run All

Two levels:
  1. Property graph  - the full model (Listing, District, State, PostcodeRegion, HeatingType,
                       FlatType) on a reproducible sample of listings.
  2. District graph  - projection used for analytics: districts are connected when they share a
                       2-digit postcode region; edge weight = number of shared regions.
"""


# In[ ]:


from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside the notebook: find the folder that contains data/raw
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").is_dir())

CLEAN = ROOT / "data" / "cleaned" / "immo_data_clean.csv"
OUT = ROOT / "outputs" / "graph"

SAMPLE_SIZE = 5_000   # listings in the instance graph
SEED = 42             # same sample and layout on every run
MIN_SHARED = 3        # a district counts as "in" a postcode region only with >= 3 listings there
                      # (filters out single stray listings caused by data-entry errors)

SCHEMA = [  # (source label, relationship, target label)
    ("Listing", "LOCATED_IN", "District"),
    ("District", "PART_OF", "State"),
    ("Listing", "IN_POSTCODE_REGION", "PostcodeRegion"),
    ("Listing", "HAS_HEATING", "HeatingType"),
    ("Listing", "OF_TYPE", "FlatType"),
]


# In[ ]:


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    cols = ["scoutId", "regio1", "regio2", "geo_plz", "baseRent", "livingSpace",
            "noRooms", "date", "heatingType", "typeOfFlat"]
    df = pd.read_csv(CLEAN, usecols=cols, dtype={"geo_plz": str}, low_memory=False)
    df = df.dropna(subset=["geo_plz", "regio2"])
    df["rent_per_m2"] = (df["baseRent"] / df["livingSpace"]).round(2)
    df["plz_region"] = df["geo_plz"].str[:2]
    return df


# In[ ]:


# ---------------------------------------------------------------------------
# 1. Property graph model (schema) and instance graph
# ---------------------------------------------------------------------------
def draw_schema(path):
    S = nx.DiGraph()
    for src, rel, dst in SCHEMA:
        S.add_edge(src, dst, rel=rel)
    pos = {"Listing": (0, 0), "District": (2, 1), "State": (4, 1), "PostcodeRegion": (2, -1),
           "HeatingType": (-2, 1), "FlatType": (-2, -1)}
    props = {
        "Listing": "scoutId, baseRent, livingSpace,\nnoRooms (-1 = missing), rent_per_m2, date",
        "District": "name, state,\nn_listings, median_rent_m2",
        "State": "name",
        "PostcodeRegion": "code (2 digits)",
        "HeatingType": "name",
        "FlatType": "name",
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    nx.draw_networkx_nodes(S, pos, node_size=5200, node_color="#DCE6F2", edgecolors="#1F4E79", ax=ax)
    nx.draw_networkx_labels(S, pos, font_weight="bold", font_size=9, ax=ax)
    nx.draw_networkx_edges(S, pos, arrowsize=20, node_size=5200, edge_color="#1F4E79", ax=ax)
    nx.draw_networkx_edge_labels(S, pos, edge_labels=nx.get_edge_attributes(S, "rel"), font_size=8, ax=ax)
    for n, (x, y) in pos.items():
        ax.text(x, y - 0.42, props[n], ha="center", va="top", fontsize=7.5, color="#404040")
    ax.set_title("Property graph model: node labels, relationships and properties")
    ax.set_xlim(-3, 5); ax.set_ylim(-2, 1.8); ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def build_property_graph(df, districts):
    """Instance graph on a reproducible sample. 'unknown' categories are not turned into nodes,
    because a hub node for 'missing' would be an artefact, not a real heating or flat type."""
    s = df.sample(min(SAMPLE_SIZE, len(df)), random_state=SEED)
    G = nx.MultiDiGraph()

    def node(nid, ntype, name, **props):
        if nid not in G:
            G.add_node(nid, type=ntype, label=str(name), **props)
        return nid

    for r in s.itertuples(index=False):
        lst = node(f"Listing:{r.scoutId}", "Listing", r.scoutId, baseRent=float(r.baseRent),
                   livingSpace=float(r.livingSpace), noRooms=float(r.noRooms) if pd.notna(r.noRooms) else -1.0,
                   rent_per_m2=float(r.rent_per_m2), date=str(r.date))
        d = districts.loc[r.regio2]
        dist = node(f"District:{r.regio2}", "District", r.regio2, state=d.state,
                    n_listings=int(d.n_listings), median_rent_m2=float(d.median_rent_m2))
        state = node(f"State:{r.regio1}", "State", r.regio1)
        region = node(f"PostcodeRegion:{r.plz_region}", "PostcodeRegion", r.plz_region)
        G.add_edge(lst, dist, relation="LOCATED_IN")
        G.add_edge(lst, region, relation="IN_POSTCODE_REGION")
        if not G.has_edge(dist, state):
            G.add_edge(dist, state, relation="PART_OF")
        if isinstance(r.heatingType, str) and r.heatingType != "unknown":
            G.add_edge(lst, node(f"HeatingType:{r.heatingType}", "HeatingType", r.heatingType), relation="HAS_HEATING")
        if isinstance(r.typeOfFlat, str) and r.typeOfFlat != "unknown":
            G.add_edge(lst, node(f"FlatType:{r.typeOfFlat}", "FlatType", r.typeOfFlat), relation="OF_TYPE")
    return G



# In[ ]:


# ---------------------------------------------------------------------------
# 2. District projection
# ---------------------------------------------------------------------------
def district_table(df):
    return (df.groupby("regio2")
              .agg(state=("regio1", lambda x: x.mode().iloc[0]),
                   n_listings=("scoutId", "size"),
                   median_rent_m2=("rent_per_m2", "median"))
              .round({"median_rent_m2": 2}))


def build_district_graph(df, districts):
    pairs = df.groupby(["regio2", "plz_region"]).size().reset_index(name="n")
    pairs = pairs[pairs["n"] >= MIN_SHARED]
    D = nx.Graph()
    for name, d in districts.iterrows():
        D.add_node(name, label=name, state=d.state, n_listings=int(d.n_listings),
                   median_rent_m2=float(d.median_rent_m2))
    for region, grp in pairs.groupby("plz_region"):
        for a, b in combinations(sorted(grp["regio2"]), 2):
            if D.has_edge(a, b):
                D[a][b]["weight"] += 1
            else:
                D.add_edge(a, b, weight=1)
    return D



# In[ ]:


# ---------------------------------------------------------------------------
# 3. Analytics
# ---------------------------------------------------------------------------
def analyse(D):
    deg = nx.degree_centrality(D)
    btw = nx.betweenness_centrality(D)  # shortest paths by hops
    comms = sorted(community.greedy_modularity_communities(D, weight="weight"), key=len, reverse=True)
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    modularity = community.modularity(D, comms, weight="weight")

    # Rent assortativity: Pearson correlation of median rent/m2 at both ends of every edge
    x = np.array([[D.nodes[u]["median_rent_m2"], D.nodes[v]["median_rent_m2"]] for u, v in D.edges()])
    both = np.vstack([x, x[:, ::-1]])
    rent_assort = float(np.corrcoef(both[:, 0], both[:, 1])[0, 1]) if len(x) > 1 else float("nan")

    # How well do communities follow state borders? (share of districts in their community's main state)
    states = nx.get_node_attributes(D, "state")
    purity = sum(pd.Series([states[n] for n in c]).value_counts().iloc[0] for c in comms) / D.number_of_nodes()

    metrics = pd.DataFrame({
        "district": list(D.nodes()),
        "state": [states[n] for n in D.nodes()],
        "n_listings": [D.nodes[n]["n_listings"] for n in D.nodes()],
        "median_rent_m2": [D.nodes[n]["median_rent_m2"] for n in D.nodes()],
        "degree": [D.degree(n) for n in D.nodes()],
        "degree_centrality": [round(deg[n], 4) for n in D.nodes()],
        "betweenness": [round(btw[n], 4) for n in D.nodes()],
        "community": [comm_of[n] for n in D.nodes()],
    })
    for n, i in comm_of.items():
        D.nodes[n]["community"] = i
        D.nodes[n]["betweenness"] = round(btw[n], 4)
        D.nodes[n]["degree_centrality"] = round(deg[n], 4)
    summary = {"modularity": modularity, "communities": len(comms),
               "rent_assortativity": rent_assort, "state_purity": purity}
    return metrics, summary, comms


def draw_district_graph(D, metrics, comms, path):
    pos = nx.spring_layout(D, seed=SEED, weight="weight", k=0.35)
    palette = plt.get_cmap("tab10")
    colors = [palette(D.nodes[n]["community"]) if D.nodes[n]["community"] < 10 else (0.8, 0.8, 0.8)
              for n in D.nodes()]
    sizes = [20 + 0.03 * D.nodes[n]["n_listings"] for n in D.nodes()]
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(D, pos, alpha=0.15, width=0.5, ax=ax)
    nx.draw_networkx_nodes(D, pos, node_color=colors, node_size=sizes, linewidths=0.3,
                           edgecolors="white", ax=ax)
    top = metrics.nlargest(8, "betweenness")["district"]
    nx.draw_networkx_labels(D, pos, labels={n: n for n in top}, font_size=8, font_weight="bold", ax=ax)
    ax.set_title("District network (shared postcode regions)\n"
                 "colour = community, size = number of listings, labels = top-8 betweenness")
    ax.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    districts = district_table(df)
    print(f"Loaded {len(df):,} listings in {len(districts)} districts\n")

    draw_schema(OUT / "graph_schema.png")

    G = build_property_graph(df, districts)
    nx.write_gexf(G, OUT / "property_graph_sample.gexf")
    types = pd.Series(nx.get_node_attributes(G, "type")).value_counts()
    rels = pd.Series([d["relation"] for _, _, d in G.edges(data=True)]).value_counts()
    print(f"Property graph (sample of {SAMPLE_SIZE:,} listings): "
          f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print("  nodes by label:", types.to_dict())
    print("  edges by type: ", rels.to_dict(), "\n")

    D = build_district_graph(df, districts)
    metrics, summary, comms = analyse(D)
    nx.write_gexf(D, OUT / "district_graph.gexf")
    metrics.sort_values("betweenness", ascending=False).to_csv(OUT / "district_metrics.csv", index=False)
    draw_district_graph(D, metrics, comms, OUT / "district_graph.png")

    iso = nx.number_of_isolates(D)
    print(f"District graph: {D.number_of_nodes()} nodes, {D.number_of_edges():,} edges, "
          f"density {nx.density(D):.3f}, {nx.number_connected_components(D)} components ({iso} isolated)")
    print(f"  Communities: {summary['communities']} | modularity {summary['modularity']:.3f} | "
          f"state purity {summary['state_purity']:.1%}")
    print(f"  Rent assortativity (median rent/m2 across edges): {summary['rent_assortativity']:.2f}")
    print(f"  corr(degree, median rent/m2): "
          f"{metrics['degree'].corr(metrics['median_rent_m2'], method='spearman'):.2f} (Spearman)\n")
    cols = ["district", "state", "degree", "betweenness", "median_rent_m2"]
    print("Top 10 by degree:\n", metrics.nlargest(10, "degree")[cols].to_string(index=False), "\n")
    print("Top 10 by betweenness:\n", metrics.nlargest(10, "betweenness")[cols].to_string(index=False), "\n")
    print("Largest communities (size, main state):")
    states = nx.get_node_attributes(D, "state")
    for i, c in enumerate(comms[:6]):
        main = pd.Series([states[n] for n in c]).value_counts()
        print(f"  {i}: {len(c)} districts, main state {main.index[0]} ({main.iloc[0] / len(c):.0%})")
    print(f"\nWrote files to {OUT}")


if __name__ == "__main__":
    main()



# In[ ]:


c = pd.read_csv(CLEAN,
                usecols=["regio2", "geo_plz"], dtype={"geo_plz": str})
e = c[c["regio2"] == "Eichstätt_Kreis"]
print(len(e), "listings")
print(e["geo_plz"].str[:2].value_counts())

