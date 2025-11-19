import numpy as np
import pandas as pd
import streamlit as st
import pgeocode

st.set_page_config(page_title="Nearby Postcodes Finder (AU)", page_icon="📍", layout="wide")

# ---------------- Data ----------------
@st.cache_data
def load_postcodes_au():
    # pgeocode bundles AU postcodes with centroids
    nomi = pgeocode.Nominatim("AU")
    df = nomi._data[["postal_code", "latitude", "longitude"]].dropna().copy()
    df["postal_code"] = df["postal_code"].astype(str).str.strip().str.zfill(4)
    df = df[(df["latitude"].notna()) & (df["longitude"].notna())]
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def get_geodist():
    return pgeocode.GeoDistance("AU")

def parse_postcodes(text: str):
    if not text:
        return []
    tokens = text.replace("\n", ",").split(",")
    out = []
    for t in tokens:
        s = "".join(ch for ch in t.strip() if ch.isdigit())
        if s:
            out.append(s.zfill(4))
    return out

def find_neighbours_pgeocode(df, inputs, radius_km):
    all_codes = df["postal_code"].tolist()
    code_set = set(all_codes)
    geodist = get_geodist()

    results_map = {}
    missing = []

    unique_inputs = list(dict.fromkeys(inputs))
    for p in unique_inputs:
        if p not in code_set:
            results_map[p] = []
            missing.append(p)
            continue
        dists = geodist.query_postal_code(p, all_codes)  # numpy array (km)
        neighbours = [c for c, d in zip(all_codes, dists) if (not np.isnan(d)) and d <= radius_km]
        results_map[p] = sorted(set(neighbours))

    rows = []
    for p in inputs:
        ns = results_map.get(p, [])
        rows.append({"input_postcode": p, "neighbours": ";".join(ns)})
    return pd.DataFrame(rows), sorted(set(missing))

def flatten_neighbour_list(results_df, dedupe=True, sort=True):
    items = []
    for s in results_df["neighbours"]:
        if isinstance(s, str) and s:
            items.extend([x for x in s.split(";") if x])

    if dedupe:
        items = list(set(items))

    if sort:
        items = sorted(items)

    # IMPORTANT: include a space after each comma in the output string
    csv_str = ", ".join(items)
    return items, csv_str

# ---------------- UI ----------------
st.title("📍 Nearby Postcodes Finder (AU)")
st.write("Paste comma-separated postcodes, set a radius (km). Output is a single comma-separated list of neighbouring postcodes.")

df_postcodes = load_postcodes_au()
st.caption(f"Loaded {len(df_postcodes):,} AU postcodes.")

col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area(
        "Postcodes",
        placeholder="2010, 5159, 4814, 6330, 4035, ...",
        height=150,
    )
with col2:
    radius = st.number_input(
        "Radius (km)",
        min_value=1.0,
        max_value=1000.0,
        value=15.0,
        step=1.0
    )

run = st.button("Find neighbours")

if run:
    inputs = parse_postcodes(text)
    if not inputs:
        st.warning("Please enter at least one postcode.")
        st.stop()

    results_df, missing = find_neighbours_pgeocode(df_postcodes, inputs, radius)

    # Flatten to a single comma-and-space separated list from the neighbours column
    neighbours_list, neighbours_csv = flatten_neighbour_list(results_df, dedupe=True, sort=True)

    computed = len(set(inputs)) - len(missing)
    st.success(f"Computed neighbours for {computed} postcode(s). Total neighbours in list: {len(neighbours_list)}.")
    if missing:
        st.warning("Not found in dataset: " + ", ".join(missing))

    # Display the comma + space separated list
    st.text_area("Comma-separated neighbours", neighbours_csv, height=150)

    # Download as TXT (comma + space separated)
    st.download_button(
        label="Download neighbours (TXT)",
        data=neighbours_csv.encode("utf-8"),
        file_name=f"neighbours_within_{int(radius)}km.txt",
        mime="text/plain",
    )

    # Download as CSV (single column)
    csv_df = pd.DataFrame({"postcode": neighbours_list})
    st.download_button(
        label="Download neighbours (CSV)",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name=f"neighbours_within_{int(radius)}km.csv",
        mime="text/csv",
    )


