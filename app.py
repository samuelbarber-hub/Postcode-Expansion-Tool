import io
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
    # Normalize to 4-digit strings (preserve leading zeros)
    df["postal_code"] = df["postal_code"].astype(str).str.strip().str.zfill(4)
    # Filter out rows where lat/lon are missing or invalid
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
        # distance from p to every AU postcode
        dists = geodist.query_postal_code(p, all_codes)  # numpy array of km
        neighbours = [c for c, d in zip(all_codes, dists) if (not np.isnan(d)) and d <= radius_km]
        results_map[p] = sorted(set(neighbours))

    rows = []
    for p in inputs:
        ns = results_map.get(p, [])
        rows.append(
            {
                "input_postcode": p,
                "neighbours": ";".join(ns),
                "neighbour_count": len(ns),
            }
        )
    return pd.DataFrame(rows), sorted(set(missing))

# ---------------- UI ----------------
st.title("📍 Nearby Postcodes Finder (AU)")
st.write("Paste comma-separated postcodes, set a radius (km), and download results. No latitude/longitude required.")

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

run = st.button("Find nearby postcodes")

if run:
    inputs = parse_postcodes(text)
    if not inputs:
        st.warning("Please enter at least one postcode.")
        st.stop()

    results_df, missing = find_neighbours_pgeocode(df_postcodes, inputs, radius)

    computed = len(set(inputs)) - len(missing)
    st.success(f"Computed neighbours for {computed} postcode(s).")
    if missing:
        st.warning("Not found in dataset: " + ", ".join(missing))

    st.dataframe(results_df, use_container_width=True, height=420)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name=f"nearby_within_{int(radius)}km.csv",
        mime="text/csv",
    )

    st.markdown("**Summary**")
    st.write(
        {
            "inputs_total": len(inputs),
            "unique_inputs": len(set(inputs)),
            "missing_inputs": len(missing),
            "rows_in_output": len(results_df),
        }
    )