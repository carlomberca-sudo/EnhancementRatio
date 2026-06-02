import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Enhancement Ratio Analyzer", layout="wide")

# -------------------------------------------------
# Session state
# -------------------------------------------------
def init_session_state():
    defaults = {
        "er_results_ready": False,
        "er_summary_df": pd.DataFrame(),
        "er_review_df": pd.DataFrame(),
        "er_warnings_df": pd.DataFrame(),
        "er_details": {},
        "er_thickness_editor_df": pd.DataFrame(),
        "er_uploader_key_counter": 0,
        "er_thickness_csv_key_counter": 0,
        "er_reference_keywords_text": "REF, REFERENCE, BLANK, CONTROL",
        "er_last_reference_keywords_text": "REF, REFERENCE, BLANK, CONTROL",
        "er_last_matching_mode": "Smart mode",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app_state():
    """Reset uploaded files, optional CSV, editor tables, results, warnings, and graph widgets."""
    st.session_state.er_results_ready = False
    st.session_state.er_summary_df = pd.DataFrame()
    st.session_state.er_review_df = pd.DataFrame()
    st.session_state.er_warnings_df = pd.DataFrame()
    st.session_state.er_details = {}
    st.session_state.er_thickness_editor_df = pd.DataFrame()
    st.session_state.er_uploader_key_counter += 1
    st.session_state.er_thickness_csv_key_counter += 1

    keys_to_delete = [
        "er_data_editor",
        "er_graph_multiselect",
        "er_graph_mode",
        "er_manual_y_axis",
        "er_x_min",
        "er_x_max",
        "er_y_min",
        "er_y_max",
        "er_sim_x_min",
        "er_sim_x_max",
        "er_sim_y_min",
        "er_sim_y_max",
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]


init_session_state()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_spectrum(uploaded_file, skiprows=1, max_rows=1024):
    data = np.loadtxt(uploaded_file, skiprows=skiprows, max_rows=max_rows)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Spectrum file must contain at least 3 columns.")
    channels = data[:, 0]
    intensity = data[:, 2]
    return channels, intensity


def calculate_wavelengths(channels, center_wavelength=550, grating_number=1):
    g = 0.4196 if grating_number == 1 else 0.4192
    return np.array([center_wavelength - ((i - 513) * g) for i in channels], dtype=float)


def normalize_name(name: str) -> str:
    stem = Path(str(name)).stem.upper().strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def split_name_tokens(name: str):
    return [token for token in re.split(r"[-_ .]+", normalize_name(name)) if token]


def parse_reference_keywords(reference_keywords_text: str):
    keywords = [
        normalize_name(k)
        for k in str(reference_keywords_text).split(",")
        if str(k).strip()
    ]
    return keywords or ["REF"]


def detect_is_reference(name: str, reference_keywords=None) -> bool:
    """
    Safer reference detection.
    Short keywords such as REF must match a filename token.
    Longer custom keywords can also match as substrings.
    """
    reference_keywords = reference_keywords or ["REF"]
    n = normalize_name(name)
    tokens = split_name_tokens(name)

    for keyword in reference_keywords:
        keyword_norm = normalize_name(keyword)
        if keyword_norm in tokens:
            return True
        if len(keyword_norm) > 3 and keyword_norm in n:
            return True
    return False


def detect_material_family(name: str):
    n = normalize_name(name)
    tokens = split_name_tokens(name)
    ordered = ["PMMA", "EMA", "PET", "PE", "LAM"]

    for fam in ordered:
        if fam in tokens:
            return fam

    for fam in ordered:
        if fam in n:
            return fam

    return None


def extract_sample_name(filename: str):
    stem = Path(filename).stem
    stem = stem.replace("_Excplasma_Cen550_NewM266Gr1_Slit100_Filter4_t500ms", "")
    stem = re.sub(r"_Exc.*$", "", stem, flags=re.IGNORECASE)
    return normalize_name(stem)


def extract_thickness_from_name(sample_name: str):
    wet_to_dry = {50: 11.0, 100: 12.0, 150: 18.0, 200: 38.0, 400: 60.0}
    parts = re.split(r"[-_ ]+", normalize_name(sample_name))
    for part in parts:
        if part.isdigit():
            val = int(part)
            if val in wet_to_dry:
                return wet_to_dry[val], f"inferred_from_name({val})"
    return np.nan, "missing"


def match_reference(sample_name: str, available_references: list[str]):
    sample_norm = normalize_name(sample_name)
    family = detect_material_family(sample_norm)

    if not available_references:
        return None, "no_references_uploaded"

    if family is not None:
        family_matches = [r for r in available_references if detect_material_family(r) == family]
        if len(family_matches) == 1:
            return family_matches[0], f"matched_family:{family}"
        if len(family_matches) > 1:
            a_matches = [r for r in family_matches if normalize_name(r).endswith(" A")]
            if len(a_matches) == 1:
                return a_matches[0], f"matched_family_prefer_A:{family}"
            return family_matches[0], f"multiple_family_matches:{family}"

    generic_priority = ["LAM", "PET", "EMA", "PMMA", "PE"]
    for fam in generic_priority:
        fam_matches = [r for r in available_references if detect_material_family(r) == fam]
        if len(fam_matches) == 1:
            return fam_matches[0], f"fallback_family:{fam}"
        if len(fam_matches) > 1:
            return fam_matches[0], f"fallback_multiple_family:{fam}"

    return available_references[0], "fallback_first_reference"


def safe_float_or_nan(value):
    if value is None:
        return np.nan
    if isinstance(value, str) and not value.strip():
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return np.nan


def build_review_table(measurement_files, manual_thickness_map=None, reference_keywords=None):
    manual_thickness_map = manual_thickness_map or {}
    reference_keywords = reference_keywords or ["REF"]
    rows = []

    uploaded_names = [extract_sample_name(f.name) for f in measurement_files]
    ref_names = [n for n in uploaded_names if detect_is_reference(n, reference_keywords)]

    for f in measurement_files:
        parsed_name = extract_sample_name(f.name)
        is_ref = detect_is_reference(parsed_name, reference_keywords)
        family = detect_material_family(parsed_name)
        matched_ref, match_reason = (
            (parsed_name, "self_reference")
            if is_ref
            else match_reference(parsed_name, ref_names)
        )

        manual_thickness = manual_thickness_map.get(parsed_name)
        if manual_thickness is not None and not pd.isna(manual_thickness):
            thickness = float(manual_thickness)
            thickness_source = "manual_upload"
        else:
            thickness, thickness_source = extract_thickness_from_name(parsed_name)

        rows.append({
            "File": f.name,
            "Parsed name": parsed_name,
            "Type": "Reference" if is_ref else "Sample",
            "Family": family,
            "Matched reference": matched_ref,
            "Reference match reason": match_reason,
            "Thickness (µm)": thickness,
            "Thickness source": thickness_source,
        })

    return pd.DataFrame(rows)


def build_manual_review_table(measurement_files, manual_thickness_map=None):
    manual_thickness_map = manual_thickness_map or {}
    rows = []

    for f in measurement_files:
        parsed_name = extract_sample_name(f.name)
        manual_thickness = manual_thickness_map.get(parsed_name)

        if manual_thickness is not None and not pd.isna(manual_thickness):
            thickness = float(manual_thickness)
            thickness_source = "manual_upload"
        else:
            thickness, thickness_source = extract_thickness_from_name(parsed_name)

        rows.append({
            "File": f.name,
            "Parsed name": parsed_name,
            "Type": "Sample",
            "Family": detect_material_family(parsed_name),
            "Matched reference": None,
            "Reference match reason": "manual",
            "Thickness (µm)": thickness,
            "Thickness source": thickness_source,
        })

    return pd.DataFrame(rows)


def make_downloadable_summary(results_long: pd.DataFrame):
    if results_long.empty:
        return pd.DataFrame()
    summary = results_long[[
        "Sample",
        "Reference",
        "Family",
        "Thickness (µm)",
        "Thickness source",
        "Mean ratio 400-700",
        "Min ratio 400-700",
        "Max ratio 400-700",
        "PAR transmission (%)",
        "Red transmission (%)",
        "SFQY",
    ]].copy()
    return summary.sort_values(by=["Sample"]).reset_index(drop=True)


def band_stats(wl, ratio, lo=400, hi=700):
    mask = (wl >= lo) & (wl <= hi)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    vals = ratio[mask]
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def band_average(wl, y, lo, hi):
    left = min(lo, hi)
    right = max(lo, hi)
    mask = (wl >= left) & (wl <= right)
    if np.sum(mask) < 2:
        return np.nan
    x = wl[mask]
    vals = y[mask]
    width = x[-1] - x[0]
    if abs(width) < 1e-12:
        return np.nan
    return float(np.trapezoid(vals, x) / width)


def parse_thickness_csv(uploaded_file):
    if uploaded_file is None:
        return {}
    df = pd.read_csv(uploaded_file)
    required = {"Parsed name", "Thickness (µm)"}
    if not required.issubset(df.columns):
        raise ValueError("Thickness CSV must contain columns: 'Parsed name' and 'Thickness (µm)'.")
    return {
        normalize_name(str(row["Parsed name"])): float(row["Thickness (µm)"])
        for _, row in df.iterrows()
        if pd.notna(row["Parsed name"]) and pd.notna(row["Thickness (µm)"])
    }


def make_sample_label(sample_name, thickness=None, ref_name=None):
    label = sample_name
    if thickness is not None and pd.notna(thickness):
        label += f" / {float(thickness):.1f} µm"
    if ref_name:
        label += f" | ref: {ref_name}"
    return label


def build_plotly_figure(
    details_dict,
    selected_samples,
    mode="ratio",
    d_ref=None,
    x_range=None,
    y_range=None,
    show_metric_bands=False,
    par_band=(400, 750),
    red_band=(600, 750),
):
    fig = go.Figure()
    title = "Graph"
    y_label = "Value"

    for sample_name in selected_samples:
        d = details_dict[sample_name]

        if mode == "ratio":
            y = d["ratio"]
            y_label = "Transmission normalized"
            title = "Enhancement ratio"
            trace_label = make_sample_label(d["sample_name"], d["thickness"], d["ref_name"])

        elif mode == "raw_sample":
            y = d["sample_i"]
            y_label = "Intensity"
            title = "Raw sample spectra"
            trace_label = d["sample_name"]

        elif mode == "raw_reference":
            y = d["ref_i"]
            y_label = "Intensity"
            title = "Raw reference spectra"
            trace_label = f"{d['sample_name']} | {d['ref_name']}"

        elif mode == "thickness_norm":
            if d["norm_ratio"] is None:
                continue
            y = d["norm_ratio"]
            y_label = f"Transmission at d = {d_ref:.1f} µm" if d_ref is not None else "Transmission"
            title = "Thickness-normalized transmission"
            trace_label = f"{d['sample_name']} → {d_ref:.1f} µm"

        else:
            continue

        fig.add_trace(
            go.Scatter(
                x=d["wl"],
                y=y,
                mode="lines",
                name=trace_label,
            )
        )

    if mode in ["ratio", "thickness_norm"]:
        fig.add_hline(y=1, line_width=1.5, line_color="black")

    if show_metric_bands:
        par_left, par_right = min(par_band), max(par_band)
        red_left, red_right = min(red_band), max(red_band)

        fig.add_vrect(
            x0=par_left,
            x1=par_right,
            fillcolor="lightgreen",
            opacity=0.12,
            line_width=0,
            annotation_text="PAR band",
            annotation_position="top left",
        )
        fig.add_vrect(
            x0=red_left,
            x1=red_right,
            fillcolor="lightcoral",
            opacity=0.12,
            line_width=0,
            annotation_text="Red band",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Samples",
    )

    fig.update_xaxes(range=x_range if x_range is not None else [360, 770])
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    return fig


def build_simulation_figure(simulation, x_range=None, y_range=None):
    fig = go.Figure()
    wl = simulation["wl"]

    fig.add_trace(
        go.Scatter(
            x=wl,
            y=simulation["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=simulation["lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Envelope from all μ(λ) samples",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=simulation["mean"],
            mode="lines",
            name=f"Simulated, d = {simulation['thickness']:.1f} µm (mean μ)",
        )
    )

    fig.update_layout(
        title="Predicted transmission by thickness",
        xaxis_title="Wavelength (nm)",
        yaxis_title=f"Predicted transmission at d = {simulation['thickness']:.1f} µm",
        hovermode="x unified",
    )
    fig.update_xaxes(range=x_range if x_range is not None else [360, 770])
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def duplicate_parsed_name_warnings(measurement_files):
    parsed_names = [extract_sample_name(f.name) for f in measurement_files]
    counts = pd.Series(parsed_names).value_counts()
    duplicate_counts = counts[counts > 1]

    warnings = []
    for name, count in duplicate_counts.items():
        warnings.append({
            "Type": "Duplicate parsed name",
            "Sample": name,
            "Message": f"{count} uploaded files resolve to the same parsed name. Rename files or adjust parsing logic before trusting results.",
        })
    return warnings


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Enhancement Ratio Analyzer")
st.caption(
    "Upload all measurement files. The app will detect references, match samples to references, suggest thickness values, and run enhancement-ratio analysis."
)

left, right = st.columns([1, 1.7], gap="large")

with left:
    st.subheader("Inputs")

    measurement_files = st.file_uploader(
        "1. Drop all measurement files",
        type=["txt", "dat", "csv"],
        accept_multiple_files=True,
        key=f"er_measurement_files_{st.session_state.er_uploader_key_counter}",
    )

    if st.button("Clear uploaded files", type="secondary", width="stretch"):
        reset_app_state()
        st.rerun()

    matching_mode = st.radio(
        "2. Reference matching mode",
        options=["Smart mode", "Manual mode"],
        index=0,
    )

    reference_keywords_text = st.text_input(
        "Reference keywords",
        value=st.session_state.er_reference_keywords_text,
        help="Comma-separated words used to identify reference files. Example: REF, REFERENCE, BLANK, CONTROL",
    )
    st.session_state.er_reference_keywords_text = reference_keywords_text
    reference_keywords = parse_reference_keywords(reference_keywords_text)

    mode_changed = matching_mode != st.session_state.er_last_matching_mode
    keywords_changed = reference_keywords_text != st.session_state.er_last_reference_keywords_text

    if mode_changed or keywords_changed:
        st.info("Matching settings changed. Click 'Build / rebuild review table' to apply them. Existing manual edits are not overwritten automatically.")

    center_wavelength = st.number_input(
        "3. Center wavelength (nm)",
        min_value=200,
        max_value=1200,
        value=550,
        step=1,
    )

    grating_number = st.selectbox("4. Grating", options=[1, 2], index=1)

    st.subheader("Optional switches")
    plot_raw = st.toggle("Show raw spectra", value=False)
    solve_thickness = st.toggle("Run thickness normalization", value=False)

    run_simulation = st.toggle(
        "Run simulation",
        value=False,
        disabled=not solve_thickness,
        help="Simulation requires thickness normalization because it uses μ(λ).",
    )
    if not solve_thickness:
        run_simulation = False

    simulated_thickness = None
    if run_simulation:
        simulated_thickness = st.number_input(
            "Simulated thickness (µm)",
            min_value=1.0,
            max_value=5000.0,
            value=146.0,
            step=1.0,
        )

    st.subheader("Advanced optical metrics")

    red_band_min = st.number_input(
        "Red band min (nm)",
        min_value=400,
        max_value=900,
        value=600,
        step=1,
    )

    red_band_max = st.number_input(
        "Red band max (nm)",
        min_value=400,
        max_value=900,
        value=750,
        step=1,
    )

    show_metric_bands = st.toggle(
        "Show PAR and red integration bands on graphs",
        value=True,
    )

    thickness_csv = st.file_uploader(
        "Optional thickness CSV",
        type=["csv"],
        help="Optional CSV with columns: Parsed name, Thickness (µm)",
        key=f"er_thickness_csv_{st.session_state.er_thickness_csv_key_counter}",
    )

    preview = st.button("Build / rebuild review table", type="secondary", width="stretch")
    run_analysis = st.button("Run enhancement analysis", type="primary", width="stretch")

with right:
    st.subheader("Review and results")

    manual_thickness_map = {}
    if thickness_csv is not None:
        try:
            manual_thickness_map = parse_thickness_csv(thickness_csv)
        except Exception as e:
            st.error(f"Thickness CSV error: {e}")

    should_build_preview = preview or (
        measurement_files and st.session_state.er_thickness_editor_df.empty
    )

    if should_build_preview:
        if not measurement_files:
            st.warning("Upload the measurement files first.")
        else:
            if matching_mode == "Smart mode":
                review_df = build_review_table(
                    measurement_files,
                    manual_thickness_map=manual_thickness_map,
                    reference_keywords=reference_keywords,
                )
            else:
                review_df = build_manual_review_table(
                    measurement_files,
                    manual_thickness_map=manual_thickness_map,
                )

            st.session_state.er_review_df = review_df
            st.session_state.er_thickness_editor_df = review_df[
                [
                    "File",
                    "Parsed name",
                    "Type",
                    "Family",
                    "Matched reference",
                    "Thickness (µm)",
                    "Thickness source",
                ]
            ].copy()
            st.session_state.er_last_matching_mode = matching_mode
            st.session_state.er_last_reference_keywords_text = reference_keywords_text

    if not st.session_state.er_thickness_editor_df.empty:
        st.subheader("Editable thickness / reference review")
        st.caption(
            "Manual edits in this table are preserved during analysis. Use Type to mark references, and Matched reference to correct pairings."
        )

        disabled_cols = ["File", "Parsed name", "Family", "Thickness source"]
        reference_options = st.session_state.er_thickness_editor_df["Parsed name"].tolist()

        edited_df = st.data_editor(
            st.session_state.er_thickness_editor_df,
            width="stretch",
            num_rows="fixed",
            key="er_data_editor",
            disabled=disabled_cols,
            column_config={
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["Sample", "Reference"],
                    required=True,
                ),
                "Matched reference": st.column_config.SelectboxColumn(
                    "Matched reference",
                    options=reference_options,
                ),
                "Thickness (µm)": st.column_config.NumberColumn(
                    "Thickness (µm)",
                    min_value=0.0,
                    step=1.0,
                ),
            },
        )
        st.session_state.er_thickness_editor_df = edited_df.copy()

        review_csv = edited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download thickness review CSV",
            data=review_csv,
            file_name="enhancement_ratio_thickness_review.csv",
            mime="text/csv",
            width="stretch",
        )

    if run_analysis:
        try:
            if not measurement_files:
                st.error("Please upload the measurement files.")
                st.stop()

            editor_df = st.session_state.er_thickness_editor_df.copy()

            if editor_df.empty:
                if matching_mode == "Smart mode":
                    generated_review_df = build_review_table(
                        measurement_files,
                        manual_thickness_map=manual_thickness_map,
                        reference_keywords=reference_keywords,
                    )
                else:
                    generated_review_df = build_manual_review_table(
                        measurement_files,
                        manual_thickness_map=manual_thickness_map,
                    )
                editor_df = generated_review_df[
                    [
                        "File",
                        "Parsed name",
                        "Type",
                        "Family",
                        "Matched reference",
                        "Thickness (µm)",
                        "Thickness source",
                    ]
                ].copy()

            review_df = editor_df.copy()
            review_df["Reference match reason"] = "manual_or_reviewed"

            warnings = duplicate_parsed_name_warnings(measurement_files)
            details = {}
            results_long = []
            mu_list = []
            sim_wl = None
            d_ref = None

            parsed_uploaded_names = [extract_sample_name(f.name) for f in measurement_files]
            duplicate_names = pd.Series(parsed_uploaded_names).value_counts()
            duplicate_names = duplicate_names[duplicate_names > 1]

            if not duplicate_names.empty:
                st.session_state.er_summary_df = pd.DataFrame()
                st.session_state.er_review_df = review_df
                st.session_state.er_warnings_df = pd.DataFrame(warnings)
                st.session_state.er_details = {
                    "samples": {},
                    "simulation": None,
                    "d_ref": None,
                    "plot_raw": plot_raw,
                    "solve_thickness": solve_thickness,
                    "run_simulation": run_simulation,
                }
                st.session_state.er_results_ready = True
                st.warning("Duplicate parsed names found. Fix filenames or parsing before running analysis.")
                st.stop()

            file_lookup = {extract_sample_name(f.name): f for f in measurement_files}
            reference_names_declared = set(
                review_df[review_df["Type"] == "Reference"]["Parsed name"].tolist()
            )

            if solve_thickness:
                for _, row in review_df.iterrows():
                    thickness_value = safe_float_or_nan(row.get("Thickness (µm)"))
                    if row.get("Type") == "Sample" and pd.notna(thickness_value):
                        if d_ref is None or thickness_value > d_ref:
                            d_ref = thickness_value

            for _, row in review_df.iterrows():
                if row.get("Type") != "Sample":
                    continue

                sample_name = row.get("Parsed name")
                ref_name = row.get("Matched reference")
                family = row.get("Family")
                thickness = safe_float_or_nan(row.get("Thickness (µm)"))
                thickness_source = row.get("Thickness source")

                if pd.isna(ref_name) or ref_name is None or str(ref_name).strip() == "":
                    warnings.append({
                        "Type": "Missing reference assignment",
                        "Sample": sample_name,
                        "Message": "No reference assigned to this sample.",
                    })
                    continue

                if sample_name not in file_lookup:
                    warnings.append({
                        "Type": "Missing sample file",
                        "Sample": sample_name,
                        "Message": f"Sample file '{sample_name}' was not found among uploaded files.",
                    })
                    continue

                if ref_name not in file_lookup:
                    warnings.append({
                        "Type": "Missing reference file",
                        "Sample": sample_name,
                        "Message": f"Matched reference '{ref_name}' was not found among uploaded files.",
                    })
                    continue

                if ref_name not in reference_names_declared:
                    warnings.append({
                        "Type": "Reference points to non-reference row",
                        "Sample": sample_name,
                        "Message": f"'{ref_name}' is selected as reference, but its Type is not marked as Reference.",
                    })

                sample_file = file_lookup[sample_name]
                ref_file = file_lookup[ref_name]

                try:
                    try:
                        sample_file.seek(0)
                        ref_file.seek(0)
                    except Exception:
                        pass

                    channels_s, sample_i = load_spectrum(sample_file)
                    channels_r, ref_i = load_spectrum(ref_file)

                    if len(channels_s) != len(channels_r):
                        raise ValueError("Sample and reference files do not have the same number of points.")

                    wl = calculate_wavelengths(
                        channels_s,
                        center_wavelength=center_wavelength,
                        grating_number=grating_number,
                    )
                    ratio = sample_i / np.clip(ref_i, 1e-12, None)
                    mean_ratio, min_ratio, max_ratio = band_stats(wl, ratio, 400, 700)

                    par_fraction = band_average(wl, ratio, 400, 750)
                    par_percent = par_fraction * 100 if pd.notna(par_fraction) else np.nan

                    red_fraction = band_average(wl, ratio, red_band_min, red_band_max)
                    red_percent = red_fraction * 100 if pd.notna(red_fraction) else np.nan

                    sfqy = np.nan
                    if (
                        pd.notna(par_fraction)
                        and pd.notna(red_fraction)
                        and abs(1.0 - par_fraction) > 1e-12
                    ):
                        sfqy = (red_fraction - 1.0) / (1.0 - par_fraction)

                    norm_ratio = None
                    mu_lambda = None
                    if solve_thickness and pd.notna(thickness) and d_ref is not None and thickness > 0:
                        ratio_clipped = np.clip(ratio, 1e-9, None)
                        norm_ratio = ratio_clipped ** (d_ref / float(thickness))
                        mu_lambda = (-np.log(ratio_clipped)) / float(thickness)
                        mu_list.append(mu_lambda)
                        if sim_wl is None:
                            sim_wl = wl

                    results_long.append({
                        "Sample": sample_name,
                        "Reference": ref_name,
                        "Family": family,
                        "Thickness (µm)": thickness,
                        "Thickness source": thickness_source,
                        "Mean ratio 400-700": mean_ratio,
                        "Min ratio 400-700": min_ratio,
                        "Max ratio 400-700": max_ratio,
                        "PAR transmission (%)": par_percent,
                        "Red transmission (%)": red_percent,
                        "SFQY": sfqy,
                    })

                    details[sample_name] = {
                        "wl": wl,
                        "sample_i": sample_i,
                        "ref_i": ref_i,
                        "ratio": ratio,
                        "norm_ratio": norm_ratio,
                        "mu_lambda": mu_lambda,
                        "sample_name": sample_name,
                        "ref_name": ref_name,
                        "family": family,
                        "thickness": thickness,
                        "par_percent": par_percent,
                        "red_percent": red_percent,
                        "sfqy": sfqy,
                    }

                except Exception as e:
                    warnings.append({
                        "Type": "Processing error",
                        "Sample": sample_name,
                        "Message": str(e),
                    })

            summary_df = make_downloadable_summary(pd.DataFrame(results_long))
            warnings_df = pd.DataFrame(warnings)

            simulation = None
            if run_simulation and mu_list and sim_wl is not None and simulated_thickness is not None:
                mu_stack = np.vstack(mu_list)
                mu_mean = np.mean(mu_stack, axis=0)
                d_sim = float(simulated_thickness)
                t_sim = np.exp(-mu_mean * d_sim)
                t_all = np.exp(-mu_stack * d_sim)
                t_lo = np.min(t_all, axis=0)
                t_hi = np.max(t_all, axis=0)
                simulation = {
                    "wl": sim_wl,
                    "mean": t_sim,
                    "lower": t_lo,
                    "upper": t_hi,
                    "thickness": d_sim,
                }

            st.session_state.er_summary_df = summary_df
            st.session_state.er_review_df = review_df
            st.session_state.er_warnings_df = warnings_df
            st.session_state.er_details = {
                "samples": details,
                "simulation": simulation,
                "d_ref": d_ref,
                "plot_raw": plot_raw,
                "solve_thickness": solve_thickness,
                "run_simulation": run_simulation,
            }
            st.session_state.er_results_ready = True

        except Exception as e:
            st.error(f"Error while running enhancement analysis: {e}")

    if st.session_state.er_results_ready:
        summary_df = st.session_state.er_summary_df
        review_df = st.session_state.er_review_df
        warnings_df = st.session_state.er_warnings_df
        details_state = st.session_state.er_details
        details = details_state.get("samples", {})
        simulation = details_state.get("simulation")
        d_ref = details_state.get("d_ref")
        plot_raw_state = details_state.get("plot_raw", False)
        solve_thickness_state = details_state.get("solve_thickness", False)
        run_simulation_state = details_state.get("run_simulation", False)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Summary",
            "Review table",
            "Graphs",
            "Simulation",
            "Warnings",
        ])

        with tab1:
            st.subheader("Enhancement ratio summary")
            if summary_df.empty:
                st.info("No results generated.")
            else:
                st.dataframe(summary_df, width="stretch")
                csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download summary CSV",
                    data=csv_bytes,
                    file_name="enhancement_ratio_summary.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with tab2:
            st.subheader("Parsed files and matching")
            st.dataframe(review_df, width="stretch")

        with tab3:
            st.subheader("Interactive graphs")

            if not details:
                st.info("No graphable results available.")
            else:
                sample_options = sorted(details.keys())

                selected_samples = st.multiselect(
                    "Select one or more samples to overlay",
                    options=sample_options,
                    default=sample_options[: min(3, len(sample_options))],
                    key="er_graph_multiselect",
                )

                if not selected_samples:
                    st.warning("Select at least one sample.")
                else:
                    graph_mode_options = ["Enhancement ratio"]
                    if plot_raw_state:
                        graph_mode_options.extend(["Raw sample spectra", "Raw reference spectra"])
                    if solve_thickness_state:
                        has_norm = any(details[s]["norm_ratio"] is not None for s in selected_samples)
                        if has_norm:
                            graph_mode_options.append("Thickness-normalized transmission")

                    graph_mode = st.radio(
                        "Graph type",
                        options=graph_mode_options,
                        horizontal=True,
                        key="er_graph_mode",
                    )

                    ratio_like_graph = graph_mode in [
                        "Enhancement ratio",
                        "Thickness-normalized transmission",
                    ]

                    st.markdown("#### Axis limits")
                    col_x1, col_x2 = st.columns(2)
                    with col_x1:
                        x_min = st.number_input(
                            "X min",
                            min_value=200,
                            max_value=1200,
                            value=360,
                            step=1,
                            key="er_x_min",
                        )
                    with col_x2:
                        x_max = st.number_input(
                            "X max",
                            min_value=200,
                            max_value=1200,
                            value=770,
                            step=1,
                            key="er_x_max",
                        )

                    manual_y_axis = st.checkbox(
                        "Use manual Y-axis limits",
                        value=ratio_like_graph,
                        key="er_manual_y_axis",
                        help="Recommended for enhancement-ratio graphs. Leave off for raw spectra unless you know the intensity scale.",
                    )

                    y_range = None
                    if manual_y_axis:
                        col_y1, col_y2 = st.columns(2)
                        with col_y1:
                            y_min = st.number_input(
                                "Y min",
                                value=0.0,
                                step=0.1,
                                key="er_y_min",
                            )
                        with col_y2:
                            y_max = st.number_input(
                                "Y max",
                                value=2.0,
                                step=0.1,
                                key="er_y_max",
                            )
                        y_range = [y_min, y_max]
                    else:
                        y_min, y_max = None, None

                    valid_axes = x_min < x_max and (
                        not manual_y_axis or y_min < y_max
                    )

                    if not valid_axes:
                        st.warning("Axis limits are invalid. X min must be smaller than X max, and Y min must be smaller than Y max.")
                    else:
                        if graph_mode == "Enhancement ratio":
                            fig = build_plotly_figure(
                                details_dict=details,
                                selected_samples=selected_samples,
                                mode="ratio",
                                d_ref=d_ref,
                                x_range=[x_min, x_max],
                                y_range=y_range,
                                show_metric_bands=show_metric_bands,
                                par_band=(400, 750),
                                red_band=(red_band_min, red_band_max),
                            )
                        elif graph_mode == "Raw sample spectra":
                            fig = build_plotly_figure(
                                details_dict=details,
                                selected_samples=selected_samples,
                                mode="raw_sample",
                                d_ref=d_ref,
                                x_range=[x_min, x_max],
                                y_range=y_range,
                                show_metric_bands=show_metric_bands,
                                par_band=(400, 750),
                                red_band=(red_band_min, red_band_max),
                            )
                        elif graph_mode == "Raw reference spectra":
                            fig = build_plotly_figure(
                                details_dict=details,
                                selected_samples=selected_samples,
                                mode="raw_reference",
                                d_ref=d_ref,
                                x_range=[x_min, x_max],
                                y_range=y_range,
                                show_metric_bands=show_metric_bands,
                                par_band=(400, 750),
                                red_band=(red_band_min, red_band_max),
                            )
                        else:
                            fig = build_plotly_figure(
                                details_dict=details,
                                selected_samples=selected_samples,
                                mode="thickness_norm",
                                d_ref=d_ref,
                                x_range=[x_min, x_max],
                                y_range=y_range,
                                show_metric_bands=show_metric_bands,
                                par_band=(400, 750),
                                red_band=(red_band_min, red_band_max),
                            )

                        st.plotly_chart(fig, width="stretch")

                    st.subheader("Selected sample details")
                    detail_rows = []
                    for s in selected_samples:
                        d = details[s]
                        detail_rows.append({
                            "Sample": d["sample_name"],
                            "Reference": d["ref_name"],
                            "Family": d["family"],
                            "Thickness (µm)": d["thickness"],
                            "PAR transmission (%)": d["par_percent"],
                            "Red transmission (%)": d["red_percent"],
                            "SFQY": d["sfqy"],
                        })
                    st.dataframe(pd.DataFrame(detail_rows), width="stretch")

        with tab4:
            st.subheader("Simulation")
            if not run_simulation_state:
                st.info("Simulation was turned off.")
            elif simulation is None:
                st.info("No simulation available. Enable thickness normalization and ensure at least one valid thickness is present.")
            else:
                st.markdown("#### Axis limits")
                col_sx1, col_sx2, col_sy1, col_sy2 = st.columns(4)

                with col_sx1:
                    sim_x_min = st.number_input(
                        "Simulation X min",
                        min_value=200,
                        max_value=1200,
                        value=360,
                        step=1,
                        key="er_sim_x_min",
                    )
                with col_sx2:
                    sim_x_max = st.number_input(
                        "Simulation X max",
                        min_value=200,
                        max_value=1200,
                        value=770,
                        step=1,
                        key="er_sim_x_max",
                    )
                with col_sy1:
                    sim_y_min = st.number_input(
                        "Simulation Y min",
                        value=0.0,
                        step=0.1,
                        key="er_sim_y_min",
                    )
                with col_sy2:
                    sim_y_max = st.number_input(
                        "Simulation Y max",
                        value=2.0,
                        step=0.1,
                        key="er_sim_y_max",
                    )

                valid_sim_axes = sim_x_min < sim_x_max and sim_y_min < sim_y_max

                if not valid_sim_axes:
                    st.warning("Simulation axis limits are invalid.")
                else:
                    fig_sim = build_simulation_figure(
                        simulation,
                        x_range=[sim_x_min, sim_x_max],
                        y_range=[sim_y_min, sim_y_max],
                    )
                    st.plotly_chart(fig_sim, width="stretch")

                sim_df = pd.DataFrame({
                    "Wavelength_nm": simulation["wl"],
                    "T_sim": simulation["mean"],
                    "T_sim_lower": simulation["lower"],
                    "T_sim_upper": simulation["upper"],
                })
                st.dataframe(sim_df, width="stretch")
                st.download_button(
                    "Download simulation CSV",
                    data=sim_df.to_csv(index=False).encode("utf-8"),
                    file_name="enhancement_ratio_simulation.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with tab5:
            st.subheader("Warnings")
            if warnings_df.empty:
                st.success("No warnings.")
            else:
                st.dataframe(warnings_df, width="stretch")

    else:
        st.info("Upload files, build the review table, optionally edit thickness/reference assignment, then run the analysis.")
