import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- Page Config ---
st.set_page_config(page_title="Advanced 2T Tandem Simulator", layout="wide")
st.title("Comprehensive 2T Tandem Simulator (AM1.5G Integrated)")

# --- Load AM1.5G Spectrum ---
@st.cache_data
def load_spectrum():
    # Make sure ASTMG173.csv is in the same directory
    file_path = "ASTMG173.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: Could not find '{file_path}'. Please ensure it is in the same folder as app.py.")
        st.stop()
        
    # Skip the first row of text metadata
    df = pd.read_csv(file_path, skiprows=1)
    
    wavelengths = df.iloc[:, 0].values  # Wavelength (nm)
    am15g = df.iloc[:, 2].values        # Global tilt Irradiance (W*m-2*nm-1)
    
    # Convert Irradiance to Spectral Current Density (mA/cm2/nm)
    # Constants combined: q / (h * c) * conversion factors
    spectral_jsc = am15g * wavelengths * 8.0655e-5
    
    # Cumulative numerical integration (Trapezoidal rule)
    cum_jsc = np.concatenate(([0], np.cumsum((spectral_jsc[1:] + spectral_jsc[:-1]) / 2.0 * np.diff(wavelengths))))
    
    return wavelengths, cum_jsc

wavelengths, cum_jsc = load_spectrum()

# Function to get exact optical Jsc limit from the spectrum
def get_jsc_limit(eg):
    # Convert Bandgap (eV) to Wavelength (nm)
    lambda_g = 1240.0 / eg
    # Interpolate cumulative integral for the exact cutoff wavelength
    return np.interp(lambda_g, wavelengths, cum_jsc)

# --- Sidebar Inputs ---
st.sidebar.header("1. Top Cell (Perovskite)")
top_eg = st.sidebar.slider("Bandgap (eV)", 1.40, 1.80, 1.60, 0.01)

# Calculate true optical limit based on uploaded spectrum
theoretical_jsc = get_jsc_limit(top_eg)
default_jsc = round(theoretical_jsc * 0.90, 2)
st.sidebar.caption(f"Theoretical AM1.5G Optical Jsc limit: {theoretical_jsc:.2f} mA/cm²")

top_jsc = st.sidebar.number_input("Actual Jsc (mA/cm²)", value=float(default_jsc), step=0.1)
top_voc = st.sidebar.number_input("Actual Voc (V)", value=float(round(top_eg - 0.4, 2)), step=0.01)

st.sidebar.header("2. Bottom Cell (Silicon)")
si_type = st.sidebar.selectbox("Silicon Technology", ["PERC", "TOPCon", "HJT"], index=2)
albedo = st.sidebar.slider("Ground Albedo (%)", 0, 50, 20, 1)

# --- Silicon Physics Dictionary ---
si_params = {
    "PERC": {"jsc_base": 39.5, "voc": 0.69, "bifi": 0.70},
    "TOPCon": {"jsc_base": 41.5, "voc": 0.72, "bifi": 0.80},
    "HJT": {"jsc_base": 40.0, "voc": 0.74, "bifi": 0.87}
}

si_jsc = si_params[si_type]["jsc_base"]
si_voc = si_params[si_type]["voc"]
si_bifi = si_params[si_type]["bifi"]

st.sidebar.info(f"**{si_type} Baseline Parameters:**\n\nFront Jsc: {si_jsc} mA/cm²\n\nVoc: {si_voc} V\n\nBifaciality Factor: {int(si_bifi*100)}%")

# --- Calculations for Current Config ---
# True optical absorption based on spectrum
optical_absorption = get_jsc_limit(top_eg)

# Transmitted light available to the bottom cell
j_bot_mono = si_jsc - optical_absorption
bifi_boost = (albedo / 100.0) * si_jsc * si_bifi
j_bot_bifi = j_bot_mono + bifi_boost

j_tandem_mono = min(top_jsc, j_bot_mono)
j_tandem_bifi = min(top_jsc, j_bot_bifi)

v_tandem = top_voc + si_voc
ff = 0.80  # Assumed static Fill Factor

eff_mono = j_tandem_mono * v_tandem * ff
eff_bifi = j_tandem_bifi * v_tandem * ff

# --- Top Row: Metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Operating Tandem Jsc", f"{j_tandem_bifi:.2f} mA/cm²", help="Current bottleneck in bifacial operation.")
m2.metric("Operating Tandem Voc", f"{v_tandem:.2f} V")
m3.metric("Monofacial Efficiency", f"{eff_mono:.1f}%")
m4.metric("Bifacial Efficiency", f"{eff_bifi:.1f}%")

st.divider()

# --- Middle Row: Bar Charts ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sub-Cell Current Densities")
    curr_data = pd.DataFrame({
        "Mode": ["Top Cell (PVSK)", "Bottom (Si Mono)", "Bottom (Si Bifi)"],
        "mA/cm²": [top_jsc, j_bot_mono, j_bot_bifi]
    })
    st.bar_chart(curr_data, x="Mode", y="mA/cm²", color="#4169E1")

with col_b:
    st.subheader("Efficiency Output")
    eff_data = pd.DataFrame({
        "Configuration": ["Monofacial", "Bifacial"],
        "Efficiency (%)": [eff_mono, eff_bifi]
    })
    st.bar_chart(eff_data, x="Configuration", y="Efficiency (%)", color="#FFD700")

st.divider()

# --- Bottom Row: Sensitivity Curves (True Spectrum) ---
st.subheader("Performance Landscape Across Bandgaps (AM1.5G Spectrum)")
st.write("These curves are now derived directly from the integrated AM1.5G dataset. Notice the non-linear atmospheric absorption features.")

eqe_factor = top_jsc / theoretical_jsc if theoretical_jsc > 0 else 0
voc_deficit = top_eg - top_voc

eg_range = np.linspace(1.4, 1.8, 60) # Increased resolution to see spectrum details
j_top_vals, j_bot_mono_vals, j_bot_bifi_vals = [], [], []
eff_mono_vals, eff_bifi_vals = [], []

for e in eg_range:
    # Use exact spectrum integral for sweep
    opt = get_jsc_limit(e)
    jt = opt * eqe_factor
    jbm = si_jsc - opt
    jbb = jbm + ((albedo / 100.0) * si_jsc * si_bifi)
    
    vt = (e - voc_deficit) + si_voc
    
    j_top_vals.append(jt)
    j_bot_mono_vals.append(jbm)
    j_bot_bifi_vals.append(jbb)
    eff_mono_vals.append(min(jt, jbm) * vt * ff)
    eff_bifi_vals.append(min(jt, jbb) * vt * ff)

col_c, col_d = st.columns(2)

with col_c:
    fig_jsc = go.Figure()
    fig_jsc.add_trace(go.Scatter(x=eg_range, y=j_top_vals, name="Top Cell Jsc", line=dict(color='black', width=3)))
    fig_jsc.add_trace(go.Scatter(x=eg_range, y=j_bot_mono_vals, name="Bottom Mono Jsc", line=dict(dash='dash')))
    fig_jsc.add_trace(go.Scatter(x=eg_range, y=j_bot_bifi_vals, name="Bottom Bifi Jsc", line=dict(color='red')))
    fig_jsc.add_vline(x=top_eg, line_width=2, line_dash="dot", line_color="green", annotation_text="Your Config")
    fig_jsc.update_layout(title="Current Matching Limits", xaxis_title="Top Cell Bandgap (eV)", yaxis_title="Current Density (mA/cm²)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_jsc, use_container_width=True)

with col_d:
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(x=eg_range, y=eff_mono_vals, name="Monofacial Eff.", line=dict(color='blue', width=3)))
    fig_eff.add_trace(go.Scatter(x=eg_range, y=eff_bifi_vals, name="Bifacial Eff.", line=dict(color='orange', width=3)))
    fig_eff.add_vline(x=top_eg, line_width=2, line_dash="dot", line_color="green", annotation_text="Your Config")
    fig_eff.update_layout(title="Efficiency vs. Bandgap", xaxis_title="Top Cell Bandgap (eV)", yaxis_title="Efficiency (%)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_eff, use_container_width=True)
