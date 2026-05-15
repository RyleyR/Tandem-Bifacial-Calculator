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
    file_path = "ASTMG173.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: Could not find '{file_path}'. Please ensure it is in the same folder as app.py.")
        st.stop()
        
    df = pd.read_csv(file_path, skiprows=1)
    wavelengths = df.iloc[:, 0].values  
    am15g = df.iloc[:, 2].values        
    
    spectral_jsc = am15g * wavelengths * 8.0655e-5
    cum_jsc = np.concatenate(([0], np.cumsum((spectral_jsc[1:] + spectral_jsc[:-1]) / 2.0 * np.diff(wavelengths))))
    
    return wavelengths, cum_jsc

wavelengths, cum_jsc = load_spectrum()

def get_jsc_limit(eg):
    lambda_g = 1240.0 / eg
    return np.interp(lambda_g, wavelengths, cum_jsc)

# Absolute AM1.5G limit for Silicon
si_max_jsc = get_jsc_limit(1.10) 

# --- Sidebar Inputs ---
st.sidebar.header("1. Top Cell (Perovskite)")
top_eg = st.sidebar.slider("Bandgap (eV)", 1.40, 1.80, 1.60, 0.01)

# New Top Cell Multiplier
top_eqe = st.sidebar.slider("Top Cell EQE Multiplier (%)", 50.0, 100.0, 90.0, 0.1) / 100.0

theoretical_jsc = get_jsc_limit(top_eg)
top_jsc = theoretical_jsc * top_eqe

st.sidebar.caption(f"Theoretical Limit: {theoretical_jsc:.2f} mA/cm²")
st.sidebar.caption(f"After Multiplier: {top_jsc:.2f} mA/cm²")

top_voc = st.sidebar.number_input("Actual Voc (V)", value=float(round(top_eg - 0.4, 2)), step=0.01)

st.sidebar.header("2. Bottom Cell (Silicon)")
si_type = st.sidebar.selectbox("Silicon Technology", ["PERC", "TOPCon", "HJT"], index=2)

# --- Silicon Physics Dictionary ---
# (Preserved your custom baseline tweaks)
si_params = {
    "PERC": {"default_ir_eqe": 0.92, "voc": 0.69, "bifi": 0.70, "rear_eqe": 0.90},
    "TOPCon": {"default_ir_eqe": 0.95, "voc": 0.72, "bifi": 0.80, "rear_eqe": 0.92},
    "HJT": {"default_ir_eqe": 0.92, "voc": 0.74, "bifi": 0.87, "rear_eqe": 0.95}
}

# New Bottom Cell Multiplier (Defaults dynamically based on Si Type)
default_ir = si_params[si_type]["default_ir_eqe"] * 100.0
si_ir_eqe = st.sidebar.slider("Bottom Cell IR-EQE Multiplier (%)", 50.0, 100.0, default_ir, 0.1) / 100.0

albedo = st.sidebar.slider("Ground Albedo (%)", 0, 50, 20, 1)

si_voc = si_params[si_type]["voc"]
si_bifi = si_params[si_type]["bifi"]
si_rear_eqe = si_params[si_type]["rear_eqe"]

st.sidebar.info(f"**{si_type} Baseline Parameters:**\n\nVoc: {si_voc} V\n\nBifaciality Factor: {int(si_bifi*100)}%\n\nRear EQE: {int(si_rear_eqe*100)}%")

# --- 3. Fill Factor Dynamics ---
st.sidebar.header("3. Fill Factor Dynamics")
base_ff = st.sidebar.slider("Baseline Fill Factor (%)", 60.0, 90.0, 80.0, 0.1) / 100.0
ff_penalty = st.sidebar.slider("Current Matching Penalty (%)", 0.0, 10.0, 5.0, 0.1) / 100.0
st.sidebar.caption("Simulates the drop in FF when Jtop and Jbot are perfectly matched.")

def calculate_dynamic_ff(j_top, j_bot, base, penalty):
    # Gaussian penalty curve based on current mismatch
    mismatch = abs(j_top - j_bot)
    dip = penalty * np.exp(- (mismatch**2) / 3.0)
    return base - dip

# --- Calculations for Current Config ---
optical_absorption = get_jsc_limit(top_eg)
transmitted_ir_light = si_max_jsc - optical_absorption

# Applying the explicit multipliers
j_bot_mono = transmitted_ir_light * si_ir_eqe
bifi_boost = (albedo / 100.0) * si_max_jsc * si_bifi * si_rear_eqe
j_bot_bifi = j_bot_mono + bifi_boost

j_tandem_mono = min(top_jsc, j_bot_mono)
j_tandem_bifi = min(top_jsc, j_bot_bifi)

v_tandem = top_voc + si_voc

# Dynamic Fill Factor calculation for current config
ff_mono_actual = calculate_dynamic_ff(top_jsc, j_bot_mono, base_ff, ff_penalty)
ff_bifi_actual = calculate_dynamic_ff(top_jsc, j_bot_bifi, base_ff, ff_penalty)

eff_mono = j_tandem_mono * v_tandem * ff_mono_actual
eff_bifi = j_tandem_bifi * v_tandem * ff_bifi_actual

# --- Top Row: Metrics ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Op. Tandem Jsc", f"{j_tandem_bifi:.2f} mA/cm²")
m2.metric("Op. Tandem Voc", f"{v_tandem:.2f} V")
m3.metric("Bifacial FF", f"{ff_bifi_actual*100:.1f}%")
m4.metric("Monofacial Eff.", f"{eff_mono:.1f}%")
m5.metric("Bifacial Eff.", f"{eff_bifi:.1f}%")

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
st.subheader("Performance Landscape Across Bandgaps")
st.write("Notice the smooth 'dent' at the peak of the efficiency curves where the Fill Factor drops due to exact current matching.")

voc_deficit = top_eg - top_voc
eg_range = np.linspace(1.4, 1.8, 80) # Increased resolution to make the FF dent smoother
j_top_vals, j_bot_mono_vals, j_bot_bifi_vals = [], [], []
eff_mono_vals, eff_bifi_vals = [], []

for e in eg_range:
    opt = get_jsc_limit(e)
    
    # Sweep applying the user multipliers
    jt = opt * top_eqe
    trans_ir = si_max_jsc - opt
    jbm = trans_ir * si_ir_eqe
    jbb = jbm + ((albedo / 100.0) * si_max_jsc * si_bifi * si_rear_eqe)
    
    vt = (e - voc_deficit) + si_voc
    
    # Dynamic Fill Factor logic for the sweep
    ff_m = calculate_dynamic_ff(jt, jbm, base_ff, ff_penalty)
    ff_b = calculate_dynamic_ff(jt, jbb, base_ff, ff_penalty)
    
    j_top_vals.append(jt)
    j_bot_mono_vals.append(jbm)
    j_bot_bifi_vals.append(jbb)
    eff_mono_vals.append(min(jt, jbm) * vt * ff_m)
    eff_bifi_vals.append(min(jt, jbb) * vt * ff_b)

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
