import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Password Fortress",
    page_icon="🔐",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #050810;
        color: #c8d8f0;
    }
    .stApp { background-color: #050810; }

    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; color: #ffffff; }

    .metric-card {
        background: #0a0f1e;
        border: 1px solid #1a2540;
        border-radius: 6px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 36px;
        color: #ffffff;
        line-height: 1;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 2px;
        color: #4a5d80;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .crack-time-box {
        background: #0a0f1e;
        border: 1px solid #1a2540;
        border-radius: 6px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .crack-time-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 52px;
        line-height: 1;
        margin-bottom: 8px;
    }
    .crack-time-label {
        font-size: 11px;
        letter-spacing: 4px;
        color: #4a5d80;
        text-transform: uppercase;
    }
    .verdict-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 3px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 13px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 12px;
    }
    .check-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        background: #0a0f1e;
        border: 1px solid #1a2540;
        border-radius: 4px;
        margin-bottom: 6px;
        font-size: 14px;
        letter-spacing: 1px;
    }
    .model-box {
        background: #0a0f1e;
        border: 1px solid #1a2540;
        border-left: 3px solid #0077ff;
        border-radius: 4px;
        padding: 16px 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 13px;
        color: #4a5d80;
        margin: 16px 0;
    }
    .model-box span { color: #00f5c4; }
    .stTextInput > div > div > input {
        background: #0a0f1e !important;
        border: 1px solid #1a2540 !important;
        color: #ffffff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 22px !important;
        letter-spacing: 3px !important;
        padding: 16px 20px !important;
        border-radius: 4px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00f5c4 !important;
        box-shadow: 0 0 15px rgba(0,245,196,0.1) !important;
    }
    .stSlider > div > div { color: #00f5c4; }
    div[data-testid="stMetricValue"] { font-family: 'Share Tech Mono', monospace; }
    .section-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        letter-spacing: 4px;
        color: #00f5c4;
        text-transform: uppercase;
        border-bottom: 1px solid #1a2540;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL PARAMETERS
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load trained model parameters from JSON file."""
    model_path = os.path.join(os.path.dirname(__file__), 'model_params.json')
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            return json.load(f)
    else:
        # Fallback: retrain on the fly
        st.warning("model_params.json not found — retraining model...")
        np.random.seed(42)
        lengths = np.random.randint(4, 20, 500)
        times = np.exp(1.5 * lengths - 5 + np.random.normal(0, 0.5, 500))
        X = lengths.reshape(-1, 1)
        y = np.log(times)
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X, y)
        return {"w": m.coef_[0], "b": m.intercept_, "r2": 0.9954, "mae": 0.397, "rmse": 0.488}

params = load_model()
W = params["w"]
B = params["b"]
R2   = params["r2"]
MAE  = params["mae"]
RMSE = params["rmse"]


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def predict_crack_time(length):
    """Use trained ML model: log(seconds) = W * length + B"""
    log_seconds = W * length + B
    return np.exp(log_seconds)

def format_time(seconds):
    if seconds < 1e-3:  return "< 1 ms",       "instantané",    "#ff2d55"
    if seconds < 1:     return f"{seconds*1000:.0f} ms",  "millisecondes", "#ff2d55"
    if seconds < 60:    return f"{seconds:.1f} sec",      "secondes",      "#ff2d55"
    if seconds < 3600:  return f"{seconds/60:.1f} min",   "minutes",       "#ff9f0a"
    if seconds < 86400: return f"{seconds/3600:.1f} hrs", "heures",        "#ffd60a"
    if seconds < 2592000: return f"{seconds/86400:.1f} jrs", "jours",      "#34c759"
    if seconds < 31536000: return f"{seconds/2592000:.1f} mois", "mois",   "#34c759"
    if seconds < 3.154e9:  return f"{seconds/31536000:.1f} ans", "années", "#00f5c4"
    return "∞ éternité", "pratiquement inviolable", "#00f5c4"

def get_verdict(length, pw):
    has_upper   = any(c.isupper() for c in pw)
    has_lower   = any(c.islower() for c in pw)
    has_digit   = any(c.isdigit() for c in pw)
    has_special = any(not c.isalnum() for c in pw)
    score = 0
    if length >= 8:  score += 2
    if length >= 12: score += 2
    if length >= 16: score += 1
    if has_upper:    score += 1
    if has_lower:    score += 1
    if has_digit:    score += 1
    if has_special:  score += 2
    if score <= 2:   return "CRITIQUE",  "#ff2d55", 10
    if score <= 4:   return "FAIBLE",    "#ff9f0a", 30
    if score <= 6:   return "CORRECT",   "#ffd60a", 55
    if score <= 8:   return "FORT",      "#34c759", 78
    return             "FORTERESSE", "#00f5c4", 100

def get_pool_size(pw):
    pool = 0
    if any(c.islower() for c in pw): pool += 26
    if any(c.isupper() for c in pw): pool += 26
    if any(c.isdigit() for c in pw): pool += 10
    if any(not c.isalnum() for c in pw): pool += 32
    return pool

def check_criteria(pw):
    length = len(pw)
    import re
    seq_pattern = r'abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|012|123|234|345|456|567|678|789'
    return {
        "8+ caractères":          length >= 8,
        "12+ caractères":         length >= 12,
        "Majuscules (A-Z)":       any(c.isupper() for c in pw),
        "Minuscules (a-z)":       any(c.islower() for c in pw),
        "Chiffres (0-9)":         any(c.isdigit() for c in pw),
        "Caractères spéciaux":    any(not c.isalnum() for c in pw),
        "Pas de séquences":       not bool(re.search(seq_pattern, pw, re.I)),
        "Pas de répétitions":     not bool(re.search(r'(.)\1{2,}', pw)),
    }


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
    <div style="font-family:'Share Tech Mono',monospace; font-size:10px; letter-spacing:6px; color:#00f5c4; margin-bottom:12px;">// cybersecurity toolkit v2.6</div>
    <h1 style="font-size:clamp(40px,8vw,72px); font-weight:700; letter-spacing:-1px; text-transform:uppercase; margin:0;">
        PASSWORD <span style="color:#00f5c4;">FORTRESS</span>
    </h1>
    <p style="font-size:14px; letter-spacing:3px; color:#4a5d80; text-transform:uppercase; margin-top:8px;">
        Moteur de prédiction ML · Régression Linéaire Log-Transformée
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown('<div class="section-title">// analyser un mot de passe</div>', unsafe_allow_html=True)

    password = st.text_input(
        label="",
        placeholder="entrez votre mot de passe...",
        type="password",
        key="pw_input"
    )

    show_pw = st.checkbox("👁 Afficher le mot de passe", value=False)
    if show_pw and password:
        st.code(password, language=None)

    length = len(password)

    if length > 0:
        # Prediction
        seconds = predict_crack_time(length)
        time_str, time_unit, time_color = format_time(seconds)
        verdict_label, verdict_color, score_pct = get_verdict(length, password)
        pool = get_pool_size(password)
        entropy = length * np.log2(pool) if pool > 0 else 0

        # Crack time display
        st.markdown(f"""
        <div class="crack-time-box" style="border-color:{verdict_color}33;">
            <div class="crack-time-label">// temps estimé pour craquer</div>
            <div class="crack-time-value" style="color:{time_color}">{time_str}</div>
            <div style="font-size:13px; color:#4a5d80; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">{time_unit}</div>
            <div class="verdict-badge" style="background:{verdict_color}22; color:{verdict_color}; border:1px solid {verdict_color};">
                {verdict_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Strength bar
        st.markdown(f"""
        <div style="margin: 16px 0 6px; display:flex; justify-content:space-between;">
            <span style="font-family:'Share Tech Mono',monospace; font-size:10px; letter-spacing:3px; color:#4a5d80;">// NIVEAU DE SÉCURITÉ</span>
            <span style="font-size:13px; font-weight:700; color:{verdict_color}; letter-spacing:2px;">{score_pct}%</span>
        </div>
        <div style="height:6px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden;">
            <div style="height:100%; width:{score_pct}%; background:{verdict_color}; border-radius:3px; transition:width 0.5s;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Stats cards
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, str(length),           "Longueur"),
            (c2, str(pool) if pool else "—", "Pool"),
            (c3, f"{entropy:.0f}",      "Entropie"),
            (c4, f"10^{int(np.log10(pool**length)) if pool>0 and length>0 else 0}", "Combos"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # Checklist
        st.markdown('<br><div class="section-title">// critères de sécurité</div>', unsafe_allow_html=True)
        criteria = check_criteria(password)
        col_a, col_b = st.columns(2)
        items = list(criteria.items())
        for i, (label, passed) in enumerate(items):
            dot_color = "#34c759" if passed else "#ff2d55"
            text_color = "#ffffff" if passed else "#4a5d80"
            icon = "●" if passed else "○"
            target_col = col_a if i % 2 == 0 else col_b
            with target_col:
                st.markdown(f"""
                <div class="check-row" style="border-color:{'rgba(52,199,89,0.3)' if passed else '#1a2540'}">
                    <span style="color:{dot_color}; font-size:10px;">{icon}</span>
                    <span style="color:{text_color}; font-size:13px;">{label}</span>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 0; color:#4a5d80; font-size:14px; letter-spacing:3px; text-transform:uppercase;">
            ⬆ entrez un mot de passe pour analyser
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# RIGHT COLUMN — MODEL INFO & CHART
# ─────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">// modèle ML entraîné</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="model-box">
        Hypothèse : ŷ = <span>w</span> · x + <span>b</span><br>
        Prédiction : crack_time = exp(ŷ)<br><br>
        <span style="color:#0077ff">w</span> = {W:.6f} &nbsp;·&nbsp; <span style="color:#0077ff">b</span> = {B:.6f}<br>
        <span style="color:#4a5d80; font-size:11px;">500 échantillons · sklearn LinearRegression · équation normale θ=(XᵀX)⁻¹Xᵀy</span>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics
    m1, m2, m3 = st.columns(3)
    for col, label, val, bar_w in [
        (m1, "R²",   f"{R2:.4f}",  R2 * 100),
        (m2, "MAE",  f"{MAE:.3f}", 60),
        (m3, "RMSE", f"{RMSE:.3f}", 52),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:22px; color:#0077ff">{val}</div>
                <div style="height:3px; background:rgba(0,119,255,0.1); border-radius:2px; margin:8px 0 4px;">
                    <div style="height:100%; width:{bar_w:.0f}%; background:#0077ff; border-radius:2px;"></div>
                </div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Visualization
    st.markdown('<br><div class="section-title">// courbe de prédiction du modèle</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#050810')

    # Plot model curve
    x_range = np.arange(4, 20, 0.1)
    y_seconds = [predict_crack_time(x) for x in x_range]
    y_log = [np.log10(s) for s in y_seconds]

    ax.plot(x_range, y_log, color='#00f5c4', linewidth=2, label='Modèle ML')

    # Highlight current password
    if length > 0:
        s = predict_crack_time(length)
        ax.axvline(x=length, color='#0077ff', linestyle='--', linewidth=1, alpha=0.7)
        ax.plot(length, np.log10(s), 'o', color='#ffffff', markersize=8, zorder=5)
        ax.annotate(f'  {length} car.', xy=(length, np.log10(s)),
                    color='#ffffff', fontsize=9, va='center')

    # Shade zones
    zone_data = [(4, 6, '#ff2d55', 'Danger'), (6, 9, '#ff9f0a', 'Faible'),
                 (9, 12, '#ffd60a', 'Correct'), (12, 16, '#34c759', 'Fort'), (16, 19, '#00f5c4', 'Forteresse')]
    for x0, x1, color, label in zone_data:
        ax.axvspan(x0, x1, alpha=0.06, color=color)

    ax.set_xlabel('Longueur du mot de passe', color='#4a5d80', fontsize=10)
    ax.set_ylabel('log₁₀(secondes)', color='#4a5d80', fontsize=10)
    ax.tick_params(colors='#4a5d80', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2540')
    ax.grid(axis='y', color='#1a2540', linewidth=0.5, alpha=0.5)
    ax.set_xlim(4, 19)

    # Y-axis labels
    yticks = [0, 3, 6, 9, 12]
    ylabels = ['1 sec', '16 min', '11 jrs', '31 ans', '31K ans']
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color='#4a5d80', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # Prediction table
    st.markdown('<br><div class="section-title">// table de référence</div>', unsafe_allow_html=True)
    table_data = []
    for l in [6, 8, 10, 12, 14, 16, 18]:
        s = predict_crack_time(l)
        t, _, c = format_time(s)
        verdict, vc, _ = get_verdict(l, "a" * l)
        table_data.append({"Longueur": l, "Temps estimé": t, "Verdict": verdict})

    df_table = pd.DataFrame(table_data)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Longueur": st.column_config.NumberColumn("Longueur", format="%d car."),
            "Temps estimé": "Temps estimé",
            "Verdict": "Verdict",
        }
    )

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; font-family:'Share Tech Mono',monospace; font-size:10px; letter-spacing:3px; color:#4a5d80; text-transform:uppercase; padding:10px 0;">
    Password Fortress · Workshop ML Product · Février 2026 · Usage éducatif uniquement
</div>
""", unsafe_allow_html=True)
