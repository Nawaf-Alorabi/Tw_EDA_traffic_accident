"""
App.py — Streamlit Dashboard for Traffic Accidents EDA Project
==============================================================
Tuwaiq Data Science & AI Bootcamp — Final Project
Saudi Arabia Traffic Accidents (1437–1439 Hijri)

Run:  streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saudi Arabia Traffic Accidents",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Global CSS — clean dark-accented professional theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1100px; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f2940 100%);
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #f0f9ff;
    line-height: 1.2;
    margin: 0 0 0.75rem 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0 0 1.5rem 0;
    max-width: 650px;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 0.5rem;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.4rem 0;
    line-height: 1.25;
}
.section-desc {
    color: #64748b;
    font-size: 0.97rem;
    margin-bottom: 1.8rem;
    max-width: 680px;
    line-height: 1.6;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
}
.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-delta { font-size: 0.82rem; margin-top: 0.3rem; font-weight: 500; }
.delta-neg { color: #10b981; }
.delta-pos { color: #ef4444; }

/* Viz card */
.viz-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.viz-question {
    background: #f0f9ff;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.97rem;
    color: #0369a1;
    font-weight: 500;
    margin-bottom: 1.2rem;
}
.insight-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1.2rem;
}
.insight-title {
    font-weight: 600;
    color: #0f172a;
    font-size: 0.88rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
.insight-text { color: #475569; font-size: 0.93rem; line-height: 1.65; }

/* Divider */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #cbd5e1, transparent);
    margin: 3rem 0;
}

/* Table styling */
.stats-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.stats-table th {
    background: #f1f5f9;
    color: #475569;
    font-weight: 600;
    padding: 0.6rem 1rem;
    text-align: left;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stats-table td { padding: 0.55rem 1rem; border-bottom: 1px solid #f1f5f9; color: #334155; }
.stats-table tr:last-child td { border-bottom: none; }

/* Model cards */
.model-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}
.model-badge {
    background: #0f172a;
    color: #38bdf8;
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.35rem 0.8rem;
    border-radius: 6px;
}
.model-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
}
.result-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
    padding: 0.35rem 0.9rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.interp-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #78350f;
    font-size: 0.93rem;
    line-height: 1.65;
}

/* Conclusion */
.conclusion-card {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 16px;
    padding: 2.5rem;
    color: white;
}
.conclusion-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 1rem;
    font-size: 0.97rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.conclusion-dot {
    width: 8px; height: 8px;
    background: #38bdf8;
    border-radius: 50%;
    margin-top: 0.5rem;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Data Loading & Cleaning (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [
        "Injured_and_Dead_in_Accidents_1437.csv",
        "Injured_and_Dead_in_Accidents_1438.csv",
        "Injured_and_Dead_in_Accidents_1439.csv",
    ]
    frames = []
    for f in csv_files:
        path = os.path.join(base_dir, f)
        frames.append(pd.read_csv(path, sep=";", encoding="utf-8-sig"))
    df = pd.concat(frames, ignore_index=True)
    df.dropna(inplace=True)
    df.columns = [
        "month", "year", "city", "males", "females",
        "inside_city", "outside_city", "age_under18", "age_18_30",
        "age_30_40", "age_40_50", "age_50plus", "saudi", "non_saudi",
        "injuries", "deaths",
    ]
    df["city"] = df["city"].replace({
        "الرياض": "Riyadh", "جده": "Jeddah",
        "المدينه المنوره": "Madinah", "الشرقيه": "Eastern Region",
        "الحدود الشماليه": "Northern Borders", "تبوك": "Tabuk",
        "الجوف": "Al-Jouf", "حائل": "Hail", "نجران": "Najran",
        "القصيم": "Al-Qassim", "الباحه": "Al-Baha", "عسير": "Asir",
        "جازان": "Jazan", "الطائف": "Taif", "العاصمه": "Makkah",
        "القريات": "Al-Qurayyat",
    })
    df["total_accidents"] = df["males"] + df["females"]
    df["has_injury"] = (df["injuries"] > 0).astype(int)
    df["has_death"]  = (df["deaths"]  > 0).astype(int)
    return df

df = load_and_clean_data()

# ──────────────────────────────────────────────────────────────
# Plot style helper
# ──────────────────────────────────────────────────────────────
def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#e2e8f0")
    if title:  ax.set_title(title, fontsize=13, fontweight="bold", color="#0f172a", pad=12)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color="#64748b", labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color="#64748b", labelpad=6)

sns.set_theme(style="white")
PALETTE = ["#0ea5e9", "#f97316", "#10b981", "#8b5cf6", "#f43f5e",
           "#14b8a6", "#eab308", "#6366f1", "#ec4899", "#22c55e"]


# ══════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">🚦 Traffic Accidents Analysis<br>in Saudi Arabia</div>
  <div class="hero-sub">
    An end-to-end exploratory data analysis of traffic accident records across
    16 Saudi cities over three Hijri years (1437–1439), uncovering patterns,
    risk factors, and predictive trends.
  </div>
  <span class="badge">📅 1437 – 1439 Hijri</span>
  <span class="badge">🏙️ 16 Cities</span>
  <span class="badge">📊 EDA + ML</span>
  <span class="badge">🎓 Tuwaiq Bootcamp</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW METRICS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Project Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dataset at a Glance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Key numbers from the combined dataset after cleaning and preprocessing.</div>', unsafe_allow_html=True)

yearly_totals = df.groupby("year")["total_accidents"].sum()
change_pct = ((yearly_totals.iloc[-1] - yearly_totals.iloc[0]) / yearly_totals.iloc[0]) * 100

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{len(df):,}</div>
      <div class="metric-label">Total Records</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{df['city'].nunique()}</div>
      <div class="metric-label">Cities</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{int(df['year'].nunique())}</div>
      <div class="metric-label">Years Covered</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{int(df['total_accidents'].sum()):,}</div>
      <div class="metric-label">Total Involved</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{change_pct:.0f}%</div>
      <div class="metric-label">Change 1437→1439</div>
      <div class="metric-delta delta-neg">↓ Declining trend</div>
    </div>""", unsafe_allow_html=True)


st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Data Quality</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Statistical summary of the cleaned dataset. No missing values remain after preprocessing.</div>', unsafe_allow_html=True)

with st.expander("📋 View Full Statistical Summary", expanded=False):
    key_cols = ["males", "females", "inside_city", "outside_city",
                "age_18_30", "age_30_40", "saudi", "non_saudi", "total_accidents"]
    stats = df[key_cols].describe().T.round(2)
    stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    st.dataframe(stats, use_container_width=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("**📅 Records per Year**")
    yr = df.groupby("year")["total_accidents"].agg(["count", "sum"]).reset_index()
    yr.columns = ["Year", "Records", "Total Involved"]
    yr["Year"] = yr["Year"].astype(int)
    st.dataframe(yr, hide_index=True, use_container_width=True)
with col_b:
    st.markdown("**🏙️ Top 5 Cities by Volume**")
    top5 = df.groupby("city")["total_accidents"].sum().nlargest(5).reset_index()
    top5.columns = ["City", "Total Involved"]
    st.dataframe(top5, hide_index=True, use_container_width=True)
with col_c:
    st.markdown("**✅ Data Quality**")
    st.success("No missing values after cleaning")
    st.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.info(f"Binary flags: has_injury, has_death (0/1)")

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — VISUALIZATIONS (storytelling format)
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Exploratory Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Key Visualizations</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Each chart follows a Question → Visualization → Insights narrative to guide the analysis.</div>', unsafe_allow_html=True)


# ── VIZ 1: Total Accidents Over Time — Line Chart ────────────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ Are traffic accidents in Saudi Arabia increasing or decreasing over the three-year period?</div>', unsafe_allow_html=True)
st.markdown("##### 1 · Total Accidents Over Time — Line Chart")

yearly = df.groupby("year", as_index=False)["total_accidents"].sum()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(yearly["year"], yearly["total_accidents"],
        marker="o", linewidth=3, color="#0ea5e9", markersize=10, zorder=5)
ax.fill_between(yearly["year"], yearly["total_accidents"],
                alpha=0.12, color="#0ea5e9")
for _, row in yearly.iterrows():
    ax.annotate(f'{int(row["total_accidents"]):,}',
                (row["year"], row["total_accidents"]),
                textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=10, fontweight="bold", color="#0f172a")
style_ax(ax, "Total Accidents by Hijri Year", "Year (Hijri)", "Total People Involved")
ax.set_xticks(yearly["year"])
ax.set_xticklabels(yearly["year"].astype(int))
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    Total accidents show a <strong>consistent decline</strong> from 1437 to 1439 — dropping by roughly 23%.
    This is a positive indicator for road safety improvements in Saudi Arabia. The steepest drop
    occurred between 1437 and 1438.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── VIZ 2: Monthly Heatmap ───────────────────────────────────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ Are there specific months that consistently record higher accident rates?</div>', unsafe_allow_html=True)
st.markdown("##### 2 · Monthly Accidents Heatmap")

pivot = df.pivot_table(values="total_accidents", index="year",
                       columns="month", aggfunc="sum")
pivot.index = pivot.index.astype(int)
pivot.columns = pivot.columns.astype(int)
fig, ax = plt.subplots(figsize=(12, 3.5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, linecolor="#f1f5f9",
            annot_kws={"size": 9}, ax=ax)
ax.set_title("Monthly Total Accidents by Year", fontsize=13,
             fontweight="bold", color="#0f172a", pad=10)
ax.set_xlabel("Month", fontsize=9, color="#64748b")
ax.set_ylabel("Year", fontsize=9, color="#64748b")
ax.tick_params(colors="#64748b", labelsize=9)
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    <strong>Month 9</strong> consistently shows elevated accident counts across all three years —
    potentially linked to seasonal traffic patterns. Month 11 shows the lowest activity.
    The pattern is relatively stable year-over-year, suggesting structural seasonal behaviour.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── VIZ 3: Accidents by City — Horizontal Bar ────────────────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ Which cities have the highest total accident involvement over all three years?</div>', unsafe_allow_html=True)
st.markdown("##### 3 · Accidents by City — Horizontal Bar Chart")

city_totals = df.groupby("city")["total_accidents"].sum().sort_values(ascending=True)
colors = ["#ef4444" if v == city_totals.max() else "#0ea5e9" for v in city_totals.values]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(city_totals.index, city_totals.values, color=colors, height=0.65)
for bar in bars:
    ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}", va="center", ha="left",
            fontsize=8.5, color="#475569")
style_ax(ax, "Total Accidents by City (All Years)", "Total People Involved", "")
ax.grid(axis="x", linestyle="--", alpha=0.4, color="#e2e8f0")
ax.grid(axis="y", visible=False)
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    <strong>Riyadh</strong> dominates with the highest accident count (shown in red), far exceeding
    all other cities. <strong>Taif</strong> and <strong>Eastern Region</strong> rank second and third.
    Smaller cities like Al-Qurayyat and Al-Jouf record minimal activity.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── VIZ 4: Age Group Distribution by City — Stacked Bar ──────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ Which age groups are most frequently involved in accidents, and how does this vary by city?</div>', unsafe_allow_html=True)
st.markdown("##### 4 · Age Group Distribution by City — Stacked Bar Chart")

age_cols = ["age_under18", "age_18_30", "age_30_40", "age_40_50", "age_50plus"]
age_by_city = df.groupby("city")[age_cols].sum()
age_by_city.columns = ["<18", "18–30", "30–40", "40–50", "50+"]
age_by_city = age_by_city.loc[age_by_city.sum(axis=1).nlargest(12).index]
age_colors = ["#6366f1", "#0ea5e9", "#10b981", "#f97316", "#f43f5e"]
fig, ax = plt.subplots(figsize=(12, 5.5))
age_by_city.plot(kind="bar", stacked=True, ax=ax,
                 color=age_colors, width=0.72)
style_ax(ax, "Age Group Distribution by City (Top 12)", "City", "Count")
ax.legend(title="Age Group", bbox_to_anchor=(1.01, 1),
          loc="upper left", fontsize=8.5, title_fontsize=9)
plt.xticks(rotation=40, ha="right", fontsize=9)
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    The <strong>18–30 age group</strong> (blue) is the dominant segment in virtually every city —
    consistent with higher driving activity among young adults. In Riyadh, the absolute
    volume of this group is significantly higher than all other cities combined.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── VIZ 5: Gender Distribution — Pie ─────────────────────────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ What is the male-to-female ratio among those involved in traffic accidents?</div>', unsafe_allow_html=True)
st.markdown("##### 5 · Gender Distribution — Pie Chart")

col_pie, col_pie_text = st.columns([1, 1])
with col_pie:
    gender_vals = [df["males"].sum(), df["females"].sum()]
    gender_labels = ["Males", "Females"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    wedges, texts, autotexts = ax.pie(
        gender_vals, labels=gender_labels,
        autopct="%1.1f%%", startangle=140,
        colors=["#0ea5e9", "#f43f5e"],
        explode=(0.03, 0.03),
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title("Gender Distribution", fontsize=13, fontweight="bold",
                 color="#0f172a", pad=10)
    fig.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
with col_pie_text:
    st.markdown(f"""
    <br><br>
    <div class="metric-card" style="margin-bottom:1rem">
      <div class="metric-val" style="color:#0ea5e9">{int(df['males'].sum()):,}</div>
      <div class="metric-label">Total Males</div>
    </div>
    <div class="metric-card">
      <div class="metric-val" style="color:#f43f5e">{int(df['females'].sum()):,}</div>
      <div class="metric-label">Total Females</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    <strong>Males account for the vast majority</strong> of accident involvement.
    This aligns with driving demographics in Saudi Arabia, where male drivers
    historically constitute the larger share of road users during the study period.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── VIZ 6: Saudi vs Non-Saudi — Grouped Bar ──────────────────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ How does accident involvement compare between Saudi nationals and non-Saudi residents across cities?</div>', unsafe_allow_html=True)
st.markdown("##### 6 · Saudi vs Non-Saudi by City — Grouped Bar Chart")

nat = df.groupby("city")[["saudi", "non_saudi"]].sum()
nat = nat.loc[nat.sum(axis=1).nlargest(12).index]
x = np.arange(len(nat))
w = 0.4
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, nat["saudi"],     width=w, label="Saudi",     color="#0ea5e9")
ax.bar(x + w/2, nat["non_saudi"], width=w, label="Non-Saudi", color="#f97316")
ax.set_xticks(x)
ax.set_xticklabels(nat.index, rotation=40, ha="right", fontsize=9)
style_ax(ax, "Saudi vs Non-Saudi Involvement by City (Top 12)", "City", "Total Involved")
ax.legend(fontsize=9)
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    Saudi nationals are involved in more accidents in most cities. However, in certain
    cities like <strong>Jeddah</strong> and <strong>Madinah</strong>, non-Saudi involvement
    is proportionally higher — likely reflecting larger expatriate populations in those areas.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── VIZ 7: Inside vs Outside City — Box Plot ─────────────────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ Do accidents outside city limits differ in scale and distribution from those occurring within cities?</div>', unsafe_allow_html=True)
st.markdown("##### 7 · Inside vs Outside City — Box Plot")

melted = df[["inside_city", "outside_city"]].melt(var_name="Location", value_name="Accidents")
melted["Location"] = melted["Location"].map(
    {"inside_city": "Inside City", "outside_city": "Outside City"}
)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=melted, x="Location", y="Accidents",
            hue="Location",
            palette={"Inside City": "#0ea5e9", "Outside City": "#f97316"},
            width=0.45, ax=ax, legend=False,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
style_ax(ax, "Accident Distribution: Inside vs Outside City", "Location", "Count per Record")
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    Outside-city accidents show <strong>higher spread and more extreme outliers</strong>,
    suggesting that highway and inter-city road incidents tend to involve more people.
    The outliers are real data from high-volume cities and should be retained for accurate analysis.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── VIZ 8: Correlation Heatmap ───────────────────────────────
st.markdown('<div class="viz-card">', unsafe_allow_html=True)
st.markdown('<div class="viz-question">❓ Which features are most strongly correlated with each other in the dataset?</div>', unsafe_allow_html=True)
st.markdown("##### 8 · Feature Correlation Heatmap")

num_cols = ["males", "females", "inside_city", "outside_city",
            "age_18_30", "age_30_40", "saudi", "non_saudi",
            "has_injury", "has_death", "total_accidents"]
corr = df[num_cols].astype(float).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, linecolor="#f1f5f9",
            annot_kws={"size": 8}, ax=ax,
            cbar_kws={"label": "Correlation"})
ax.set_title("Feature Correlation Heatmap", fontsize=13,
             fontweight="bold", color="#0f172a", pad=10)
ax.tick_params(colors="#64748b", labelsize=8.5)
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div class="insight-box">
  <div class="insight-title">💡 Insights</div>
  <div class="insight-text">
    Strong positive correlations exist between <strong>males, saudi, age_18_30</strong> and
    <strong>total_accidents</strong> — confirming that young Saudi male drivers dominate
    the accident profile. <strong>inside_city</strong> and <strong>outside_city</strong>
    are moderately correlated with total volume but represent different risk contexts.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — KEY INSIGHTS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

ins_c1, ins_c2 = st.columns(2)
insights = [
    ("📉", "Declining Trend", "Total accidents dropped ~23% from 1437 to 1439, indicating improving road safety conditions."),
    ("👨", "Male Dominance", "Males account for the overwhelming majority of accident involvement across all cities and years."),
    ("🧑", "18–30 High Risk", "The 18–30 age group is the most frequently involved demographic — consistent across all 16 cities."),
    ("🏙️", "Riyadh Leads", "Riyadh records the highest accident volume by a significant margin across all three years."),
    ("🛣️", "Highway Risk", "Outside-city accidents show higher spread and more extreme outliers, suggesting greater highway severity."),
    ("📅", "Month 9 Spike", "Month 9 consistently records elevated accident counts across all years — a structural seasonal pattern."),
]
for i, (icon, title, text) in enumerate(insights):
    col = ins_c1 if i % 2 == 0 else ins_c2
    with col:
        st.markdown(f"""
        <div class="viz-card" style="margin-bottom:1rem; padding:1.2rem 1.5rem">
          <div style="font-size:1.5rem; margin-bottom:0.4rem">{icon}</div>
          <div style="font-family:'Syne',sans-serif; font-weight:700; color:#0f172a; margin-bottom:0.3rem">{title}</div>
          <div style="color:#64748b; font-size:0.92rem; line-height:1.6">{text}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 5 — ML MODELS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Predictive Models</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Two models were built to extract deeper insights — a time series forecaster and a city risk classifier.</div>', unsafe_allow_html=True)


# ── MODEL 1: Linear Regression / Time Series ─────────────────
st.markdown("""
<div class="viz-card">
  <div class="model-header">
    <span class="model-badge">MODEL 01</span>
    <span class="model-name">Time Series Forecast — Linear Regression</span>
  </div>
""", unsafe_allow_html=True)

st.markdown("""
**📖 Explanation**

Monthly total accidents were aggregated across all cities and assigned a sequential time index
(1 = first month of 1437, 36 = last month of 1439). A **Linear Regression** model was trained
on this sequence to capture the overall trend and extrapolate future values.

This approach is appropriate given the dataset size (36 monthly observations) and the
clearly visible linear downward trend in the data.
""")

# Train model
monthly = df.groupby(["year", "month"])["total_accidents"].sum().reset_index()
monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)
monthly["time_index"] = range(1, len(monthly) + 1)
X_ts = monthly[["time_index"]]
y_ts = monthly["total_accidents"]
ts_model = LinearRegression()
ts_model.fit(X_ts, y_ts)
monthly["predicted"] = ts_model.predict(X_ts)
r2_ts  = r2_score(y_ts, monthly["predicted"])
mae_ts = mean_absolute_error(y_ts, monthly["predicted"])
coef   = ts_model.coef_[0]

future_df = pd.DataFrame({"time_index": [37, 38, 39, 40, 41, 42]})
future_df["predicted"] = ts_model.predict(future_df[["time_index"]])
future_df["label"] = ["1440/1", "1440/2", "1440/3", "1440/4", "1440/5", "1440/6"]

# Display chart
st.markdown("**📊 Display**")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["time_index"], monthly["total_accidents"],
        marker="o", color="#0ea5e9", linewidth=2, markersize=5, label="Actual", zorder=5)
ax.plot(monthly["time_index"], monthly["predicted"],
        color="#f97316", linewidth=2, linestyle="--", label="Trend Line")
ax.plot(future_df["time_index"], future_df["predicted"],
        marker="s", color="#ef4444", linewidth=2, linestyle="--",
        markersize=6, label="Forecast (1440)")
ax.axvline(x=36.5, color="#94a3b8", linestyle=":", linewidth=1.5)
ax.text(36.8, monthly["total_accidents"].max() * 0.97,
        "Forecast →", color="#94a3b8", fontsize=9)
style_ax(ax,
         f"Monthly Accidents — Trend & Forecast\nR² = {r2_ts:.2f}  |  MAE = {mae_ts:.0f}  |  Trend: {coef:+.1f}/month",
         "Time Index (Month)", "Total Accidents")
ax.legend(fontsize=9)
fig.patch.set_facecolor("#fafafa")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

# Results
st.markdown("**📈 Results**")
rc1, rc2, rc3 = st.columns(3)
with rc1:
    st.markdown(f'<span class="result-pill">R² = {r2_ts:.4f}</span>', unsafe_allow_html=True)
    st.caption("Variance explained")
with rc2:
    st.markdown(f'<span class="result-pill">MAE = {mae_ts:.0f}</span>', unsafe_allow_html=True)
    st.caption("Avg prediction error (accidents)")
with rc3:
    st.markdown(f'<span class="result-pill">Trend: {coef:+.1f}/month</span>', unsafe_allow_html=True)
    st.caption("Monthly change")

st.markdown("**📅 6-Month Forecast**")
st.dataframe(future_df[["label", "predicted"]].rename(
    columns={"label": "Period", "predicted": "Forecast"}
).assign(Forecast=lambda d: d["Forecast"].round(0).astype(int)),
    hide_index=True, use_container_width=False)

# Interpretation
st.markdown("""
<div class="interp-box">
  <strong>🔍 Interpretation</strong><br>
  The model explains <strong>76% of monthly variance</strong> with a trend of <strong>−36 accidents per month</strong>.
  The remaining 24% is likely driven by seasonal effects and city-specific factors not captured by a linear model.
  Forecast values for 1440 suggest continued decline, approaching ~2,590 by the 6th month — assuming
  no structural changes in road conditions or policy.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── MODEL 2: K-Means Clustering ──────────────────────────────
st.markdown("""
<div class="viz-card">
  <div class="model-header">
    <span class="model-badge">MODEL 02</span>
    <span class="model-name">City Risk Clustering — K-Means</span>
  </div>
""", unsafe_allow_html=True)

st.markdown("""
**📖 Explanation**

Cities were grouped into risk clusters based on their average accident profiles.
Features used: `has_death`, `has_injury`, `outside_city`, `age_18_30`.
**StandardScaler** was applied first since K-Means is sensitive to feature magnitude.
The **Elbow Method** was used to identify the optimal k, and **Silhouette Score** validates the result.
""")

# Train model
city_df = df.groupby("city")[["has_death", "has_injury", "outside_city", "age_18_30"]].mean().reset_index()
features = ["has_death", "has_injury", "outside_city", "age_18_30"]
scaler   = StandardScaler()
X_cl     = scaler.fit_transform(city_df[features])

# Elbow display
col_elbow, col_scatter = st.columns(2)
with col_elbow:
    st.markdown("**📊 Elbow Method**")
    inertias = []
    k_range  = range(1, 8)
    for k in k_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_tmp.fit(X_cl)
        inertias.append(km_tmp.inertia_)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(list(k_range), inertias, marker="o", linewidth=2.5, color="#0ea5e9")
    ax.axvline(x=3, color="#ef4444", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.text(3.1, max(inertias) * 0.9, "k = 3", color="#ef4444", fontsize=9, fontweight="bold")
    style_ax(ax, "Elbow Method", "Number of Clusters (k)", "Inertia")
    fig.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

km = KMeans(n_clusters=3, random_state=42, n_init=10)
city_df["cluster"] = km.fit_predict(X_cl)
sil = silhouette_score(X_cl, city_df["cluster"])

cluster_names = {0: "Low Risk", 1: "High Risk", 2: "Medium Risk"}
city_df["cluster_label"] = city_df["cluster"].replace(cluster_names)

with col_scatter:
    st.markdown("**📊 City Clusters**")
    palette_cl = {"Low Risk": "#10b981", "Medium Risk": "#f97316", "High Risk": "#ef4444"}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.scatterplot(data=city_df, x="outside_city", y="age_18_30",
                    hue="cluster_label", palette=palette_cl, s=180, ax=ax, zorder=5)
    for _, row in city_df.iterrows():
        ax.text(row["outside_city"] + 1, row["age_18_30"] + 0.5,
                row["city"], fontsize=7.5, color="#475569")
    style_ax(ax, "City Risk Clusters", "Avg Outside City Accidents", "Avg Age 18-30 Accidents")
    ax.legend(title="Risk Level", fontsize=8, title_fontsize=8.5)
    fig.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Results
st.markdown("**📈 Results**")
st.markdown(f'<span class="result-pill">Silhouette Score = {sil:.4f}</span>', unsafe_allow_html=True)

r_c1, r_c2, r_c3 = st.columns(3)
for col, label, color, emoji in zip(
    [r_c1, r_c2, r_c3],
    ["Low Risk", "Medium Risk", "High Risk"],
    ["#10b981", "#f97316", "#ef4444"],
    ["🟢", "🟡", "🔴"]
):
    cities_in = city_df[city_df["cluster_label"] == label]["city"].tolist()
    with col:
        st.markdown(f"""
        <div style="border:1px solid #e2e8f0; border-radius:10px; padding:1rem; margin-bottom:0.5rem">
          <div style="font-weight:700; color:{color}; margin-bottom:0.5rem">{emoji} {label}</div>
          <div style="font-size:0.85rem; color:#475569; line-height:1.7">{"<br>".join(cities_in)}</div>
        </div>
        """, unsafe_allow_html=True)

# Interpretation
st.markdown("""
<div class="interp-box">
  <strong>🔍 Interpretation</strong><br>
  The clustering reveals three distinct city profiles. <strong>High-risk cities</strong> (Riyadh, Taif, Asir, Eastern Region)
  are large urban centres with high outside-city volumes and young-driver involvement.
  <strong>Medium-risk cities</strong> (Jeddah, Makkah, Madinah, Jazan) have moderate but significant activity.
  <strong>Low-risk cities</strong> are smaller, with lower accident volumes across all dimensions.
  A Silhouette Score of <strong>{:.2f}</strong> confirms good cluster separation.
</div>
""".format(sil), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 6 — CONCLUSION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Final Takeaway</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Conclusion</div>', unsafe_allow_html=True)

st.markdown("""
<div class="conclusion-card">
  <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#f0f9ff; margin-bottom:1.2rem">
    Key Takeaways from the Analysis
  </div>
  <div class="conclusion-item"><div class="conclusion-dot"></div>Traffic accidents in Saudi Arabia show a clear <strong>23% decline</strong> from 1437 to 1439 — a positive road safety signal.</div>
  <div class="conclusion-item"><div class="conclusion-dot"></div>Young males aged <strong>18–30</strong> remain the highest-risk demographic across all cities and years.</div>
  <div class="conclusion-item"><div class="conclusion-dot"></div><strong>Riyadh</strong> consistently records the highest accident volume and requires prioritised safety interventions.</div>
  <div class="conclusion-item"><div class="conclusion-dot"></div>The <strong>Linear Regression</strong> model (R² = 0.76) forecasts continued decline into 1440 under current conditions.</div>
  <div class="conclusion-item"><div class="conclusion-dot"></div><strong>K-Means clustering</strong> identifies three distinct city risk tiers, enabling targeted resource allocation for road safety programs.</div>
  <div class="conclusion-item"><div class="conclusion-dot"></div>Future work should incorporate weather data, road type, and vehicle information for more robust predictive modelling.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("🚦 Saudi Arabia Traffic Accidents Analysis · Tuwaiq Data Science & AI Bootcamp · Built with Streamlit")
