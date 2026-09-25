"""
NYC Food Safety Complaint Intelligence Dashboard
================================================
An interactive EDA dashboard built with Streamlit and Plotly.
Uses the full GT-A dataset (labelled_complaints_ground_truth.csv, 73,450 records)
and the raw DOHMH inspections dataset for a comprehensive view of food safety
complaints across New York City.

Run locally:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="NYC Food Safety Intelligence Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border: 1px solid #3a3d4e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #9e9e9e; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e0e0;
        border-bottom: 2px solid #4fc3f7;
        padding-bottom: 6px;
        margin: 24px 0 16px 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    gt_path = ROOT / "data" / "labelled" / "labelled_complaints_ground_truth.csv"
    raw_path = ROOT / "data" / "raw" / "nyc_311_food_complaints.csv"
    dohmh_path = ROOT / "data" / "raw" / "dohmh_inspections.csv"

    df = pd.read_csv(gt_path, parse_dates=["created_date"], low_memory=False)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["month"] = df["created_date"].dt.to_period("M").astype(str)
    df["year"] = df["created_date"].dt.year
    df["day_of_week"] = df["created_date"].dt.day_name()
    df["hour"] = df["created_date"].dt.hour
    df["hazard_label"] = df["priority_label"].map({1: "Actionable Hazard", 0: "Routine"})
    df["tier_label"] = df["hazard_tier"].map({2: "Critical", 1: "Moderate", 0: "Routine"})

    dohmh = None
    if dohmh_path.exists():
        dohmh = pd.read_csv(dohmh_path, parse_dates=["inspection_date"], low_memory=False)

    return df, dohmh

df, dohmh = load_data()

# ── Sidebar Filters ────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/320px-New_york_times_square-terabass.jpg", use_container_width=True)
st.sidebar.markdown("## 🔍 Filters")

boroughs = ["All"] + sorted(df["borough"].dropna().unique().tolist())
sel_borough = st.sidebar.selectbox("Borough", boroughs)

years = sorted(df["year"].dropna().unique().tolist())
sel_years = st.sidebar.multiselect("Year(s)", years, default=years[-2:] if len(years) >= 2 else years)

tiers = ["All", "Critical", "Moderate", "Routine"]
sel_tier = st.sidebar.selectbox("Hazard Tier", tiers)

top_n = st.sidebar.slider("Top N complaint types to show", 5, 20, 10)

# Apply filters
fdf = df.copy()
if sel_borough != "All":
    fdf = fdf[fdf["borough"] == sel_borough]
if sel_years:
    fdf = fdf[fdf["year"].isin(sel_years)]
if sel_tier != "All":
    fdf = fdf[fdf["tier_label"] == sel_tier]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 24px 0 8px 0;">
    <h1 style="font-size:2.2rem; margin-bottom:4px;">🍽️ NYC Food Safety Complaint Intelligence</h1>
    <p style="color:#9e9e9e; font-size:1rem;">
        Interactive Exploratory Data Analysis | Ground Truth A Dataset | 73,450 complaints
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
actionable_pct = fdf["priority_label"].mean() * 100
critical_n = (fdf["hazard_tier"] == 2).sum()
unique_addr = fdf["incident_address"].nunique() if "incident_address" in fdf.columns else 0
boroughs_shown = fdf["borough"].nunique()

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(fdf):,}</div>
        <div class="metric-label">Total Complaints</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#ef5350;">{critical_n:,}</div>
        <div class="metric-label">Critical Hazards</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#ffa726;">{actionable_pct:.1f}%</div>
        <div class="metric-label">Actionable Priority Rate</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#66bb6a;">{unique_addr:,}</div>
        <div class="metric-label">Unique Establishments</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#ab47bc;">{boroughs_shown}</div>
        <div class="metric-label">Boroughs</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Temporal Trends",
    "🗺️ Geographic Distribution",
    "🏷️ Complaint Analysis",
    "⚠️ Hazard Profile",
    "🔬 Inspection Data"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Temporal Trends
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Complaint Volume Over Time</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        monthly = (
            fdf.groupby(["month", "tier_label"])
            .size()
            .reset_index(name="count")
        )
        fig_monthly = px.area(
            monthly, x="month", y="count", color="tier_label",
            color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726", "Routine": "#66bb6a"},
            labels={"count": "Complaints", "month": "Month", "tier_label": "Hazard Tier"},
            title="Monthly Complaint Volume by Hazard Tier",
        )
        fig_monthly.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#e0e0e0", legend_title="Hazard Tier",
            xaxis_tickangle=-45, hovermode="x unified"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = fdf["day_of_week"].value_counts().reindex(dow_order).reset_index()
        dow.columns = ["day", "count"]
        fig_dow = px.bar(
            dow, x="count", y="day", orientation="h",
            color="count", color_continuous_scale="Blues",
            title="Complaints by Day of Week",
            labels={"count": "Count", "day": ""},
        )
        fig_dow.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#e0e0e0", showlegend=False,
            yaxis={"categoryorder": "array", "categoryarray": dow_order[::-1]}
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown('<div class="section-header">Hourly and Seasonal Patterns</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        hourly = fdf.groupby(["hour", "tier_label"]).size().reset_index(name="count")
        fig_hour = px.line(
            hourly, x="hour", y="count", color="tier_label",
            color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726", "Routine": "#66bb6a"},
            title="Complaints by Hour of Day",
            labels={"hour": "Hour of Day (24h)", "count": "Complaints"},
            markers=True,
        )
        fig_hour.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="#e0e0e0"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with col4:
        fdf["month_name"] = fdf["created_date"].dt.month
        monthly_avg = (
            fdf.groupby(["month_name", "tier_label"])
            .size()
            .reset_index(name="count")
        )
        fig_seasonal = px.box(
            fdf.assign(month_n=fdf["created_date"].dt.month),
            x="month_n", y="month_n", color="tier_label",
            title="Seasonal Distribution",
            labels={"month_n": "Month"},
            color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726", "Routine": "#66bb6a"}
        )
        # Simpler: heatmap of month x borough
        pivot = fdf.groupby([fdf["created_date"].dt.month_name(), "borough"]).size().unstack(fill_value=0)
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        pivot = pivot.reindex([m for m in month_order if m in pivot.index])
        fig_heat = px.imshow(
            pivot, color_continuous_scale="Blues",
            title="Complaint Volume: Month × Borough",
            labels=dict(x="Borough", y="Month", color="Complaints"),
            aspect="auto",
        )
        fig_heat.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="#e0e0e0"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — Geographic Distribution
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Spatial Distribution of Complaints</div>', unsafe_allow_html=True)

    map_df = fdf.dropna(subset=["latitude", "longitude"]).copy()
    map_df = map_df[(map_df["latitude"] > 40.4) & (map_df["latitude"] < 40.95) &
                    (map_df["longitude"] > -74.3) & (map_df["longitude"] < -73.6)]

    col1, col2 = st.columns([3, 1])

    with col1:
        # Sample for performance
        sample_n = min(5000, len(map_df))
        map_sample = map_df.sample(sample_n, random_state=42)

        fig_map = px.scatter_mapbox(
            map_sample,
            lat="latitude", lon="longitude",
            color="tier_label",
            color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726", "Routine": "#4caf50"},
            hover_name="descriptor",
            hover_data={"borough": True, "tier_label": True, "created_date": True,
                        "latitude": False, "longitude": False},
            zoom=10, height=500,
            mapbox_style="carto-darkmatter",
            title=f"Complaint Map — {sample_n:,} sampled points",
            opacity=0.7,
        )
        fig_map.update_layout(
            paper_bgcolor="#1e2130", font_color="#e0e0e0",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        boro_counts = fdf["borough"].value_counts().reset_index()
        boro_counts.columns = ["borough", "count"]
        fig_boro = px.bar(
            boro_counts, x="count", y="borough", orientation="h",
            color="count", color_continuous_scale="Reds",
            title="Complaints by Borough",
            labels={"count": "Total", "borough": ""}
        )
        fig_boro.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#e0e0e0", showlegend=False
        )
        st.plotly_chart(fig_boro, use_container_width=True)

        # Borough hazard breakdown
        boro_tier = (
            fdf.groupby(["borough", "tier_label"])
            .size()
            .reset_index(name="count")
        )
        fig_bt = px.bar(
            boro_tier, x="borough", y="count", color="tier_label",
            barmode="stack",
            color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726", "Routine": "#66bb6a"},
            title="Hazard Tier by Borough",
            labels={"count": "Complaints", "borough": "", "tier_label": "Tier"},
        )
        fig_bt.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#e0e0e0", xaxis_tickangle=-30
        )
        st.plotly_chart(fig_bt, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — Complaint Analysis
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Top Complaint Types</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        top_desc = fdf["descriptor"].value_counts().head(top_n).reset_index()
        top_desc.columns = ["descriptor", "count"]
        fig_desc = px.bar(
            top_desc, x="count", y="descriptor", orientation="h",
            color="count", color_continuous_scale="Viridis",
            title=f"Top {top_n} Complaint Descriptors",
            labels={"count": "Complaints", "descriptor": ""},
        )
        fig_desc.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#e0e0e0", showlegend=False,
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig_desc, use_container_width=True)

    with col2:
        # Sunburst of descriptor → hazard tier
        sun_df = fdf.groupby(["tier_label", "descriptor"]).size().reset_index(name="count")
        # Limit per tier for readability
        top_per_tier = (
            sun_df.sort_values("count", ascending=False)
            .groupby("tier_label").head(6)
        )
        fig_sun = px.sunburst(
            top_per_tier,
            path=["tier_label", "descriptor"],
            values="count",
            color="tier_label",
            color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726", "Routine": "#66bb6a"},
            title="Complaint Type Distribution by Hazard Tier",
        )
        fig_sun.update_layout(
            paper_bgcolor="#1e2130", font_color="#e0e0e0",
            margin={"t": 60, "l": 0, "r": 0, "b": 0}
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown('<div class="section-header">Complaint Type vs Hazard Tier Cross-Analysis</div>', unsafe_allow_html=True)

    cross = pd.crosstab(
        fdf["descriptor"].where(fdf["descriptor"].isin(
            fdf["descriptor"].value_counts().head(12).index)),
        fdf["tier_label"]
    ).fillna(0)
    cross = cross.reindex(columns=["Critical", "Moderate", "Routine"], fill_value=0)

    fig_cross = px.imshow(
        cross, color_continuous_scale="YlOrRd",
        title="Complaint Descriptor vs Hazard Tier (Top 12 Types)",
        labels=dict(x="Hazard Tier", y="Descriptor", color="Count"),
        aspect="auto",
    )
    fig_cross.update_layout(
        paper_bgcolor="#1e2130", font_color="#e0e0e0"
    )
    st.plotly_chart(fig_cross, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4 — Hazard Profile
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Ground Truth Label Distribution</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        tier_counts = fdf["tier_label"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        fig_pie = px.pie(
            tier_counts, names="tier", values="count",
            color="tier",
            color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726", "Routine": "#66bb6a"},
            title="3-Tier Hazard Distribution",
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="#1e2130", font_color="#e0e0e0"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        binary_counts = fdf["hazard_label"].value_counts().reset_index()
        binary_counts.columns = ["label", "count"]
        fig_binary = px.pie(
            binary_counts, names="label", values="count",
            color="label",
            color_discrete_map={"Actionable Hazard": "#ef5350", "Routine": "#66bb6a"},
            title="Binary Priority Label (Production BiLSTM target)",
            hole=0.45,
        )
        fig_binary.update_layout(
            paper_bgcolor="#1e2130", font_color="#e0e0e0"
        )
        st.plotly_chart(fig_binary, use_container_width=True)

    with col3:
        # Critical complaints by borough
        crit_boro = (
            fdf[fdf["tier_label"] == "Critical"]["borough"]
            .value_counts()
            .reset_index()
        )
        crit_boro.columns = ["borough", "count"]
        fig_cb = px.bar(
            crit_boro, x="borough", y="count",
            color="count", color_continuous_scale="Reds",
            title="Critical Hazards by Borough",
            labels={"count": "Critical Complaints", "borough": ""},
        )
        fig_cb.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#e0e0e0", showlegend=False
        )
        st.plotly_chart(fig_cb, use_container_width=True)

    st.markdown('<div class="section-header">Temporal Hazard Escalation Trends</div>', unsafe_allow_html=True)
    hazard_monthly = (
        fdf.groupby(["month", "tier_label"])
        .size()
        .reset_index(name="count")
    )
    total_monthly = hazard_monthly.groupby("month")["count"].transform("sum")
    hazard_monthly["pct"] = hazard_monthly["count"] / total_monthly * 100
    fig_trend = px.line(
        hazard_monthly[hazard_monthly["tier_label"] != "Routine"],
        x="month", y="pct", color="tier_label",
        color_discrete_map={"Critical": "#ef5350", "Moderate": "#ffa726"},
        title="% of Complaints Classified as Critical or Moderate Over Time",
        labels={"pct": "% of Monthly Complaints", "month": "Month", "tier_label": "Tier"},
        markers=True,
    )
    fig_trend.update_layout(
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        font_color="#e0e0e0", xaxis_tickangle=-45
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 5 — DOHMH Inspection Data
# ═══════════════════════════════════════════════════════════
with tab5:
    if dohmh is None:
        st.warning("DOHMH inspections data not loaded.")
    else:
        st.markdown('<div class="section-header">NYC DOHMH Inspection Overview</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            grade_counts = dohmh["grade"].value_counts().reset_index()
            grade_counts.columns = ["grade", "count"]
            grade_counts = grade_counts[grade_counts["grade"].isin(["A", "B", "C"])]
            fig_grade = px.bar(
                grade_counts, x="grade", y="count",
                color="grade",
                color_discrete_map={"A": "#66bb6a", "B": "#ffa726", "C": "#ef5350"},
                title="DOHMH Inspection Grade Distribution",
                labels={"count": "Number of Inspections", "grade": "Grade"},
            )
            fig_grade.update_layout(
                plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                font_color="#e0e0e0", showlegend=False
            )
            st.plotly_chart(fig_grade, use_container_width=True)

        with col2:
            crit_flag = dohmh["critical_flag"].value_counts().reset_index()
            crit_flag.columns = ["flag", "count"]
            fig_flag = px.pie(
                crit_flag, names="flag", values="count",
                color="flag",
                color_discrete_map={"Critical": "#ef5350", "Not Critical": "#66bb6a", "Not Applicable": "#9e9e9e"},
                title="DOHMH Violation Criticality",
                hole=0.45,
            )
            fig_flag.update_layout(
                paper_bgcolor="#1e2130", font_color="#e0e0e0"
            )
            st.plotly_chart(fig_flag, use_container_width=True)

        st.markdown('<div class="section-header">Violation Score Distribution</div>', unsafe_allow_html=True)
        dohmh["score"] = pd.to_numeric(dohmh["score"], errors="coerce")
        dohmh_valid = dohmh.dropna(subset=["score"])
        fig_score = px.histogram(
            dohmh_valid, x="score", nbins=50,
            color_discrete_sequence=["#4fc3f7"],
            title="Distribution of DOHMH Inspection Scores (Higher = More Violations)",
            labels={"score": "Inspection Score", "count": "Inspections"},
        )
        fig_score.add_vline(x=dohmh_valid["score"].mean(), line_dash="dash",
                            line_color="#ef5350",
                            annotation_text=f"Mean: {dohmh_valid['score'].mean():.1f}",
                            annotation_position="top right")
        fig_score.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font_color="#e0e0e0"
        )
        st.plotly_chart(fig_score, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            top_cuisine = dohmh["cuisine_description"].value_counts().head(12).reset_index()
            top_cuisine.columns = ["cuisine", "count"]
            fig_cuis = px.bar(
                top_cuisine, x="count", y="cuisine", orientation="h",
                color="count", color_continuous_scale="Oranges",
                title="Most Inspected Cuisine Types",
                labels={"count": "Inspections", "cuisine": ""},
            )
            fig_cuis.update_layout(
                plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                font_color="#e0e0e0", showlegend=False,
                yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(fig_cuis, use_container_width=True)

        with col4:
            insp_type = dohmh["inspection_type"].value_counts().head(8).reset_index()
            insp_type.columns = ["type", "count"]
            fig_itype = px.pie(
                insp_type, names="type", values="count",
                title="Inspection Type Breakdown",
                hole=0.4,
            )
            fig_itype.update_layout(
                paper_bgcolor="#1e2130", font_color="#e0e0e0"
            )
            st.plotly_chart(fig_itype, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #616161; font-size: 0.8rem; padding: 12px;">
    NYC Food Safety Complaint Intelligence Dashboard • Ground Truth A Dataset (73,450 records) •
    Data Source: NYC Open Data (311 Complaints + DOHMH Inspections)
</div>
""", unsafe_allow_html=True)
