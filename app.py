import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Global Diabetes Clustering",
    page_icon="🩺",
    layout="wide"
)

# ── Load Model & Data ─────────────────────────────────────
@st.cache_resource
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('merged_diabetes_data.csv')

model, scaler = load_model()
df = load_data()

cluster_names = {
    0: 'High Burden, Mid Capacity',
    1: 'Low Burden, Low Capacity',
    2: 'Healthy & Wealthy',
    3: 'Silent Crisis'
}

cluster_colors = {
    'High Burden, Mid Capacity': '#E74C3C',
    'Low Burden, Low Capacity':  '#F39C12',
    'Healthy & Wealthy':         '#27AE60',
    'Silent Crisis':             '#8E44AD'
}

df['Cluster_Name'] = df['Cluster'].map(cluster_names)

# ── Sidebar Navigation ────────────────────────────────────
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📦 Data & Method",
    "🗺️ World Map",
    "📊 Cluster Profiles",
    "🔍 Country Lookup",
    "🚨 Anomaly Detection",
    "💡 Key Findings"
])

# ════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🩺 Global Diabetes Burden Clustering")
    st.markdown("### *Which countries are sitting on an untreated diabetes crisis?*")
    st.markdown("---")

    st.markdown("""
    ## 🔴 The Problem
    Over **50% of diabetics globally are undiagnosed and untreated** — mostly in countries 
    that can't afford to manage the disease. But no one has clearly mapped *which* countries 
    are in the most dangerous position.

    This project uses **unsupervised machine learning** to answer that question — with no 
    predefined labels, letting the data discover natural country groupings itself.
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌍 Countries Analyzed", "124")
    col2.metric("📅 Data Year", "2022")
    col3.metric("📊 Features Used", "4")
    col4.metric("🔵 Clusters Found", "4")

    st.markdown("---")

    st.markdown("""
    ## 🎯 What We Discovered

    | Cluster | Countries | Key Insight |
    |---|---|---|
    | 🟢 Healthy & Wealthy | 46 | Low diabetes, high spending, many doctors |
    | 🔴 High Burden, Mid Capacity | 21 | High diabetes despite mid-level resources |
    | 🟠 Low Burden, Low Capacity | 38 | Low diabetes now but fragile healthcare |
    | 🟣 Silent Crisis | 18 | High diabetes, almost no healthcare, patients paying 64% out of pocket |

    > **Most alarming finding:** 18 countries have high diabetes burden but patients are 
    paying 64% of costs from their own pockets with almost no doctors available.
    """)

# ════════════════════════════════════════════════════════
# PAGE 2 — DATA & METHOD
# ════════════════════════════════════════════════════════
elif page == "📦 Data & Method":
    st.title("📦 Data Sources & Methodology")
    st.markdown("---")

    st.markdown("## 📂 Datasets Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. 🧬 Diabetes Prevalence (%)**
        - Source: NCD-RisC (2022)
        - Coverage: 200 countries
        - What: % of adults with diabetes (age-standardised, averaged across sexes)

        **2. 💰 Health Spending per Person (PPP)**
        - Source: World Bank via Our World in Data
        - Coverage: 204 countries
        - What: Healthcare spending adjusted for purchasing power
        """)

    with col2:
        st.markdown("""
        **3. 👨‍⚕️ Doctors per 1,000 People**
        - Source: WHO via Our World in Data
        - Coverage: 138 countries
        - What: Medical doctor density per 1,000 population

        **4. 💸 Out-of-Pocket Spending (%)**
        - Source: WHO via Our World in Data
        - Coverage: 205 countries
        - What: % of health costs paid directly by patients
        """)

    st.markdown("---")
    st.markdown("## 🛠️ Machine Learning Pipeline")

    st.markdown("""""")

    st.markdown("---")
    st.markdown("## 🧠 Why These Techniques?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **K-Means**
        No labeled data exists for country risk profiles — 
        K-Means discovers natural groupings without predefined categories.
        """)

    with col2:
        st.markdown("""
        **PCA**
        Compresses 4 features into 2 dimensions for visualization 
        while retaining 76.1% of all information.
        """)

    with col3:
        st.markdown("""
        **Isolation Forest**
        Detects countries that don't fit any cluster pattern — 
        these outliers are often the most interesting findings.
        """)

# ════════════════════════════════════════════════════════
# PAGE 3 — WORLD MAP
# ════════════════════════════════════════════════════════
elif page == "🗺️ World Map":
    st.title("🗺️ Global Diabetes Clusters (2022)")
    st.markdown("Hover over any country to see its full profile.")
    st.markdown("---")

    fig_map = px.choropleth(
        df,
        locations='ISO',
        color='Cluster_Name',
        hover_name='Country',
        hover_data={
            'diabetes_prevalence': ':.1%',
            'health_spending': ':$.0f',
            'doctors_per_1000': ':.2f',
            'oop_spending': ':.1f',
            'ISO': False
        },
        color_discrete_map=cluster_colors,
        title='Global Diabetes Burden Clustering (2022)'
    )
    fig_map.update_layout(
        legend_title='Cluster',
        geo=dict(showframe=False, showcoastlines=True),
        height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **🔴 High Burden, Mid Capacity** — Middle East, North Africa, India, parts of Latin America  
    **🟢 Healthy & Wealthy** — Western Europe, North America, Australia, Japan  
    **🟠 Low Burden, Low Capacity** — Sub-Saharan Africa, parts of Asia  
    **🟣 Silent Crisis** — West/Central Africa, Bangladesh, Yemen, Central Asia  
    """)

# ════════════════════════════════════════════════════════
# PAGE 4 — CLUSTER PROFILES
# ════════════════════════════════════════════════════════
elif page == "📊 Cluster Profiles":
    st.title("📊 Cluster Profiles")
    st.markdown("---")

    cluster_summary = df.groupby('Cluster_Name').agg(
        Countries=('Country', 'count'),
        Avg_Diabetes=('diabetes_prevalence', 'mean'),
        Avg_Spending=('health_spending', 'mean'),
        Avg_Doctors=('doctors_per_1000', 'mean'),
        Avg_OOP=('oop_spending', 'mean')
    ).round(3).reset_index()

    st.dataframe(cluster_summary, use_container_width=True)
    st.markdown("---")

    # Show countries per cluster
    for cluster_name, color in cluster_colors.items():
        countries = df[df['Cluster_Name'] == cluster_name]['Country'].tolist()
        with st.expander(f"🌍 {cluster_name} ({len(countries)} countries)"):
            st.write(', '.join(sorted(countries)))

# ════════════════════════════════════════════════════════
# PAGE 5 — COUNTRY LOOKUP
# ════════════════════════════════════════════════════════
elif page == "🔍 Country Lookup":
    st.title("🔍 Country Profile Lookup")
    st.markdown("Select any country to see its diabetes burden profile and cluster assignment.")
    st.markdown("---")

    country = st.selectbox("Select a country", sorted(df['Country'].tolist()))

    if country:
        row = df[df['Country'] == country].iloc[0]
        cluster = row['Cluster_Name']
        color = cluster_colors[cluster]

        st.markdown(f"## {country}")
        st.markdown(f"**Cluster:** {cluster}")
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🧬 Diabetes Prevalence", f"{row['diabetes_prevalence']:.1%}")
        col2.metric("💰 Health Spending (PPP)", f"${row['health_spending']:,.0f}")
        col3.metric("👨‍⚕️ Doctors per 1,000", f"{row['doctors_per_1000']:.2f}")
        col4.metric("💸 Out-of-Pocket %", f"{row['oop_spending']:.1f}%")

        st.markdown("---")

        anomaly_status = "🚨 Anomaly — unusual profile" if row['Anomaly_Label'] == 'Anomaly' else "✅ Normal profile"
        st.markdown(f"**Anomaly Status:** {anomaly_status}")

        same_cluster = df[df['Cluster_Name'] == cluster]['Country'].tolist()
        same_cluster = [c for c in same_cluster if c != country]
        st.markdown(f"**Similar countries in same cluster:** {', '.join(same_cluster[:10])}")

# ════════════════════════════════════════════════════════
# PAGE 6 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Countries")
    st.markdown("""
    These 7 countries were flagged by **Isolation Forest** as having unusual profiles 
    that don't fit neatly into any cluster pattern.
    """)
    st.markdown("---")

    anomalies = df[df['Anomaly'] == -1][['Country', 'diabetes_prevalence',
                                          'health_spending', 'doctors_per_1000',
                                          'oop_spending', 'Cluster_Name']]
    st.dataframe(anomalies, use_container_width=True)

    st.markdown("---")
    st.markdown("## 🔍 Why Each is an Anomaly")

    anomaly_explanations = {
        'United States of America': '💰 Spends $12,586 per person — almost double any other country in the dataset',
        'Switzerland': '💰 Extremely high spending ($10,574) even among wealthy nations',
        'Norway': '💰 Extremely high spending ($9,796) even among wealthy nations',
        'France': '🧬 Lowest diabetes in entire dataset (2.7%) — unusually healthy even for a wealthy country',
        'Georgia': '💸 57% out-of-pocket spending despite being in the Healthy & Wealthy cluster',
        'Armenia': '🏥 Much better healthcare capacity than its Silent Crisis cluster peers',
        'United Arab Emirates': '🧬 Highest diabetes in entire dataset (25.7%) despite $3,313 spending per person'
    }

    for country, explanation in anomaly_explanations.items():
        if country in df['Country'].values:
            st.markdown(f"**{country}:** {explanation}")

# ════════════════════════════════════════════════════════
# PAGE 7 — KEY FINDINGS
# ════════════════════════════════════════════════════════
elif page == "💡 Key Findings":
    st.title("💡 Key Findings")
    st.markdown("---")

    st.markdown("""
    ## 🔴 Finding 1 — The Silent Crisis
    **18 countries** have mid-high diabetes, almost no healthcare spending, 
    very few doctors — yet patients pay **64% of costs from their own pocket.**
    These include Bangladesh, Nigeria, Yemen, Chad, and Tajikistan.

    > *These are the most vulnerable diabetic populations on earth — sick, poor, and unsupported.*

    ---

    ## 🇮🇳 Finding 2 — India's Alarming Position
    India has the **highest diabetes prevalence among major economies (22.6%)** 
    but only spends **$310 PPP per person** on healthcare with only **0.723 doctors per 1,000 people.**
    India sits in the High Burden cluster — borderline crisis.

    ---

    ## 🇦🇪 Finding 3 — The Gulf State Paradox
    UAE has the **highest diabetes in the entire dataset (25.7%)** 
    despite spending $3,313 per person. Wealth didn't prevent disease — 
    rapid lifestyle changes from oil wealth did.

    ---

    ## 🟠 Finding 4 — Africa's Ticking Time Bomb
    Sub-Saharan Africa currently has **low diabetes** but also has the 
    **worst healthcare infrastructure.** As diets westernize and urbanization increases, 
    diabetes will rise — but the system has no capacity to respond.

    ---

    ## 🇫🇷 Finding 5 — The French Exception
    France has the **lowest diabetes prevalence in the dataset (2.7%)** — 
    significantly below other wealthy nations. Diet, lifestyle, and 
    preventive healthcare policy appear to play a major role.
    """)

    st.markdown("---")
    st.markdown("""
    ## 📌 Policy Implication
    > *Countries with similar disease burden profiles should share prevention 
    strategies and healthcare resources. The gap between disease burden and 
    healthcare capacity — not just prevalence alone — is the true measure of a diabetes crisis.*
    """)

# ── Footer ─────────────────────────────────────────────
st.markdown("---")
st.caption("Data: NCD-RisC (2022), Our World in Data — WHO/World Bank | Methods: K-Means, PCA, Isolation Forest | Built with Streamlit")