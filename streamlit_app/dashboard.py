"""
dashboard.py
---------------------------------------
Dashboard Page

Contains:
• Load dashboard dataset
• Dashboard filters
• KPI cards
• (Part 2 will include charts and Tableau dashboards)
"""

import streamlit as st
from helper import (
    load_display_data,
    calculate_kpis,
    plotly_theme,
)

# -------------------------------------------------------
# Apply Plotly Theme
# -------------------------------------------------------

plotly_theme()

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = load_display_data()

# -------------------------------------------------------
# Page Title
# -------------------------------------------------------

st.title("📊 Shipping Analytics Dashboard")

st.markdown(
    "Monitor shipment performance, lead times, sales and logistics KPIs."
)

st.divider()

# -------------------------------------------------------
# Sidebar Filters
# -------------------------------------------------------

st.sidebar.header("Dashboard Filters")

# Region
region = st.sidebar.multiselect(
    "Region",
    options=sorted(df["Region"].dropna().unique()),
    default=sorted(df["Region"].dropna().unique())
)

# Division
division = st.sidebar.multiselect(
    "Division",
    options=sorted(df["Division"].dropna().unique()),
    default=sorted(df["Division"].dropna().unique())
)

# Ship Mode
ship_mode = st.sidebar.multiselect(
    "Ship Mode",
    options=sorted(df["Ship Mode"].dropna().unique()),
    default=sorted(df["Ship Mode"].dropna().unique())
)

# State
state = st.sidebar.multiselect(
    "State / Province",
    options=sorted(df["State/Province"].dropna().unique()),
    default=sorted(df["State/Province"].dropna().unique())
)

# Date Range (if available)
if "Order Date" in df.columns:

    df["Order Date"] = df["Order Date"].astype("datetime64[ns]")

    min_date = df["Order Date"].min()
    max_date = df["Order Date"].max()

    date_range = st.sidebar.date_input(
        "Order Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

else:
    date_range = None

# -------------------------------------------------------
# Apply Filters
# -------------------------------------------------------

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["Region"].isin(region)
]

filtered_df = filtered_df[
    filtered_df["Division"].isin(division)
]

filtered_df = filtered_df[
    filtered_df["Ship Mode"].isin(ship_mode)
]

filtered_df = filtered_df[
    filtered_df["State/Province"].isin(state)
]

if (
    date_range
    and len(date_range) == 2
    and "Order Date" in filtered_df.columns
):

    start_date, end_date = date_range

    filtered_df = filtered_df[
        (
            filtered_df["Order Date"]
            >= str(start_date)
        )
        &
        (
            filtered_df["Order Date"]
            <= str(end_date)
        )
    ]

# -------------------------------------------------------
# Calculate KPIs
# -------------------------------------------------------

kpis = calculate_kpis(filtered_df)

# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------

st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🚚 Shipments",
        f"{kpis['Shipments']:,}"
    )

with col2:
    st.metric(
        "🏭 Factories",
        kpis["Factories"]
    )

with col3:
    st.metric(
        "📦 Products",
        kpis["Products"]
    )

with col4:
    st.metric(
        "🌍 Regions",
        kpis["Regions"]
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "⏱ Avg Lead Time",
        f"{kpis['Average Lead Time']:.2f}"
    )

with col6:
    st.metric(
        "📈 Total Sales",
        f"${kpis['Total Sales']:,.0f}"
    )

with col7:
    st.metric(
        "💰 Gross Profit",
        f"${kpis['Gross Profit']:,.0f}"
    )

with col8:
    st.metric(
        "📊 Avg Profit",
        f"${kpis['Average Profit']:,.0f}"
    )

st.divider()

# -------------------------------------------------------
# Plotly Charts
# -------------------------------------------------------

import plotly.express as px
import streamlit.components.v1 as components

# -------------------------------
# Lead Time by Region
# -------------------------------

st.subheader("📈 Average Lead Time by Region")

lead_region = (
    filtered_df.groupby("Region", as_index=False)["Lead Time"]
    .mean()
    .sort_values("Lead Time", ascending=False)
)

fig_region = px.bar(
    lead_region,
    x="Region",
    y="Lead Time",
    color="Lead Time",
    color_continuous_scale="Blues",
    title="Average Lead Time Across Regions"
)

fig_region.update_layout(
    template="shipping_blue",
    height=450,
    xaxis_title="Region",
    yaxis_title="Lead Time"
)


st.plotly_chart(fig_region, width="stretch")
# -------------------------------
# Sales by Division
# -------------------------------

st.subheader("💰 Sales by Division")

sales_div = (
    filtered_df.groupby("Division", as_index=False)["Sales"]
    .sum()
    .sort_values("Sales", ascending=False)
)

fig_sales = px.pie(
    sales_div,
    names="Division",
    values="Sales",
    color_discrete_sequence=[
        "#2563EB",
        "#3B82F6",
        "#60A5FA",
        "#93C5FD",
        "#BFDBFE",
    ]
)

fig_sales.update_layout(
    template="shipping_blue",
    height=500
)

st.plotly_chart(fig_sales, width="stretch")

# -------------------------------
# Ship Mode Distribution
# -------------------------------

st.subheader("🚚 Ship Mode Distribution")

ship = (
    filtered_df["Ship Mode"]
    .value_counts()
    .reset_index()
)

ship.columns = ["Ship Mode", "Orders"]

fig_ship = px.bar(
    ship,
    x="Ship Mode",
    y="Orders",
    color="Orders",
    color_continuous_scale="Blues"
)

fig_ship.update_layout(
    template="shipping_blue",
    height=450
)

st.plotly_chart(fig_ship, width="stretch")

# -------------------------------
# Sales vs Gross Profit
# -------------------------------

if (
    "Sales" in filtered_df.columns
    and "Gross Profit" in filtered_df.columns
):

    st.subheader("📊 Sales vs Gross Profit")

    fig_scatter = px.scatter(
        filtered_df,
        x="Sales",
        y="Gross Profit",
        color="Region",
        hover_data=["Factory"],
        color_discrete_sequence=[
            "#2563EB",
            "#3B82F6",
            "#60A5FA",
            "#93C5FD",
            "#BFDBFE",
        ]
    )

    fig_scatter.update_layout(
        template="shipping_blue",
        height=500
    )

    st.plotly_chart(fig_scatter, width="stretch")

# -------------------------------------------------------
# Download Button
# -------------------------------------------------------

st.download_button(
    label="📥 Download Filtered Dataset",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_shipments.csv",
    mime="text/csv",
)

st.divider()

# -------------------------------------------------------
# Tableau Dashboard 1
# -------------------------------------------------------

st.subheader("📊 Executive Overview")


tableau_html = """
<div class='tableauPlaceholder' id='viz1785513560880' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ta&#47;tableau_17853525329650&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='tableau_17853525329650&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ta&#47;tableau_17853525329650&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1785513560880');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script>
"""                                           


components.html(tableau_html, height=950)
# -------------------------------------------------------
# Tableau Dashboard 3
# -------------------------------------------------------

import streamlit.components.v1 as components

st.subheader("📊 Operational &  Logistics Dashboard")

tableau_html = """
<div class='tableauPlaceholder' id='viz1785513337721' style='position: relative'>
<noscript>
<a href='#'>
<img alt='Dashboard 3' src='https://public.tableau.com/static/images/ta/tableau_17853525329650/Dashboard3/1_rss.png' style='border: none' />
</a>
</noscript>

<object class='tableauViz' style='display:none;'>

<param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
<param name='embed_code_version' value='3' />
<param name='site_root' value='' />
<param name='name' value='tableau_17853525329650/Dashboard3' />
<param name='tabs' value='no' />
<param name='toolbar' value='yes' />
<param name='static_image' value='https://public.tableau.com/static/images/ta/tableau_17853525329650/Dashboard3/1.png' />
<param name='animate_transition' value='yes' />
<param name='display_static_image' value='yes' />
<param name='display_spinner' value='yes' />
<param name='display_overlay' value='yes' />
<param name='display_count' value='yes' />
<param name='language' value='en-US' />

</object>
</div>

<script type='text/javascript'>
var divElement = document.getElementById('viz1785513337721');
var vizElement = divElement.getElementsByTagName('object')[0];
vizElement.style.width='100%';
vizElement.style.height='900px';

var scriptElement = document.createElement('script');
scriptElement.src='https://public.tableau.com/javascripts/api/viz_v1.js';
vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
"""

components.html(tableau_html, height=950)
# -------------------------------------------------------
# Executive Summary
# -------------------------------------------------------

st.subheader("📋 Executive Summary")

c1, c2 = st.columns(2)

with c1:

    st.info(
        f"""
### Dashboard Insights

- **Total Shipments:** {kpis['Shipments']:,}
- **Average Lead Time:** {kpis['Average Lead Time']:.2f}
- **Factories:** {kpis['Factories']}
- **Products:** {kpis['Products']}
"""
    )

with c2:

    st.success(
        f"""
### Financial Overview

- **Total Sales:** ${kpis['Total Sales']:,.0f}
- **Gross Profit:** ${kpis['Gross Profit']:,.0f}
- **Average Profit:** ${kpis['Average Profit']:,.0f}
"""
    )

