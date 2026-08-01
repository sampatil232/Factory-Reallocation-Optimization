"""
risk_panel.py
----------------------------------------
Supply Chain Risk Analysis

Responsibilities
----------------
• Load dataset
• Risk filters
• High-risk shipments
• Delay analysis
• Alerts
• Tableau Dashboard
"""

import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

from helper import (
    load_display_data,
    plotly_theme
)

# --------------------------------------------------------
# Apply Theme
# --------------------------------------------------------

plotly_theme()

# --------------------------------------------------------
# Load Dataset
# --------------------------------------------------------

df = load_display_data()

# --------------------------------------------------------
# Page Title
# --------------------------------------------------------

st.title("⚠️ Supply Chain Risk Panel")

st.markdown("""
Identify delayed shipments, high-risk routes and operational bottlenecks.
""")

st.divider()

# --------------------------------------------------------
# Sidebar Filters
# --------------------------------------------------------

st.sidebar.header("Risk Filters")

region = st.sidebar.multiselect(
    "Region",
    sorted(df["Region"].dropna().unique()),
    default=sorted(df["Region"].dropna().unique())
)

factory = st.sidebar.multiselect(
    "Factory",
    sorted(df["Factory"].dropna().unique()),
    default=sorted(df["Factory"].dropna().unique())
)

ship_mode = st.sidebar.multiselect(
    "Ship Mode",
    sorted(df["Ship Mode"].dropna().unique()),
    default=sorted(df["Ship Mode"].dropna().unique())
)

delay_limit = st.sidebar.slider(
    "Lead Time Threshold",
    min_value=int(df["Lead Time"].min()),
    max_value=int(df["Lead Time"].max()),
    value=int(df["Lead Time"].quantile(0.75))
)

# --------------------------------------------------------
# Apply Filters
# --------------------------------------------------------

filtered_df = df[
    (df["Region"].isin(region))
    &
    (df["Factory"].isin(factory))
    &
    (df["Ship Mode"].isin(ship_mode))
]

risk_df = filtered_df[
    filtered_df["Lead Time"] >= delay_limit
]

# --------------------------------------------------------
# KPI Cards
# --------------------------------------------------------

st.subheader("Risk Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "High Risk Shipments",
    len(risk_df)
)

c2.metric(
    "Average Lead Time",
    f"{risk_df['Lead Time'].mean():.2f}"
    if len(risk_df) else "0"
)

c3.metric(
    "Affected Factories",
    risk_df["Factory"].nunique()
)

c4.metric(
    "Affected Regions",
    risk_df["Region"].nunique()
)

st.divider()

# --------------------------------------------------------
# High Risk Routes
# --------------------------------------------------------

st.subheader("🚨 High-Risk Shipments")

display_cols = [
    col for col in [
        "Order ID",
        "Product Name",
        "Factory",
        "Region",
        "State/Province",
        "Ship Mode",
        "Lead Time",
        "Sales",
        "Gross Profit"
    ]
    if col in risk_df.columns
]

st.dataframe(
    risk_df[display_cols],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------------
# Delay Analysis
# --------------------------------------------------------

st.subheader("📈 Average Lead Time by Factory")

factory_delay = (
    risk_df.groupby("Factory", as_index=False)["Lead Time"]
    .mean()
    .sort_values("Lead Time", ascending=False)
)

fig_factory = px.bar(
    factory_delay,
    x="Factory",
    y="Lead Time",
    color="Lead Time",
    color_continuous_scale="Blues",
    title="Average Lead Time by Factory"
)

fig_factory.update_layout(
    template="shipping_blue",
    height=450
)

st.plotly_chart(fig_factory, width="stretch")

# --------------------------------------------------------
# Region Risk Chart
# --------------------------------------------------------

st.subheader("🌍 High-Risk Shipments by Region")

region_risk = (
    risk_df["Region"]
    .value_counts()
    .reset_index()
)

region_risk.columns = [
    "Region",
    "Shipments"
]

fig_region = px.pie(
    region_risk,
    names="Region",
    values="Shipments",
    color_discrete_sequence=[
        "#2563EB",
        "#3B82F6",
        "#60A5FA",
        "#93C5FD",
        "#BFDBFE"
    ]
)

fig_region.update_layout(
    template="shipping_blue",
    height=500
)

st.plotly_chart(fig_region, width="stretch")
# --------------------------------------------------------
# Alerts
# --------------------------------------------------------

st.subheader("🚨 Risk Alerts")

if len(risk_df) == 0:

    st.success(
        "✅ No high-risk shipments found."
    )

else:

    top_factory = (
        risk_df["Factory"]
        .value_counts()
        .idxmax()
    )

    top_region = (
        risk_df["Region"]
        .value_counts()
        .idxmax()
    )

    max_delay = risk_df["Lead Time"].max()

    st.warning(f"""
### Attention Required

• High-risk shipments detected.

• Highest Lead Time: **{max_delay:.2f} Days**

• Most affected Factory: **{top_factory}**

• Most affected Region: **{top_region}**

Consider reallocating production or changing the shipping mode to reduce delays.
""")

st.divider()

# --------------------------------------------------------
# Download
# --------------------------------------------------------

st.download_button(
    "📥 Download Risk Report",
    risk_df.to_csv(index=False).encode("utf-8"),
    "risk_report.csv",
    "text/csv"
)

st.divider()

# --------------------------------------------------------
# Tableau Dashboard 4
# --------------------------------------------------------

st.subheader("📊 Risk Dashboard")

tableau_html = """"<div class='tableauPlaceholder' id='viz1785515099769' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Le&#47;LeadTimeOptimizedLeadtime&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='LeadTimeOptimizedLeadtime&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Le&#47;LeadTimeOptimizedLeadtime&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1785515099769');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1127px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
components.html(tableau_html, height=950)