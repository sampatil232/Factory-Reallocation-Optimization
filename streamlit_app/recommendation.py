"""
recommendation.py
------------------------------------
AI Factory Recommendation Page
"""

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from helper import plotly_theme

# ----------------------------------------------------
# Apply Theme
# ----------------------------------------------------

plotly_theme()

# ----------------------------------------------------
# Load Recommendation Dataset
# ----------------------------------------------------

@st.cache_data
def load_recommendations():
    return pd.read_csv("data/factory_recommendations.csv")


df = load_recommendations()

# ----------------------------------------------------
# Page Title
# ----------------------------------------------------

st.title("🤖 Factory Recommendation Engine")

st.markdown(
    """
Identify the best factory allocation based on
lead time optimization and operational efficiency.
"""
)

st.divider()

# ----------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------

st.sidebar.header("Recommendation Filters")

product = st.sidebar.selectbox(
    "Product",
    ["All"] + sorted(df["Product Name"].dropna().unique().tolist())
)

factory = st.sidebar.selectbox(
    "Current Factory",
    ["All"] + sorted(df["Current Factory"].dropna().unique().tolist())
)

# ----------------------------------------------------
# Apply Filters
# ----------------------------------------------------

filtered_df = df.copy()

if product != "All":
    filtered_df = filtered_df[
        filtered_df["Product Name"] == product
    ]

if factory != "All":
    filtered_df = filtered_df[
        filtered_df["Current Factory"] == factory
    ]

# ----------------------------------------------------
# KPI Cards
# ----------------------------------------------------

st.subheader("Recommendation Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Recommendations",
    len(filtered_df)
)

col2.metric(
    "Avg Lead Time Saved",
    f"{filtered_df['Lead Time Reduction'].mean():.2f} Days"
)

col3.metric(
    "Maximum Saving",
    f"{filtered_df['Lead Time Reduction'].max():.2f} Days"
)

col4.metric(
    "Avg Improvement",
    f"{filtered_df['Lead Time Reduction (%)'].mean():.2f}%"
)

st.divider()

# ----------------------------------------------------
# Recommendation Table
# ----------------------------------------------------

st.subheader("📋 Factory Recommendations")

display_columns = [

    "Product Name",

    "Current Factory",

    "Recommended Factory",

    "Current Lead Time",

    "Predicted Lead Time",

    "Lead Time Reduction",

    "Lead Time Reduction (%)",

    "Recommendation Score"

]

available_columns = [
    col for col in display_columns
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True,
    hide_index=True
)

# ----------------------------------------------------
# Download CSV
# ----------------------------------------------------

st.download_button(
    "📥 Download Recommendations",
    filtered_df.to_csv(index=False).encode("utf-8"),
    "factory_recommendations.csv",
    "text/csv"
)

st.divider()

# ----------------------------------------------------
# Best Recommendation
# ----------------------------------------------------

if len(filtered_df):

    best = filtered_df.sort_values(
        "Lead Time Reduction",
        ascending=False
    ).iloc[0]

    st.success(f"""

### ⭐ Best Recommendation

**Product:** {best['Product Name']}

**Current Factory:** {best['Current Factory']}

**Recommended Factory:** {best['Recommended Factory']}

**Lead Time Saved:** {best['Lead Time Reduction']:.2f} Days

**Improvement:** {best['Lead Time Reduction (%)']:.2f}%

""")

st.divider()

# ----------------------------------------------------
# Tableau Dashboard 2
# ----------------------------------------------------

st.subheader("📊 Recommendation Dashboard")

tableau_html = """
<div class='tableauPlaceholder' id='viz1785513525661' style='position: relative'><noscript><a href='#'><img alt='Dashboard 2 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ta&#47;tableau_17853525329650&#47;Dashboard2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='tableau_17853525329650&#47;Dashboard2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ta&#47;tableau_17853525329650&#47;Dashboard2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1785513525661');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
"""

components.html(tableau_html, height=950)