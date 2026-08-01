"""
app.py
--------------------------------------------
Main entry point for the Streamlit application.

Responsibilities
----------------
• Configure Streamlit page
• Display application title
• Handle page navigation

No dataset loading.
No filters.
No charts.
No KPIs.
"""

import streamlit as st

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Factory Reallocation & Shipping Optimization",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)




# ---------------------------------------------------
# Application Header
# ---------------------------------------------------

st.markdown(
    """
    <h1><div class="main-title"><strong>
        🚚 Factory Reallocation & Shipping Optimization
    </strong>    
    </div></h1>

    <div class="subtitle">
        AI-Powered Supply Chain Analytics Dashboard
    </div>

    <hr>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# Navigation Pages
# ---------------------------------------------------

dashboard = st.Page(
    "dashboard.py",
    title="Dashboard",
    icon="📊",
    default=True
)

recommendation = st.Page(
    "recommendation.py",
    title="Recommendations",
    icon="🤖"
)

what_if = st.Page(
    "what_if.py",
    title="What-If Analysis",
    icon="🔄"
)

risk_panel = st.Page(
    "risk_panel.py",
    title="Risk Panel",
    icon="⚠️"
)

# ---------------------------------------------------
# Navigation
# ---------------------------------------------------

navigation = st.navigation(
    [
        dashboard,
        recommendation,
        what_if,
        risk_panel
    ]
)

navigation.run()