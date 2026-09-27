import streamlit as st

from frontend.components.navigation import render_navigation
from frontend.views.home import show_home
from frontend.views.analytics import show_analytics
from frontend.views.delivery_risk import show_delivery_risk


st.set_page_config(
    page_title="Olist AI Platform",
    page_icon="🛒",
    layout="wide",
)

page = render_navigation()

if page == "Home":
    show_home()

elif page == "Analytics":
    show_analytics()

elif page == "Delivery Risk":
    show_delivery_risk()

elif page == "AI Assistant":
    st.title("🧠 AI Assistant")
    st.info(
        "The AI Assistant will be implemented with "
        "RAG and agentic AI in a later stage."
    )