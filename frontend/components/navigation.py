import streamlit as st


def render_navigation():
    st.markdown(
        """
        <style>
        .main-nav {
            display: flex;
            align-items: center;
            gap: 2rem;
            padding: 0.75rem 0 1rem;
            border-bottom: 1px solid #ddd;
            margin-bottom: 2rem;
        }

        .brand {
            font-size: 1.25rem;
            font-weight: 700;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-nav">
            <div class="brand">
                🛒 Olist AI Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st.radio(
        "Navigation",
        [
            "Home",
            "Analytics",
            "Delivery Risk",
            "AI Assistant",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )