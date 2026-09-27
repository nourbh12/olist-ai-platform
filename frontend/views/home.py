import streamlit as st


def show_home():
    st.markdown(
        """
        <div style="text-align: center; padding: 2.5rem 1rem 1.5rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
                🛒 Olist AI Platform
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='text-align: center;'>"
        "<p style='font-size: 1.3rem; color: #666;'>"
        "Analytics · Machine Learning · AI"
        "</p>"
        "<p style='font-size: 1.1rem;'>"
        "Turn e-commerce data into actionable insights "
        "using analytics, machine learning and AI."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Business Analytics")
        st.write(
            "Explore sales, products, customers, delivery, "
            "reviews and seller performance through "
            "interactive dashboards."
        )

    with col2:
        st.markdown("### 🤖 Machine Learning")
        st.write(
            "Predict delivery risk using a trained "
            "machine-learning model based on historical "
            "Olist orders."
        )

    with col3:
        st.markdown("### 🧠 AI Assistant")
        st.write(
            "Ask questions about the business data using "
            "retrieval-augmented generation and agentic AI."
        )
        st.caption("Coming soon")

    st.divider()

   