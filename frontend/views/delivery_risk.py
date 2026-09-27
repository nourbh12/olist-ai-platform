import streamlit as st

from frontend.api_client import predict_delivery_risk


def show_delivery_risk():

    st.title("Delivery Risk Prediction")

    st.markdown(
        """
        Estimate the probability that an order will
        experience delivery risk using the trained
        machine-learning model.
        """
    )

    with st.form("delivery_risk_form"):

        col1, col2 = st.columns(2)

        with col1:

            order_item_count = st.number_input(
                "Order Item Count",
                min_value=1,
                value=2,
                step=1,
            )

            order_total_price = st.number_input(
                "Order Total Price",
                min_value=0.0,
                value=150.0,
                step=10.0,
            )

            order_total_freight = st.number_input(
                "Order Total Freight",
                min_value=0.0,
                value=25.0,
                step=5.0,
            )

        with col2:

            purchase_hour = st.slider(
                "Purchase Hour",
                min_value=0,
                max_value=23,
                value=14,
            )

            purchase_day_of_week = st.slider(
                "Purchase Day of Week",
                min_value=0,
                max_value=6,
                value=2,
            )

            estimated_delivery_duration_days = (
                st.number_input(
                    "Estimated Delivery Duration (days)",
                    min_value=1,
                    value=20,
                    step=1,
                )
            )

        submitted = st.form_submit_button(
            "Predict Delivery Risk"
        )

    if submitted:

        payload = {
            "order_item_count": order_item_count,
            "order_total_price": order_total_price,
            "order_total_freight": order_total_freight,
            "purchase_hour": purchase_hour,
            "purchase_day_of_week": purchase_day_of_week,
            "estimated_delivery_duration_days":
                estimated_delivery_duration_days,
        }

        try:

            result = predict_delivery_risk(
                payload
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Risk Score",
                    f"{result['risk_score']:.2%}",
                )

            with col2:
                st.metric(
                    "Threshold",
                    f"{result['threshold']:.2%}",
                )

            with col3:
                prediction = result["prediction"]

                st.metric(
                    "Prediction",
                    "HIGH" if prediction == 1
                    else "LOW",
                )

            if result["risk_level"] == "HIGH":

                st.error(
                    "⚠️ High delivery risk detected."
                )

            else:

                st.success(
                    "✅ Low delivery risk detected."
                )

        except Exception as exc:

            st.error(
                "Unable to obtain a prediction."
            )

            st.caption(str(exc))