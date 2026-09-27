import streamlit as st
import pandas as pd
import altair as alt

from frontend.api_client import get_analytics


# ============================================================
# HELPERS
# ============================================================

def get_datasets(endpoint: str) -> dict:
    """Fetch datasets from an analytics endpoint."""
    response = get_analytics(
        f"/api/v1/analytics/{endpoint}"
    )

    if not isinstance(response, dict):
        return {}

    return response.get("datasets", {})


def first_row(dataset: list) -> dict:
    """Return the first row of a dataset."""
    if isinstance(dataset, list) and dataset:
        return dataset[0]

    return {}


def to_dataframe(dataset: list) -> pd.DataFrame:
    """Convert an API dataset to a DataFrame."""
    if not dataset:
        return pd.DataFrame()

    return pd.DataFrame(dataset)


def format_number(value) -> str:
    """Format a numeric value."""
    if value is None:
        return "—"

    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def format_currency(value) -> str:
    """Format a currency value."""
    if value is None:
        return "—"

    try:
        return f"€{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_decimal(value, decimals: int = 2) -> str:
    """Format a decimal value."""
    if value is None:
        return "—"

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def show_dataframe(dataset, height: int = 350):
    """Display a dataset as a dataframe."""
    df = to_dataframe(dataset)

    if df.empty:
        st.info("No data available.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        hide_index=True,
    )


def prepare_monthly_data(
    dataset,
    value_column: str,
) -> pd.DataFrame:
    """
    Prepare monthly data while preserving chronological dates.

    The month column remains a real datetime so Altair can
    correctly order Jan 2017, Feb 2017, ..., Jan 2018, etc.
    """
    df = to_dataframe(dataset)

    if df.empty:
        return df

    if "month" not in df.columns:
        return pd.DataFrame()

    if value_column not in df.columns:
        return pd.DataFrame()

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce",
    )

    df = df.dropna(subset=["month"])

    df = df.sort_values("month")

    return df[["month", value_column]]


def monthly_line_chart(
    df: pd.DataFrame,
    value_column: str,
    y_title: str,
    tooltip_format: str = ",.2f",
    height: int = 400,
):
    """
    Display a chronological monthly line chart.

    The x-axis uses the real datetime value but displays
    only Month Year, e.g. Jan 2017.
    """
    if df.empty:
        st.info("No monthly data available.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(
            point=True,
        )
        .encode(
            x=alt.X(
                "month:T",
                title="Month",
                axis=alt.Axis(
                    format="%b %Y",
                    labelAngle=-45,
                ),
            ),
            y=alt.Y(
                f"{value_column}:Q",
                title=y_title,
            ),
            tooltip=[
                alt.Tooltip(
                    "month:T",
                    title="Month",
                    format="%b %Y",
                ),
                alt.Tooltip(
                    f"{value_column}:Q",
                    title=y_title,
                    format=tooltip_format,
                ),
            ],
        )
        .properties(
            height=height,
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )


# ============================================================
# SALES
# ============================================================

def show_sales():
    st.subheader("Sales Performance")
    st.caption(
        "Overview of revenue, orders and sales trends."
    )

    try:
        datasets = get_datasets("sales")
    except Exception as exc:
        st.error("Unable to load sales analytics.")
        st.caption(str(exc))
        return

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpi = first_row(
        datasets.get(
            "overall_sales_kpis",
            [],
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Orders",
            format_number(
                kpi.get("total_orders")
            ),
        )

    with col2:
        st.metric(
            "Total Revenue",
            format_currency(
                kpi.get("total_revenue")
            ),
        )

    with col3:
        st.metric(
            "Average Order Value",
            format_currency(
                kpi.get("average_order_value")
            ),
        )

    with col4:
        st.metric(
            "Maximum Order Value",
            format_currency(
                kpi.get("maximum_order_value")
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Monthly Sales Performance
    # --------------------------------------------------------

    monthly_sales = datasets.get(
        "monthly_sales_performance",
        [],
    )

    if monthly_sales:
        st.markdown("#### Monthly Sales Performance")

        chart_df = prepare_monthly_data(
            monthly_sales,
            "revenue",
        )

        monthly_line_chart(
            chart_df,
            "revenue",
            "Revenue (€)",
        )
    else:
        st.info(
            "Monthly sales data is not available."
        )

    # --------------------------------------------------------
    # Sales by Order Status
    # --------------------------------------------------------

    status_sales = datasets.get(
        "sales_by_order_status",
        [],
    )

    if status_sales:
        st.markdown("#### Sales by Order Status")

        df = to_dataframe(status_sales)

        if (
            "order_status" in df.columns
            and "total_revenue" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "order_status",
                        "total_revenue",
                    ]
                ]
                .sort_values(
                    "total_revenue",
                    ascending=False,
                )
                .set_index("order_status")
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(status_sales)

    # --------------------------------------------------------
    # Late Orders Impact
    # --------------------------------------------------------

    late_sales = datasets.get(
        "late_orders_impact_on_sales",
        [],
    )

    if late_sales:
        st.markdown(
            "#### Late Orders Impact on Sales"
        )

        df = to_dataframe(late_sales)

        if (
            "is_late" in df.columns
            and "total_revenue" in df.columns
        ):
            chart_df = df[
                [
                    "is_late",
                    "total_revenue",
                ]
            ].copy()

            chart_df["delivery_status"] = (
                chart_df["is_late"]
                .apply(
                    lambda value: (
                        "Late"
                        if str(value).lower()
                        in {
                            "true",
                            "1",
                            "yes",
                        }
                        else "On Time"
                    )
                )
            )

            chart_df = (
                chart_df[
                    [
                        "delivery_status",
                        "total_revenue",
                    ]
                ]
                .set_index("delivery_status")
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(late_sales)


# ============================================================
# PRODUCTS
# ============================================================

def show_products():
    st.subheader("Product Performance")
    st.caption(
        "Explore product, category and freight performance."
    )

    try:
        datasets = get_datasets("products")
    except Exception as exc:
        st.error("Unable to load product analytics.")
        st.caption(str(exc))
        return

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpi = first_row(
        datasets.get(
            "overall_product_kpis",
            [],
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Products",
            format_number(
                kpi.get("total_products")
            ),
        )

    with col2:
        st.metric(
            "Average Product Price",
            format_currency(
                kpi.get("average_product_price")
            ),
        )

    with col3:
        st.metric(
            "Product Revenue",
            format_currency(
                kpi.get("total_product_revenue")
            ),
        )

    with col4:
        st.metric(
            "Freight Revenue",
            format_currency(
                kpi.get("total_freight_revenue")
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Revenue by Category
    # --------------------------------------------------------

    category_data = datasets.get(
        "revenue_by_product_category",
        [],
    )

    if category_data:
        st.markdown(
            "#### Revenue by Product Category"
        )

        df = to_dataframe(category_data)

        if (
            "category" in df.columns
            and "revenue" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "category",
                        "revenue",
                    ]
                ]
                .sort_values(
                    "revenue",
                    ascending=False,
                )
                .head(15)
                .set_index("category")
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(category_data)

    # --------------------------------------------------------
    # Top Products
    # --------------------------------------------------------

    top_products = datasets.get(
        "top_products_by_revenue",
        [],
    )

    if top_products:
        st.markdown(
            "#### Top Products by Revenue"
        )

        df = to_dataframe(top_products)

        if (
            "product_id" in df.columns
            and "revenue" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "product_id",
                        "revenue",
                    ]
                ]
                .sort_values(
                    "revenue",
                    ascending=False,
                )
                .head(10)
                .set_index("product_id")
            )

            st.bar_chart(chart_df)

        show_dataframe(top_products)


# ============================================================
# CUSTOMERS
# ============================================================

def show_customers():
    st.subheader("Customer Performance")
    st.caption(
        "Explore customer activity and geographic performance."
    )

    try:
        datasets = get_datasets("customers")
    except Exception as exc:
        st.error(
            "Unable to load customer analytics."
        )
        st.caption(str(exc))
        return

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpi = first_row(
        datasets.get(
            "overall_customer_kpis",
            [],
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers",
            format_number(
                kpi.get("total_customers")
            ),
        )

    with col2:
        st.metric(
            "Orders",
            format_number(
                kpi.get("total_orders")
            ),
        )

    with col3:
        st.metric(
            "Orders / Customer",
            format_decimal(
                kpi.get("orders_per_customer")
            ),
        )

    with col4:
        st.metric(
            "Revenue / Customer",
            format_currency(
                kpi.get("revenue_per_customer")
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Customer Performance by State
    # --------------------------------------------------------

    state_data = datasets.get(
        "customer_performance_by_state",
        [],
    )

    if state_data:
        st.markdown(
            "#### Revenue by Customer State"
        )

        df = to_dataframe(state_data)

        if (
            "customer_state" in df.columns
            and "revenue" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "customer_state",
                        "revenue",
                    ]
                ]
                .sort_values(
                    "revenue",
                    ascending=False,
                )
                .set_index("customer_state")
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(state_data)

    # --------------------------------------------------------
    # Customer Segments
    # --------------------------------------------------------

    customer_segments = datasets.get(
        "customers_by_number_of_orders",
        [],
    )

    if customer_segments:
        st.markdown(
            "#### Customer Segments"
        )

        df = to_dataframe(
            customer_segments
        )

        if (
            "customer_segment" in df.columns
            and "customers" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "customer_segment",
                        "customers",
                    ]
                ]
                .set_index(
                    "customer_segment"
                )
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(
                customer_segments
            )

    # --------------------------------------------------------
    # Monthly Customer Activity
    # --------------------------------------------------------

    monthly_activity = datasets.get(
        "monthly_customer_activity",
        [],
    )

    if monthly_activity:
        st.markdown(
            "#### Monthly Customer Activity"
        )

        chart_df = prepare_monthly_data(
            monthly_activity,
            "active_customers",
        )

        monthly_line_chart(
            chart_df,
            "active_customers",
            "Active Customers",
            tooltip_format=",",
        )


# ============================================================
# DELIVERY
# ============================================================

def show_delivery():
    st.subheader("Delivery Performance")
    st.caption(
        "Monitor delivery times, delays and late deliveries."
    )

    try:
        datasets = get_datasets("delivery")
    except Exception as exc:
        st.error(
            "Unable to load delivery analytics."
        )
        st.caption(str(exc))
        return

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpi = first_row(
        datasets.get(
            "overall_delivery_kpis",
            [],
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Orders",
            format_number(
                kpi.get("total_orders")
            ),
        )

    with col2:
        estimated = kpi.get(
            "avg_estimated_delivery_days"
        )

        st.metric(
            "Estimated Delivery",
            (
                f"{float(estimated):.1f} days"
                if estimated is not None
                else "—"
            ),
        )

    with col3:
        actual = kpi.get(
            "avg_actual_delivery_days"
        )

        st.metric(
            "Actual Delivery",
            (
                f"{float(actual):.1f} days"
                if actual is not None
                else "—"
            ),
        )

    with col4:
        late_rate = kpi.get(
            "late_delivery_rate"
        )

        st.metric(
            "Late Delivery Rate",
            (
                f"{float(late_rate):.2f}%"
                if late_rate is not None
                else "—"
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Average Delivery Delay
    # --------------------------------------------------------

    avg_delay = kpi.get(
        "avg_delivery_delay_days"
    )

    if avg_delay is not None:
        st.metric(
            "Average Delivery Delay",
            f"{float(avg_delay):.1f} days",
        )

    # --------------------------------------------------------
    # Monthly Delivery Performance
    # --------------------------------------------------------

    monthly_delivery = datasets.get(
        "delivery_performance_by_month",
        [],
    )

    if monthly_delivery:
        st.markdown(
            "#### Monthly Delivery Performance"
        )

        chart_df = prepare_monthly_data(
            monthly_delivery,
            "late_rate_percentage",
        )

        monthly_line_chart(
            chart_df,
            "late_rate_percentage",
            "Late Delivery Rate (%)",
        )

    # --------------------------------------------------------
    # Delivery Performance by Order Status
    # --------------------------------------------------------

    status_delivery = datasets.get(
        "delivery_performance_by_order_status",
        [],
    )

    if status_delivery:
        st.markdown(
            "#### Delivery Performance by Order Status"
        )

        df = to_dataframe(
            status_delivery
        )

        if (
            "order_status" in df.columns
            and "avg_delay_days" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "order_status",
                        "avg_delay_days",
                    ]
                ]
                .set_index("order_status")
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(
                status_delivery
            )

    # --------------------------------------------------------
    # Late Deliveries by Seller
    # --------------------------------------------------------

    seller_delivery = datasets.get(
        "late_deliveries_by_seller",
        [],
    )

    if seller_delivery:
        st.markdown(
            "#### Late Deliveries by Seller"
        )

        df = to_dataframe(
            seller_delivery
        )

        if (
            "seller_id" in df.columns
            and "late_rate_percentage"
            in df.columns
        ):
            chart_df = (
                df[
                    [
                        "seller_id",
                        "late_rate_percentage",
                    ]
                ]
                .sort_values(
                    "late_rate_percentage",
                    ascending=False,
                )
                .head(15)
                .set_index("seller_id")
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(
                seller_delivery
            )


# ============================================================
# REVIEWS
# ============================================================

def show_reviews():
    st.subheader("Review Performance")
    st.caption(
        "Understand customer satisfaction and its "
        "relationship with delivery performance."
    )

    try:
        datasets = get_datasets("reviews")
    except Exception as exc:
        st.error(
            "Unable to load review analytics."
        )
        st.caption(str(exc))
        return

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpi = first_row(
        datasets.get(
            "overall_review_kpis",
            [],
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Reviews",
            format_number(
                kpi.get("total_reviews")
            ),
        )

    with col2:
        average_score = kpi.get(
            "average_review_score"
        )

        st.metric(
            "Average Review Score",
            (
                f"{float(average_score):.2f} / 5"
                if average_score is not None
                else "—"
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Review Score Distribution
    # --------------------------------------------------------

    distribution = datasets.get(
        "review_score_distribution",
        [],
    )

    if distribution:
        st.markdown(
            "#### Review Score Distribution"
        )

        df = to_dataframe(
            distribution
        )

        if (
            "review_score" in df.columns
            and "reviews" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "review_score",
                        "reviews",
                    ]
                ]
                .sort_values(
                    "review_score"
                )
                .set_index("review_score")
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(
                distribution
            )

    # --------------------------------------------------------
    # Review Score vs Delivery
    # --------------------------------------------------------

    review_delivery = datasets.get(
        "review_score_vs_delivery_performance",
        [],
    )

    if review_delivery:
        st.markdown(
            "#### Review Score vs Delivery Performance"
        )

        df = to_dataframe(
            review_delivery
        )

        columns = [
            "review_score",
            "reviews",
            "average_delivery_delay",
            "late_delivery_rate",
        ]

        available_columns = [
            column
            for column in columns
            if column in df.columns
        ]

        if available_columns:
            st.dataframe(
                df[available_columns],
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # Reviews by Delivery Status
    # --------------------------------------------------------

    reviews_by_delivery = datasets.get(
        "reviews_by_delivery_status",
        [],
    )

    if reviews_by_delivery:
        st.markdown(
            "#### Reviews by Delivery Status"
        )

        df = to_dataframe(
            reviews_by_delivery
        )

        if (
            "delivery_status" in df.columns
            and "average_review_score"
            in df.columns
        ):
            chart_df = (
                df[
                    [
                        "delivery_status",
                        "average_review_score",
                    ]
                ]
                .set_index(
                    "delivery_status"
                )
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(
                reviews_by_delivery
            )

    # --------------------------------------------------------
    # Review Score by Month
    # --------------------------------------------------------

    monthly_reviews = datasets.get(
        "review_score_by_month",
        [],
    )

    if monthly_reviews:
        st.markdown(
            "#### Review Score Over Time"
        )

        chart_df = prepare_monthly_data(
            monthly_reviews,
            "average_review_score",
        )

        if not chart_df.empty:
            chart = (
                alt.Chart(chart_df)
                .mark_line(
                    point=True,
                )
                .encode(
                    x=alt.X(
                        "month:T",
                        title="Month",
                        axis=alt.Axis(
                            format="%b %Y",
                            labelAngle=-45,
                        ),
                    ),
                    y=alt.Y(
                        "average_review_score:Q",
                        title="Average Review Score",
                        scale=alt.Scale(
                            domain=[0, 5]
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "month:T",
                            title="Month",
                            format="%b %Y",
                        ),
                        alt.Tooltip(
                            "average_review_score:Q",
                            title="Average Score",
                            format=".2f",
                        ),
                    ],
                )
                .properties(
                    height=400,
                )
            )

            st.altair_chart(
                chart,
                use_container_width=True,
            )


# ============================================================
# SELLERS
# ============================================================

def show_sellers():
    st.subheader("Seller Performance")
    st.caption(
        "Analyze seller activity and delivery performance."
    )

    try:
        datasets = get_datasets("sellers")
    except Exception as exc:
        st.error(
            "Unable to load seller analytics."
        )
        st.caption(str(exc))
        return

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpi = first_row(
        datasets.get(
            "overall_seller_kpis",
            [],
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Sellers",
            format_number(
                kpi.get("total_sellers")
            ),
        )

    with col2:
        st.metric(
            "Orders",
            format_number(
                kpi.get("total_orders")
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Top Sellers by Orders
    # --------------------------------------------------------

    top_sellers = datasets.get(
        "top_sellers_by_orders",
        [],
    )

    if top_sellers:
        st.markdown(
            "#### Top Sellers by Orders"
        )

        df = to_dataframe(
            top_sellers
        )

        if (
            "seller_id" in df.columns
            and "total_orders" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "seller_id",
                        "total_orders",
                    ]
                ]
                .sort_values(
                    "total_orders",
                    ascending=False,
                )
                .head(15)
                .set_index("seller_id")
            )

            st.bar_chart(chart_df)

        show_dataframe(
            top_sellers
        )

    # --------------------------------------------------------
    # Sellers with Highest Late Rate
    # --------------------------------------------------------

    worst_sellers = datasets.get(
        "worst_sellers_by_late_rate",
        [],
    )

    if worst_sellers:
        st.markdown(
            "#### Sellers with Highest Late Delivery Rate"
        )

        df = to_dataframe(
            worst_sellers
        )

        columns = [
            "seller_id",
            "total_orders",
            "late_orders",
            "late_rate_percentage",
            "average_delivery_delay_days",
        ]

        available_columns = [
            column
            for column in columns
            if column in df.columns
        ]

        st.dataframe(
            df[available_columns],
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Seller Performance Distribution
    # --------------------------------------------------------

    performance_distribution = datasets.get(
        "seller_performance_distribution",
        [],
    )

    if performance_distribution:
        st.markdown(
            "#### Seller Performance Distribution"
        )

        df = to_dataframe(
            performance_distribution
        )

        if (
            "seller_performance" in df.columns
            and "sellers" in df.columns
        ):
            chart_df = (
                df[
                    [
                        "seller_performance",
                        "sellers",
                    ]
                ]
                .set_index(
                    "seller_performance"
                )
            )

            st.bar_chart(chart_df)

        else:
            show_dataframe(
                performance_distribution
            )


# ============================================================
# MAIN ANALYTICS PAGE
# ============================================================

def show_analytics():
    st.title("📊 Business Analytics")

    st.markdown(
        """
        Explore Olist e-commerce performance across six
        business dimensions using the analytics API.
        """
    )

    st.divider()

    tabs = st.tabs(
        [
            "💰 Sales",
            "📦 Products",
            "👥 Customers",
            "🚚 Delivery",
            "⭐ Reviews",
            "🏪 Sellers",
        ]
    )

    with tabs[0]:
        show_sales()

    with tabs[1]:
        show_products()

    with tabs[2]:
        show_customers()

    with tabs[3]:
        show_delivery()

    with tabs[4]:
        show_reviews()

    with tabs[5]:
        show_sellers()