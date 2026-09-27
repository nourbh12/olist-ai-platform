from fastapi import APIRouter

from api.schemas.analytics import AnalyticsResponse
from api.services.analytics import analytics_service


router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Analytics"],
)


@router.get(
    "/customers",
    response_model=AnalyticsResponse,
)
def customer_analytics():
    return analytics_service.get_domain("customers")


@router.get(
    "/delivery",
    response_model=AnalyticsResponse,
)
def delivery_analytics():
    return analytics_service.get_domain("delivery")


@router.get(
    "/products",
    response_model=AnalyticsResponse,
)
def product_analytics():
    return analytics_service.get_domain("products")


@router.get(
    "/reviews",
    response_model=AnalyticsResponse,
)
def review_analytics():
    return analytics_service.get_domain("reviews")


@router.get(
    "/sales",
    response_model=AnalyticsResponse,
)
def sales_analytics():
    return analytics_service.get_domain("sales")


@router.get(
    "/sellers",
    response_model=AnalyticsResponse,
)
def seller_analytics():
    return analytics_service.get_domain("sellers")