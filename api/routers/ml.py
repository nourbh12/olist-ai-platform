from fastapi import APIRouter

from api.schemas.delivery_risk import (
    DeliveryRiskRequest,
    DeliveryRiskResponse,
)

from api.services.delivery_risk import delivery_risk_service


router = APIRouter(
    prefix="/api/v1/ml",
    tags=["Machine Learning"],
)


@router.post(
    "/delivery-risk",
    response_model=DeliveryRiskResponse,
)
def predict_delivery_risk(
    request: DeliveryRiskRequest,
):

    features = request.model_dump()

    result = delivery_risk_service.predict(features)

    return result