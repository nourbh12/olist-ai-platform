from pydantic import BaseModel, Field


class DeliveryRiskRequest(BaseModel):
    order_item_count: int = Field(..., ge=1)
    order_total_price: float = Field(..., ge=0)
    order_total_freight: float = Field(..., ge=0)
    purchase_hour: int = Field(..., ge=0, le=23)
    purchase_day_of_week: int = Field(..., ge=0, le=6)
    estimated_delivery_duration_days: int = Field(..., ge=0)


class DeliveryRiskResponse(BaseModel):
    risk_score: float
    threshold: float
    prediction: int
    risk_level: str