from typing import Any

from pydantic import BaseModel


class AnalyticsResponse(BaseModel):
    domain: str
    datasets: dict[str, list[dict[str, Any]]]