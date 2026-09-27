from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.exceptions import (
    AnalyticsDataNotFoundError,
    AnalyticsQueryError,
)
from api.routers import analytics, ml


app = FastAPI(
    title="AI Platform API",
    description="API for analytics, machine learning and RAG.",
    version="1.0.0",
)


app.include_router(ml.router)
app.include_router(analytics.router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-platform-api",
    }


@app.exception_handler(AnalyticsDataNotFoundError)
async def analytics_data_not_found_handler(
    request: Request,
    exc: AnalyticsDataNotFoundError,
):
    return JSONResponse(
        status_code=503,
        content={
            "error": "analytics_data_unavailable",
            "message": "Analytics data is currently unavailable.",
        },
    )


@app.exception_handler(AnalyticsQueryError)
async def analytics_query_error_handler(
    request: Request,
    exc: AnalyticsQueryError,
):
    return JSONResponse(
        status_code=500,
        content={
            "error": "analytics_query_failed",
            "message": "The analytics query could not be executed.",
        },
    )