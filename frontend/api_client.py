import requests
import os

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")


def get_health():
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_analytics(endpoint: str):
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def predict_delivery_risk(payload: dict):
    response = requests.post(
        f"{API_BASE_URL}/api/v1/ml/delivery-risk",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()