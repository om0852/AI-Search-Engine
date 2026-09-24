import pytest
from starlette.testclient import TestClient
from src.serving.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_parse_query_endpoint(client):
    resp = client.post("/search/parse", json={"query": "oayement issue occur in last 2 days"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["result"]["corrected_query"] == "payment issue"
    assert data["result"]["time_filter"]["expression"] == "last 2 days"

def test_mongo_build_endpoint(client):
    payload = {
        "query": "oayement issue occur in last 2 days",
        "filters": {"platform": "twitter", "sentiment": "negative"}
    }
    resp = client.post("/search/mongo/build", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["result"]["mongo_query"]["platform"] == "twitter"
