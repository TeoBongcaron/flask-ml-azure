import json
from app import app


def test_index_route():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "message" in data


def test_predict_route_valid():
    client = app.test_client()
    payload = {"feature1": 0.5, "feature2": 0.7}
    resp = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prediction" in data
    assert "probability" in data


def test_predict_route_invalid():
    client = app.test_client()
    payload = {"feature1": "bad"}
    resp = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400
