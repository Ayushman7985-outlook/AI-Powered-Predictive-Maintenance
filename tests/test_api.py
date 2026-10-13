import pytest
from unittest.mock import patch

from src.api import create_app
from src.config import FEATURES


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def valid_payload():
    return {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }


def test_health_endpoint_reports_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert "model_ready" in body


def test_root_lists_api_endpoints(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["endpoints"]["predict"] == "POST /predict"


def test_predict_returns_model_result(client):
    expected = {
        "prediction": 0,
        "status": "No failure predicted",
        "risk_percent": 2.5,
        "data_source": "Test data",
    }
    with patch("src.api.predict_machine", return_value=expected) as predict_mock:
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.get_json() == expected
    predict_mock.assert_called_once_with(valid_payload())


def test_predict_rejects_non_json_body(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_reports_missing_fields(client):
    response = client.post("/predict", json={"Type": "L"})
    assert response.status_code == 400
    assert set(response.get_json()["missing_fields"]) == set(FEATURES) - {"Type"}


def test_predict_rejects_unknown_machine_type(client):
    payload = valid_payload()
    payload["Type"] = "X"
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.get_json()["field"] == "Type"


@pytest.mark.parametrize("bad_value", ["1500", True, None, float("nan"), float("inf"), -1])
def test_predict_rejects_invalid_numeric_sensor_values(client, bad_value):
    payload = valid_payload()
    payload["Rotational speed [rpm]"] = bad_value
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.get_json()["field"] == "Rotational speed [rpm]"


def test_predict_returns_service_unavailable_when_model_missing(client):
    with patch("src.api.predict_machine", side_effect=FileNotFoundError):
        response = client.post("/predict", json=valid_payload())
    assert response.status_code == 503
    assert "python -m src.train" in response.get_json()["error"]
