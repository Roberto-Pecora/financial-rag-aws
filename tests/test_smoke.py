from fastapi.testclient import TestClient

from frag.api.main import app


def test_health():
    c = TestClient(app)
    assert c.get("/v1/health").status_code == 200
