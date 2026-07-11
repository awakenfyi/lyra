"""
test_api_auth.py — proves unauthenticated requests to the Lyra API are rejected.

C3 fix: all non-health endpoints must return 401 when no valid bearer token is
present. /health is always open (liveness probe).
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Mock bridge packages before importing main so the test runs without
# a full ML environment.
_BRIDGE_MOCKS = [
    "bridge",
    "bridge.bridge",
    "bridge.coherence_proxy",
]
for _mod in _BRIDGE_MOCKS:
    sys.modules.setdefault(_mod, MagicMock())

# Ensure src/api is on the path, then import the app.
_API_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from fastapi.testclient import TestClient  # noqa: E402 — after sys.path patch

TOKEN = "lyra-test-token-abc123"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch):
    monkeypatch.setenv("LYRA_API_TOKEN", TOKEN)
    # Force re-import so the middleware picks up the patched env.
    sys.modules.pop("main", None)
    yield
    sys.modules.pop("main", None)


@pytest.fixture()
def client():
    import main  # noqa: PLC0415
    return TestClient(main.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

def test_health_no_auth_required(client):
    """Health endpoint is always accessible — no token needed."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_unauthenticated_request_rejected(client):
    """A request with no Authorization header returns 401."""
    resp = client.post(
        "/v1/evaluate",
        json={"api_response": {"choices": [{"message": {"content": "hi"}}]}},
    )
    assert resp.status_code == 401


def test_wrong_token_rejected(client):
    """A request with the wrong bearer token returns 401."""
    resp = client.post(
        "/v1/evaluate",
        json={"api_response": {"choices": [{"message": {"content": "hi"}}]}},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_correct_token_passes_auth_layer(client):
    """Correct bearer token is accepted — auth layer passes the request through."""
    resp = client.post(
        "/v1/evaluate",
        json={"api_response": {"choices": [{"message": {"content": "hi"}}]}},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    # Auth passed. Business logic may return 400/500 (no real bridge), but not 401.
    assert resp.status_code != 401


def test_shadow_scan_unauthenticated_rejected(client):
    """Shadow scan endpoint also rejects unauthenticated requests."""
    resp = client.post("/v1/shadow-scan", json={"text": "Great question!"})
    assert resp.status_code == 401


def test_memory_status_unauthenticated_rejected(client):
    """Memory status endpoint also rejects unauthenticated requests."""
    resp = client.get("/v1/memory/status", params={"namespace": "test"})
    assert resp.status_code == 401
