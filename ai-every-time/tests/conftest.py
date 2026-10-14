import os

import pytest

os.environ["AIET_OFFLINE"] = "1"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "offline", True)
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
