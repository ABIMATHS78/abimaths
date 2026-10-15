import os

import pytest

os.environ["AIET_OFFLINE"] = "1"
os.environ["AIET_FETCH_PHOTOS"] = "0"  # tests never reach for starter photos on the internet; see test_starters


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "database_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "offline", True)
    from app import content, llm, photos

    # Remembered by the process, so not from another test.
    monkeypatch.setattr(llm, "account_problem", None)
    monkeypatch.setattr(content, "_first_post_tries", {})
    monkeypatch.setattr(photos, "_tried", {})
    monkeypatch.setattr(photos, "_fetching", set())
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
