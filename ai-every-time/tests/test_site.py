"""What every page can know from the server's settings (the support address, the iPhone app's links), the short
addresses for the privacy and support pages, and Safari's banner for the iPhone app."""


def test_the_support_address_and_no_app_links_by_default(client):
    assert client.get("/api/site").json() == {"support_email": "support@aieverytime.org",
                                              "ios": {"app_store": None, "testflight": None}}
    assert "apple-itunes-app" not in client.get("/").text


def test_the_app_store_and_testflight_links_once_set(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "support_email", "help@example.org")
    monkeypatch.setattr(settings, "ios_testflight_url", "https://testflight.apple.com/join/AbCd1234")
    site = client.get("/api/site").json()
    assert site == {"support_email": "help@example.org",
                    "ios": {"app_store": None, "testflight": "https://testflight.apple.com/join/AbCd1234"}}
    monkeypatch.setattr(settings, "ios_app_id", "6739001234")
    assert client.get("/api/site").json()["ios"]["app_store"] == "https://apps.apple.com/app/id6739001234"
    # Safari shows its banner for the app on the site, once the app is on the App Store.
    page = client.get("/")
    assert '<meta name="apple-itunes-app" content="app-id=6739001234">' in page.text
    assert page.headers["cache-control"] == "no-cache" and page.headers["content-type"].startswith("text/html")


def test_settings_in_the_wrong_form_are_ignored(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("AIET_IOS_APP_ID", "id6739001234")
    monkeypatch.setenv("AIET_IOS_TESTFLIGHT_URL", "https://example.com/join/AbCd1234")
    monkeypatch.setenv("AIET_SUPPORT_EMAIL", "  ")
    s = Settings()
    assert (s.ios_app_id, s.ios_testflight_url, s.support_email) == ("", "", "support@aieverytime.org")


def test_privacy_and_support_have_plain_addresses(client):
    # For App Store Connect and emails: the pages themselves, open to everyone.
    for page, title in (("privacy", "Privacy"), ("support", "Help and support")):
        r = client.get(f"/{page}", follow_redirects=False)
        assert r.status_code == 200 and f"<title>{title} · AI Every Time</title>" in r.text
        assert '<script src="/static/app.js"></script>' in r.text
