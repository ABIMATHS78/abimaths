"""Notifications in the iPhone app, through Apple's push service: turning them on, signed delivery, the page a tap
opens, devices Apple says are gone, and the daily reminders and notices that browsers get too."""

import base64
import json
import time
from datetime import datetime, timezone

import httpx
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_push import NIGHT, Device, run, service, subscribe  # noqa: F401  (service is a fixture)

APP = ("Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
       "Mobile/15E148 AIEveryTimeApp/1.0 (iOS)")
TOKEN = "A1B2C3D4" * 8  # what the app hands over: 32 bytes as hex, which Apple may print in capitals


def unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class Apple:
    """Stands in for Apple's push service: records each request and answers with a status and, if refused, a reason."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.answers: dict[str, tuple[int, str]] = {}
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, reason = self.answers.get(request.url.path.rsplit("/", 1)[-1], (200, ""))
        return httpx.Response(status, json={"reason": reason} if reason else None)

    def to(self, token: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == f"/3/device/{token.lower()}"]


@pytest.fixture()
def apple(client, monkeypatch):
    from app import apns
    from app.config import settings

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    # Pasted into a setting, a key's line breaks often become \n.
    monkeypatch.setattr(settings, "apns_key", pem.decode().replace("\n", "\\n"))
    monkeypatch.setattr(settings, "apns_key_id", "KEY1234567")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123456")
    monkeypatch.setattr(settings, "apns_bundle_id", "org.aieverytime.app")
    fake = Apple()
    fake.public_key = key.public_key()
    monkeypatch.setattr(apns, "HTTP", fake.client)
    monkeypatch.setattr(apns, "_signed", {})
    return fake


def turn_on(client, token=TOKEN, environment="production", **extra):
    return client.post("/api/push/apns", json={"token": token, "environment": environment, **extra}, headers={"User-Agent": APP})


def test_the_app_needs_the_servers_apple_settings(client, monkeypatch):
    from app import apns
    from app.config import settings

    signup(client)
    assert client.get("/api/me").json()["app_push_available"] is False
    assert turn_on(client).status_code == 409
    monkeypatch.setattr(settings, "apns_key_id", "KEY1234567")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123456")
    monkeypatch.setattr(settings, "apns_key", "not a key")
    assert not apns.available()  # a key that can't be read counts as none
    # APNs keys are P-256 elliptic-curve keys; any other key is refused.
    for other in (rsa.generate_private_key(65537, 2048), ec.generate_private_key(ec.SECP384R1())):
        pem = other.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        monkeypatch.setattr(settings, "apns_key", pem.decode())
        assert not apns.available()


def test_the_app_turns_notifications_on_with_its_token(client, apple):
    signup(client)
    assert client.get("/api/me").json()["app_push_available"] is True
    r = turn_on(client, environment="sandbox")
    assert r.status_code == 201
    [device] = r.json()
    assert device["endpoint"] == f"apns:sandbox:{TOKEN.lower()}" and device["device"] == "AI Every Time app on iPhone"
    for bad in ({"token": "z" * 64}, {"token": "ab" * 10}, {"token": "ab" * 101}, {"environment": "staging"}):
        assert client.post("/api/push/apns", json={"token": TOKEN} | bad).status_code == 422, bad
    # A new token replaces the device's old one, and turning off forgets it.
    new = "ff" * 32
    [device] = turn_on(client, token=new, replaces=f"apns:sandbox:{TOKEN.lower()}").json()
    assert device["endpoint"] == f"apns:production:{new}"
    assert client.post("/api/push/unsubscribe", json={"endpoint": device["endpoint"]}).status_code == 204
    assert client.get("/api/push/devices").json() == []


def test_app_notifications_are_signed_for_apple_and_open_the_page(client, apple):
    signup(client)
    starter = client.get("/api/companions").json()[0]
    client.post(f"/api/companions/{starter['id']}/primary")
    turn_on(client)

    before = time.time()
    assert client.post("/api/push/test").json() == {"sent": 1}
    [req] = apple.to(TOKEN)
    assert str(req.url) == f"https://api.push.apple.com/3/device/{TOKEN.lower()}" and req.method == "POST"
    h = req.headers
    assert (h["apns-topic"], h["apns-push-type"], h["apns-priority"], h["apns-collapse-id"]) == (
        "org.aieverytime.app", "alert", "10", "test")
    assert before + 600 - 2 <= int(h["apns-expiration"]) <= time.time() + 600
    # Signed with the team's key (ES256), naming the key and the team, as Apple checks.
    header, claims, signature = h["authorization"].removeprefix("bearer ").split(".")
    assert json.loads(unb64(header)) == {"alg": "ES256", "kid": "KEY1234567"}
    assert json.loads(unb64(claims))["iss"] == "TEAM123456" and abs(json.loads(unb64(claims))["iat"] - before) < 5
    raw = unb64(signature)
    der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    apple.public_key.verify(der, f"{header}.{claims}".encode(), ec.ECDSA(hashes.SHA256()))
    with pytest.raises(InvalidSignature):
        apple.public_key.verify(der, f"{header}.{claims}x".encode(), ec.ECDSA(hashes.SHA256()))
    # What the phone shows, from the learner's own companion, and the page a tap opens.
    body = json.loads(req.content)
    assert body == {"aps": {"alert": {"title": f"{starter['persona']['name']} {starter['persona']['flag']}",
                                      "body": body["aps"]["alert"]["body"]}, "sound": "default"}, "url": "/#/profile"}
    assert body["aps"]["alert"]["body"].startswith("Notifications are on")
    # The signed token is used again rather than made for every message; Apple refuses new ones too often.
    client.post("/api/push/test")
    assert apple.requests[-1].headers["authorization"] == h["authorization"]


def test_apple_says_which_app_devices_are_gone(client, apple):
    from app import push

    signup(client)
    removed, wrong_app, refused, outage = ("a1" * 32, "b2" * 32, "c3" * 32, "d4" * 32)
    for token in (removed, wrong_app, refused, outage):
        turn_on(client, token=token)
    apple.answers = {removed: (410, "Unregistered"), wrong_app: (400, "DeviceTokenNotForTopic"),
                     refused: (400, "BadCollapseId"), outage: (503, "ServiceUnavailable")}
    assert client.post("/api/push/test").status_code == 409  # nothing got through
    left = {d["endpoint"] for d in client.get("/api/push/devices").json()}
    assert left == {f"apns:production:{refused}", f"apns:production:{outage}"}
    for _ in range(push.MAX_FAILURES - 1):
        client.post("/api/push/test")
    # A refused message counts against a device; Apple's own outage doesn't.
    assert {d["endpoint"] for d in client.get("/api/push/devices").json()} == {f"apns:production:{outage}"}


def test_the_servers_own_mistakes_dont_cost_devices(client, apple, caplog):
    signup(client)
    turn_on(client)
    apple.answers = {TOKEN.lower(): (403, "InvalidProviderToken")}
    assert client.post("/api/push/test").status_code == 409
    assert "check AIET_APNS_KEY" in caplog.text
    assert len(client.get("/api/push/devices").json()) == 1  # the device isn't to blame
    first = apple.requests[-1].headers["authorization"]
    apple.answers = {}
    assert client.post("/api/push/test").json() == {"sent": 1}
    # In case the signed token was what Apple refused, the next message has a new one.
    assert apple.requests[-1].headers["authorization"] != first


def test_daily_reminders_and_notices_reach_the_app(client, apple, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    r = client.post("/api/auth/signup", json={"email": "ken@example.com", "password": "correct horse",
                                               "display_name": "Ken", "ai_consent": True, "timezone": "Asia/Tokyo"})
    assert r.status_code == 201
    client.patch("/api/me", json={"reminder_hour": 18})
    client.get("/api/lessons/tokens")  # something to remind about
    turn_on(client)

    assert run(datetime(2026, 3, 10, 9, 5, tzinfo=timezone.utc)) == 1  # 18:05 in Tokyo
    [req] = apple.to(TOKEN)
    body = json.loads(req.content)
    assert req.headers["apns-collapse-id"] == "daily" and body["url"] == "/#/lesson/tokens"
    assert "Pick up where you left off" in body["aps"]["alert"]["body"]

    zara = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{zara['id']}/publish")
    for i in range(3):
        as_user(client, f"r{i}@example.com", f"R{i}")
        client.post(f"/api/companions/{zara['id']}/report", json={"reason": "misleading"})
    as_user(client, "boss@example.com", "Boss")
    client.post(f"/api/staff/companions/{zara['id']}", json={"action": "restore"})
    assert run(NIGHT) == 1
    body = json.loads(apple.to(TOKEN)[-1].content)
    assert body["aps"]["alert"]["title"] == "AI Every Time" and body["url"] == f"/#/c/{zara['id']}"


def test_a_browser_and_the_app_both_get_it(client, apple, service):
    signup(client)
    phone = Device()
    subscribe(client, phone)
    turn_on(client)
    devices = client.get("/api/push/devices").json()
    assert {d["device"] for d in devices} == {"Chrome on Android", "AI Every Time app on iPhone"}
    assert client.post("/api/push/test").json() == {"sent": 2}
    assert len(service.to(phone)) == 1 and len(apple.to(TOKEN)) == 1
    # Signing out the other devices keeps the one asking, here the app.
    r = client.post("/api/me/sign-out-others", json={"keep_device": f"apns:production:{TOKEN.lower()}"})
    assert r.status_code == 204, r.text
    assert [d["device"] for d in client.get("/api/push/devices").json()] == ["AI Every Time app on iPhone"]


def test_the_server_can_reach_apple_over_http2():
    from app import apns

    assert isinstance(apns._http(), httpx.Client)  # made with http2=True, which needs the h2 package Apple requires
