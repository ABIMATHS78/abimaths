"""Notifications to the iPhone app, through Apple's push service (APNs).

The app gives the server its device token, which is kept in the learner's list of devices beside their browsers
(app/push.py), so daily reminders and notices reach it the same way. Each message goes over HTTP/2, which Apple's
service requires, with a short-lived token signed by the team's APNs key (ES256) to show it comes from us.

Tokens from builds run straight from Xcode belong to Apple's sandbox service; TestFlight and App Store builds use
the production one. The app says which, and the device's address records it: apns:<environment>:<token>.
"""

from __future__ import annotations

import base64
import functools
import json
import logging
import re
import threading
import time

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from app.config import settings

log = logging.getLogger(__name__)

HOSTS = {"production": "https://api.push.apple.com", "sandbox": "https://api.sandbox.push.apple.com"}
PREFIX = "apns:"
DEVICE_TOKEN = re.compile(r"[0-9a-f]{64,200}")
# Apple accepts a signed token for an hour, and refuses new ones made more often than every 20 minutes.
TOKEN_LIFETIME = 40 * 60
# Reasons Apple gives when a token will never work again: the app was removed, or it's another app's or service's.
GONE = {"BadDeviceToken", "DeviceTokenNotForTopic", "Unregistered", "ExpiredToken"}
# Reasons that mean the server's own APNs settings are wrong, so no device is to blame.
OURS = {"InvalidProviderToken", "ExpiredProviderToken", "MissingProviderToken", "BadTopic", "TopicDisallowed",
        "MissingTopic", "Forbidden", "BadCertificate", "BadCertificateEnvironment"}
# Tests put a stand-in for Apple's service here. Otherwise one HTTP/2 client is kept open, as Apple asks.
HTTP: httpx.Client | None = None
_client: httpx.Client | None = None
_lock = threading.Lock()
_signed: dict[tuple, tuple[str, float]] = {}


@functools.lru_cache(maxsize=4)
def _load(text: str) -> ec.EllipticCurvePrivateKey | None:
    try:
        key = serialization.load_pem_private_key(text.encode(), password=None)
    except (ValueError, TypeError, AttributeError) as e:
        log.error("AIET_APNS_KEY isn't the text of an APNs key (.p8 file): %s", e)
        return None
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        log.error("AIET_APNS_KEY isn't an APNs key: they're P-256 elliptic-curve keys")
        return None
    return key


def signing_key() -> ec.EllipticCurvePrivateKey | None:
    """The team's APNs key. Settings often keep line breaks as \\n, so those count as line breaks."""
    text = settings.apns_key.replace("\\n", "\n").strip()
    return _load(text) if text else None


def available() -> bool:
    return bool(settings.apns_key_id and settings.apns_team_id and settings.apns_bundle_id) and signing_key() is not None


def endpoint(environment: str, token: str) -> str:
    return f"{PREFIX}{environment}:{token}"


def parse(address: str) -> tuple[str, str] | None:
    """The environment and device token in an app's address, or None if it isn't a well-formed one."""
    if not address.startswith(PREFIX):
        return None
    environment, _, token = address[len(PREFIX):].partition(":")
    return (environment, token) if environment in HOSTS and DEVICE_TOKEN.fullmatch(token) else None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def provider_token() -> str:
    """The signed token that shows Apple a message comes from us, made again every 40 minutes."""
    key, who = signing_key(), (settings.apns_key_id, settings.apns_team_id, settings.apns_key)
    with _lock:
        made = _signed.get(who)
        if made and time.time() - made[1] < TOKEN_LIFETIME:
            return made[0]
        now = int(time.time())
        header = _b64(json.dumps({"alg": "ES256", "kid": settings.apns_key_id}, separators=(",", ":")).encode())
        claims = _b64(json.dumps({"iss": settings.apns_team_id, "iat": now}, separators=(",", ":")).encode())
        r, s = decode_dss_signature(key.sign(f"{header}.{claims}".encode(), ec.ECDSA(hashes.SHA256())))
        token = f"{header}.{claims}.{_b64(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"
        _signed.clear()
        _signed[who] = (token, now)
        return token


def _http() -> httpx.Client:
    global _client
    if HTTP is not None:
        return HTTP
    with _lock:
        if _client is None:
            _client = httpx.Client(http2=True, timeout=10, follow_redirects=False)
        return _client


def payload(data: dict) -> dict:
    """What the app shows: the same title and text as in a browser, and the page a tap opens."""
    return {"aps": {"alert": {"title": data["title"], "body": data["body"]}, "sound": "default"}, "url": data["url"]}


def send(address: str, data: dict, ttl: int) -> str:
    """Send one message to the app on one device: "sent", "gone" (the app was removed, or the token will never
    work), "failed" (Apple refused this message) or "retry" (a problem on our side or Apple's)."""
    parsed = parse(address)
    if parsed is None:
        return "gone"
    if not available():
        return "retry"  # the settings were removed; the devices may work again once they're back
    environment, token = parsed
    headers = {
        "authorization": f"bearer {provider_token()}",
        "apns-topic": settings.apns_bundle_id,
        "apns-push-type": "alert",
        "apns-priority": "10",
        "apns-expiration": str(int(time.time()) + ttl),
        # A newer message with the same tag replaces the older one, as in browsers.
        "apns-collapse-id": data["tag"].encode()[:64].decode(errors="ignore"),
    }
    try:
        r = _http().post(f"{HOSTS[environment]}/3/device/{token}", headers=headers,
                         content=json.dumps(payload(data)).encode())
    except httpx.HTTPError as e:
        log.warning("notification to the app failed: %s", e)
        return "retry"
    if r.status_code == 200:
        return "sent"
    try:
        reason = r.json().get("reason", "")
    except ValueError:
        reason = ""
    if r.status_code == 410 or reason in GONE:
        return "gone"
    if r.status_code == 403 or reason in OURS:
        with _lock:
            _signed.clear()  # a new token is made for the next message, in case this one had expired
        log.error("Apple refused the app's notifications (%s): check AIET_APNS_KEY, AIET_APNS_KEY_ID, "
                  "AIET_APNS_TEAM_ID and AIET_APNS_BUNDLE_ID", reason or r.status_code)
        return "retry"
    if r.status_code == 429 or r.status_code >= 500:
        return "retry"
    log.warning("Apple refused a notification to the app: %s %s", r.status_code, reason)
    return "failed"
