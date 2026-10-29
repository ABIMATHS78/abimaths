"""Signing for the iOS release workflow (.github/workflows/ios-release.yml), which runs on one of GitHub's Macs.

Apple signs nothing for us: the app is signed with an Apple Distribution certificate, which Apple issues for a
private key we hold. The workflow starts on a fresh Mac every time, so that key has to be one it can find again,
or each run would ask Apple for another certificate and soon reach the few a team may have. Instead of a second
secret to make and look after, the key is worked out from the App Store Connect API key the workflow already
has: the same API key always gives the same certificate key, so the workflow finds the certificate it made the
first time. Anyone holding the API key can already make certificates for the team, so this gives nothing away.
A new API key brings a new certificate; the old one can then be revoked at developer.apple.com.

    python3 signing.py certificate-key OUT                  the key (from APP_STORE_CONNECT_PRIVATE_KEY) to OUT
    python3 signing.py profile DIR BUNDLE_ID                the App Store profile for the app in DIR: UUID TEAM
    python3 signing.py export-options TEAM UUID BUNDLE_ID OUT   xcodebuild's options for an App Store export
    python3 signing.py check-app APP TEAM BUNDLE_ID         the signed app has the App Store's entitlements
"""
from __future__ import annotations

import hashlib
import hmac
import math
import os
import plistlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

# Changing this, or how the key is made from it, means a new certificate on the next run.
LABEL = b"AI Every Time: Apple Distribution certificate key, version 1"
BITS = 2048
E = 65537
SMALL_PRIMES = [n for n in range(3, 2000, 2) if all(n % d for d in range(3, math.isqrt(n) + 1, 2))]


class _Stream:
    """Bytes that look random but follow from the seed: HMAC-SHA512 of the seed over a counter."""

    def __init__(self, seed: bytes):
        self.seed, self.counter, self.buffer = seed, 0, b""

    def take(self, n: int) -> bytes:
        while len(self.buffer) < n:
            self.buffer += hmac.new(self.seed, self.counter.to_bytes(8, "big"), hashlib.sha512).digest()
            self.counter += 1
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def number(self, bits: int) -> int:
        return int.from_bytes(self.take(bits // 8), "big")


def _probably_prime(n: int, stream: _Stream, rounds: int = 40) -> bool:
    """Trial division by small primes, then Miller-Rabin with bases from the stream."""
    for p in SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for _ in range(rounds):
        x = pow(2 + stream.number(n.bit_length()) % (n - 3), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(stream: _Stream) -> int:
    """A prime of half the key's size, with its top two bits set so the product has the full size."""
    half = BITS // 2
    while True:
        n = stream.number(half) | (3 << (half - 2)) | 1
        if (n - 1) % E and _probably_prime(n, stream):
            return n


def _api_key(text: str) -> ec.EllipticCurvePrivateKey:
    """The App Store Connect API key, from the contents of its .p8 file as pasted into the secret."""
    text = text.strip()
    if "\n" not in text:
        text = text.replace("\\n", "\n")  # pasted with its line breaks written out
    try:
        key = serialization.load_pem_private_key(text.encode(), password=None)
    except ValueError:
        raise SystemExit("APP_STORE_CONNECT_PRIVATE_KEY isn't a private key: paste the whole .p8 file, "
                         "from -----BEGIN PRIVATE KEY----- to -----END PRIVATE KEY-----.") from None
    if not isinstance(key, ec.EllipticCurvePrivateKey) or key.curve.name != "secp256r1":
        raise SystemExit("APP_STORE_CONNECT_PRIVATE_KEY isn't an App Store Connect API key (.p8).")
    return key


def certificate_key(api_key_text: str) -> rsa.RSAPrivateKey:
    """The RSA key for the distribution certificate: always the same for the same API key."""
    secret = _api_key(api_key_text).private_numbers().private_value.to_bytes(32, "big")
    stream = _Stream(hmac.new(LABEL, secret, hashlib.sha512).digest())
    p = _prime(stream)
    while True:
        q = _prime(stream)
        # As the US standard for RSA keys asks (FIPS 186-5): p and q far apart, and d large.
        if abs(p - q) >> (BITS // 2 - 100):
            d = pow(E, -1, (p - 1) * (q - 1) // math.gcd(p - 1, q - 1))
            if d >> (BITS // 2):
                break
    p, q = max(p, q), min(p, q)
    return rsa.RSAPrivateNumbers(
        p=p, q=q, d=d, dmp1=rsa.rsa_crt_dmp1(d, p), dmq1=rsa.rsa_crt_dmq1(d, q), iqmp=rsa.rsa_crt_iqmp(p, q),
        public_numbers=rsa.RSAPublicNumbers(E, p * q),
    ).private_key()


def key_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                             serialization.NoEncryption())


def pick_profile(profiles: list[dict], bundle_id: str, now: datetime | None = None) -> dict | None:
    """The newest App Store profile for the app that lets it get notifications, if there is one.

    App Store profiles list no devices and don't allow debugging; development and ad hoc ones do."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)

    def fits(profile: dict) -> bool:
        team = (profile.get("TeamIdentifier") or [""])[0]
        entitlements = profile.get("Entitlements") or {}
        return (entitlements.get("application-identifier") == f"{team}.{bundle_id}"
                and entitlements.get("aps-environment") == "production"
                and not entitlements.get("get-task-allow")
                and not profile.get("ProvisionedDevices") and not profile.get("ProvisionsAllDevices")
                and profile.get("ExpirationDate", now) > now)

    usable = [p for p in profiles if fits(p)]
    return max(usable, key=lambda p: p.get("CreationDate") or datetime.min) if usable else None


def export_options(team_id: str, profile_uuid: str, bundle_id: str) -> bytes:
    """What xcodebuild -exportArchive needs to make an .ipa for App Store Connect with our certificate."""
    return plistlib.dumps({
        "method": "app-store-connect",
        "destination": "export",
        "teamID": team_id,
        "signingStyle": "manual",
        "signingCertificate": "Apple Distribution",
        "provisioningProfiles": {bundle_id: profile_uuid},
        "manageAppVersionAndBuildNumber": False,
        "uploadSymbols": True,
    })


def entitlement_problems(entitlements: dict, team_id: str, bundle_id: str) -> list[str]:
    """What's wrong, if anything, with a signed app's entitlements for the App Store."""
    problems = []
    if entitlements.get("application-identifier") != f"{team_id}.{bundle_id}":
        problems.append(f"it's signed as {entitlements.get('application-identifier')!r}, not {team_id}.{bundle_id}")
    if entitlements.get("aps-environment") != "production":
        problems.append(f"notifications are set to {entitlements.get('aps-environment')!r}, not 'production'")
    if entitlements.get("get-task-allow"):
        problems.append("it's signed for debugging (get-task-allow), which the App Store refuses")
    return problems


def _write_secret(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def main(argv: list[str]) -> None:
    command, args = (argv[0], argv[1:]) if argv else ("", [])
    if command == "certificate-key" and len(args) == 1:
        api_key = os.environ.get("APP_STORE_CONNECT_PRIVATE_KEY", "")
        if not api_key:
            raise SystemExit("APP_STORE_CONNECT_PRIVATE_KEY isn't set.")
        pem = key_pem(certificate_key(api_key))
        if os.environ.get("GITHUB_ACTIONS") == "true":
            # GitHub hides its own secrets in the log, but doesn't know this one: hide every line of it too.
            for line in pem.decode().splitlines():
                if not line.startswith("-----"):
                    print(f"::add-mask::{line}")
        _write_secret(Path(args[0]), pem)
        print("Worked out the certificate key from the App Store Connect API key.")
    elif command == "profile" and len(args) == 2:
        folder, bundle_id = Path(args[0]), args[1]
        profiles = [plistlib.loads(subprocess.run(["security", "cms", "-D", "-i", str(path)], check=True,
                                                  capture_output=True).stdout)
                    for path in sorted(folder.glob("*.mobileprovision"))]
        profile = pick_profile(profiles, bundle_id)
        if not profile:
            raise SystemExit(f"None of the {len(profiles)} profiles in {folder} is an App Store profile for "
                             f"{bundle_id} with Push Notifications.")
        print(profile["UUID"], profile["TeamIdentifier"][0])
    elif command == "export-options" and len(args) == 4:
        team_id, profile_uuid, bundle_id, out = args
        Path(out).write_bytes(export_options(team_id, profile_uuid, bundle_id))
    elif command == "check-app" and len(args) == 3:
        app, team_id, bundle_id = args
        signed = subprocess.run(["codesign", "-d", "--entitlements", "-", "--xml", app], check=True,
                                capture_output=True).stdout
        problems = entitlement_problems(plistlib.loads(signed) if signed.strip() else {}, team_id, bundle_id)
        if problems:
            raise SystemExit("The signed app isn't ready for the App Store: " + "; ".join(problems) + ".")
        print(f"Signed for the App Store as {team_id}.{bundle_id}, with notifications.")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
