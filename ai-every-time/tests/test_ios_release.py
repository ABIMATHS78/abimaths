"""Signing the iPhone app on GitHub's Macs (ios/release/signing.py): checking the App Store Connect API key from the
secrets, the certificate key worked out from it, choosing the App Store profile, the export options, and checking
the signed app."""

import importlib.util
import os
import plistlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

spec = importlib.util.spec_from_file_location("signing", Path(__file__).parents[1] / "ios" / "release" / "signing.py")
signing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signing)

BUNDLE = "org.aieverytime.app"
TEAM = "TEAM123456"
ISSUER, KEY_ID = "57246542-96fe-1a63-e053-0824d011072a", "2X9R4HXF34"
NOW = datetime(2026, 10, 29, 12, 0)


def api_key() -> str:
    """A key like the .p8 file App Store Connect gives: P-256, in PKCS#8."""
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()


def test_the_same_api_key_always_gives_the_same_certificate_key():
    first = api_key()
    started = time.monotonic()
    one = signing.key_pem(signing.certificate_key(first))
    assert time.monotonic() - started < 10
    assert signing.key_pem(signing.certificate_key(first)) == one
    # Pasted with its line breaks written out, or with spaces around it, it's still the same key.
    assert signing.key_pem(signing.certificate_key("  " + first.replace("\n", "\\n") + "\n")) == one
    assert signing.key_pem(signing.certificate_key(api_key())) != one


def test_the_certificate_key_is_a_sound_rsa_key():
    key = signing.certificate_key(api_key())
    assert isinstance(key, rsa.RSAPrivateKey) and key.key_size == 2048
    numbers = key.private_numbers()
    p, q, n = numbers.p, numbers.q, numbers.public_numbers.n
    assert numbers.public_numbers.e == 65537 and p * q == n and n.bit_length() == 2048
    for prime in (p, q):  # Fermat checks with bases the key wasn't made with
        assert all(pow(base, prime - 1, prime) == 1 for base in (3, 5, 7, 11, 13, 1_000_003))
    assert abs(p - q).bit_length() > 1024 - 100 and numbers.d.bit_length() > 1024
    signature = key.sign(b"AI Every Time", padding.PKCS1v15(), hashes.SHA256())
    key.public_key().verify(signature, b"AI Every Time", padding.PKCS1v15(), hashes.SHA256())
    # It's written in the form Apple's tools and the signing tool read.
    pem = signing.key_pem(key)
    assert pem.startswith(b"-----BEGIN RSA PRIVATE KEY-----")
    assert serialization.load_pem_private_key(pem, None).private_numbers() == numbers


def test_only_an_app_store_connect_key_is_accepted():
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    p384 = ec.generate_private_key(ec.SECP384R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    for wrong in (rsa_key, p384):
        with pytest.raises(SystemExit, match="isn't an App Store Connect API key"):
            signing.certificate_key(wrong)
    with pytest.raises(SystemExit, match="paste the whole .p8 file"):
        signing.certificate_key("ABCD1234")


def test_the_command_writes_the_key_privately_and_hides_it_in_the_log(tmp_path, monkeypatch, capsys):
    key = api_key()
    monkeypatch.setenv("APP_STORE_CONNECT_PRIVATE_KEY", key)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    out = tmp_path / "certificate.key"
    signing.main(["certificate-key", str(out)])
    pem = out.read_bytes()
    assert pem == signing.key_pem(signing.certificate_key(key))
    assert out.stat().st_mode & 0o777 == 0o600
    printed = capsys.readouterr().out
    body = [line for line in pem.decode().splitlines() if not line.startswith("-----")]
    assert all(f"::add-mask::{line}" in printed for line in body)
    assert all(line not in printed.replace(f"::add-mask::{line}", "") for line in body)

    monkeypatch.delenv("APP_STORE_CONNECT_PRIVATE_KEY")
    with pytest.raises(SystemExit, match="isn't set"):
        signing.main(["certificate-key", str(out)])


PASTED = {  # how the .p8 file can arrive in the secret
    "as downloaded": lambda pem: pem,
    "with Windows line breaks": lambda pem: pem.replace("\n", "\r\n"),
    "with its line breaks written out": lambda pem: pem.strip().replace("\n", "\\n"),
    "on one line": lambda pem: pem.strip().replace("\n", " "),
    "indented, between blank lines": lambda pem: "\n\n" + "".join(f"    {line}\n" for line in pem.splitlines()) + "\n",
    "after its file name": lambda pem: f"AuthKey_{KEY_ID}.p8\n{pem}",
}


@pytest.mark.parametrize("how", PASTED)
def test_a_pasted_key_comes_out_as_apple_wrote_it(how):
    pem = api_key()
    assert signing.p8(PASTED[how](pem)) == pem
    assert signing.api_key_secrets(ISSUER, KEY_ID, PASTED[how](pem)) == (ISSUER, KEY_ID, pem)


def read_github_env(text: str) -> dict:
    """$GITHUB_ENV as GitHub's runner reads it: NAME=value lines, and NAME<<DELIMITER blocks for longer values."""
    found, lines = {}, iter(text.split("\n"))
    for line in lines:
        equals, heredoc = line.find("="), line.find("<<")
        if equals >= 0 and (heredoc < 0 or equals < heredoc):
            found[line[:equals]] = line[equals + 1:]
        elif heredoc >= 0:
            delimiter, value = line[heredoc + 2:], []
            for more in lines:
                if more == delimiter:
                    break
                value.append(more)
            else:
                raise AssertionError(f"{line[:heredoc]} has no end")
            found[line[:heredoc]] = "\n".join(value)
    return found


def test_the_api_key_goes_on_to_the_later_steps_cleaned_and_hidden(tmp_path, monkeypatch, capsys):
    key = api_key()
    github_env = tmp_path / "github_env"
    github_env.write_text("EARLIER=kept\n")
    # Pasted the way people paste: spaces and line breaks around the IDs, Windows line breaks in the key.
    for name, value in {"APP_STORE_CONNECT_ISSUER_ID": f" {ISSUER}\n", "APP_STORE_CONNECT_KEY_ID": f"{KEY_ID} \r\n",
                        "APP_STORE_CONNECT_PRIVATE_KEY": key.replace("\n", "\r\n"), "GITHUB_ENV": str(github_env),
                        "GITHUB_ACTIONS": "true"}.items():
        monkeypatch.setenv(name, value)
    signing.main(["api-key"])
    assert read_github_env(github_env.read_text()) == {
        "EARLIER": "kept", "APP_STORE_CONNECT_ISSUER_ID": ISSUER, "APP_STORE_CONNECT_KEY_IDENTIFIER": KEY_ID,
        "APP_STORE_CONNECT_PRIVATE_KEY": key.strip()}
    # Hidden in the log before anything could show them, and shown nowhere else.
    printed = capsys.readouterr().out.splitlines()
    hidden = {line.removeprefix("::add-mask::") for line in printed if line.startswith("::add-mask::")}
    shown = "\n".join(line for line in printed if not line.startswith("::add-mask::"))
    secret = [ISSUER, KEY_ID, *key.strip().splitlines()[1:-1]]
    assert set(secret) <= hidden and not any(part in shown for part in secret) and "look right" in shown

    monkeypatch.delenv("GITHUB_ENV")
    with pytest.raises(SystemExit, match="GITHUB_ENV"):
        signing.main(["api-key"])


SAME = lambda pem: pem  # noqa: E731


@pytest.mark.parametrize("issuer, key_id, key, title", [
    ("", "", lambda pem: "", "Apple secrets missing"),
    (ISSUER, KEY_ID, lambda pem: " \n", "Apple secrets missing"),
    (KEY_ID, ISSUER, SAME, "Issuer ID and Key ID swapped"),
    ("A1B2C3D4E5", KEY_ID, SAME, "Issuer ID"),  # the Team ID
    (f'"{ISSUER}"', KEY_ID, SAME, "Issuer ID"),
    (ISSUER, "GitHub release", SAME, "Key ID"),  # the key's name
    (ISSUER, KEY_ID.lower(), SAME, "Key ID"),
    (ISSUER, ISSUER, SAME, "Key ID"),
    (ISSUER, KEY_ID, lambda pem: f"AuthKey_{KEY_ID}.p8", "Private key"),
    (ISSUER, KEY_ID, lambda pem: pem.split("\n", 1)[1], "Private key"),  # without its first line
    (ISSUER, KEY_ID, lambda pem: pem.replace("\n-----END", "\nmy key\n-----END"), "Private key"),
    (ISSUER, KEY_ID, lambda pem: pem.replace("PRIVATE", "PUBLIC"), "Private key"),
    (ISSUER, KEY_ID, lambda pem: pem[:60] + pem[61:], "Private key"),  # a letter lost
])
def test_secrets_that_cant_be_right_say_what_to_change(issuer, key_id, key, title):
    with pytest.raises(signing.Problem) as problem:
        signing.api_key_secrets(issuer, key_id, key(api_key()))
    assert problem.value.title == title


def test_in_the_workflow_a_problem_shows_at_the_top_of_the_run(tmp_path):
    env = {**os.environ, "GITHUB_ACTIONS": "true", "GITHUB_ENV": str(tmp_path / "github_env"),
           "APP_STORE_CONNECT_ISSUER_ID": ISSUER, "APP_STORE_CONNECT_KEY_ID": "GitHub release",
           "APP_STORE_CONNECT_PRIVATE_KEY": api_key()}
    run = subprocess.run([sys.executable, spec.origin, "api-key"], env=env, capture_output=True, text=True)
    assert run.returncode == 1 and run.stderr == ""
    assert run.stdout.startswith("::error title=Key ID::APP_STORE_CONNECT_KEY_ID isn't a Key ID")
    assert not (tmp_path / "github_env").exists()


def profile(uuid, kind="store", bundle=BUNDLE, push=True, created=datetime(2026, 10, 1), expires=datetime(2027, 10, 1)):
    entitlements = {"application-identifier": f"{TEAM}.{bundle}", "get-task-allow": kind == "development"}
    if push:
        entitlements["aps-environment"] = "development" if kind == "development" else "production"
    found = {"UUID": uuid, "TeamIdentifier": [TEAM], "Entitlements": entitlements, "CreationDate": created,
             "ExpirationDate": expires}
    if kind in ("development", "adhoc"):
        found["ProvisionedDevices"] = ["00008110-000A1B2C3D4E5F6A"]
    return found


def test_the_newest_app_store_profile_with_notifications_is_used():
    assert signing.pick_profile([], BUNDLE, NOW) is None
    others = [profile("dev", kind="development"), profile("adhoc", kind="adhoc"), profile("nopush", push=False),
              profile("other", bundle="org.aieverytime.app.widget"), profile("old", expires=datetime(2026, 9, 1))]
    assert signing.pick_profile(others, BUNDLE, NOW) is None
    chosen = signing.pick_profile(others + [profile("first"), profile("newer", created=datetime(2026, 10, 20))],
                                  BUNDLE, NOW)
    assert chosen["UUID"] == "newer"


def test_export_options_sign_with_our_certificate_and_profile(tmp_path):
    out = tmp_path / "ExportOptions.plist"
    signing.main(["export-options", TEAM, "PROFILE-UUID", BUNDLE, str(out)])
    options = plistlib.loads(out.read_bytes())
    assert options == {
        "method": "app-store-connect", "destination": "export", "teamID": TEAM, "signingStyle": "manual",
        "signingCertificate": "Apple Distribution", "provisioningProfiles": {BUNDLE: "PROFILE-UUID"},
        "manageAppVersionAndBuildNumber": False, "uploadSymbols": True,
    }


def test_a_signed_app_is_checked_before_it_goes_to_apple():
    store = {"application-identifier": f"{TEAM}.{BUNDLE}", "aps-environment": "production", "get-task-allow": False}
    assert signing.entitlement_problems(store, TEAM, BUNDLE) == []
    development = {"application-identifier": f"{TEAM}.{BUNDLE}", "aps-environment": "development",
                   "get-task-allow": True}
    problems = signing.entitlement_problems(development, TEAM, BUNDLE)
    assert len(problems) == 2 and "'development'" in problems[0] and "debugging" in problems[1]
    assert len(signing.entitlement_problems({}, TEAM, BUNDLE)) == 2


def test_an_unknown_command_explains_the_commands():
    with pytest.raises(SystemExit, match="certificate-key OUT"):
        signing.main(["sign-everything"])
