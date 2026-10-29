"""Account security and privacy: password reset by email, changing the password, signing out
other devices, limits on password guessing, downloading your data and deleting your account."""

import json
import time

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_email import mailbox  # noqa: F401  (fixture)
from tests.test_push import Device, subscribe


def login(client, email="ada@example.com", password="correct horse"):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def other_device(client) -> str:
    """Sign in on another phone or browser: a second session for the current learner."""
    from app import auth

    return auth.create_session(client.get("/api/me").json()["id"])


def session_works(client, token) -> bool:
    """Whether that other device is still signed in. An explicit Cookie header takes the place of the jar's."""
    return client.get("/api/me", headers={"Cookie": f"aiet_session={token}"}).status_code == 200


def test_forgotten_password_reset_by_email(client, mailbox, monkeypatch):
    from app import db
    from app.config import settings

    signup(client)
    mailbox.sent.clear()  # the link that confirms the address
    old_session = other_device(client)
    monkeypatch.setattr(settings, "push_contact", "mailto:team@aieverytime.example")
    subscribe(client, Device())
    client.post("/api/auth/logout")

    assert client.post("/api/auth/forgot", json={"email": "nobody@example.com"}).status_code == 202
    assert mailbox.sent == []  # same answer, no email
    assert client.post("/api/auth/forgot", json={"email": "ADA@example.com"}).status_code == 202
    [msg] = mailbox.sent
    assert msg["To"] == "ada@example.com" and "List-Unsubscribe" not in msg
    token = mailbox.reset_link()
    assert f"https://aieverytime.example/reset-password/{token}" in msg.get_body(("plain",)).get_content()
    with db.connect() as conn:  # only a hash of the token is kept
        assert conn.execute("SELECT COUNT(*) FROM password_resets WHERE token_hash = ?", (token,)).fetchone()[0] == 0

    assert client.post("/api/auth/reset", json={"token": token, "password": "short"}).status_code == 422
    assert client.post("/api/auth/reset", json={"token": "x" * 43, "password": "a new password"}).status_code == 400
    r = client.post("/api/auth/reset", json={"token": token, "password": "a new password"})
    assert r.status_code == 204
    assert client.get("/api/me").json()["email"] == "ada@example.com"  # signed in here
    assert not session_works(client, old_session)  # and signed out everywhere else
    assert client.get("/api/push/devices").json() == []  # notifications stopped too
    assert client.post("/api/auth/reset", json={"token": token, "password": "another password"}).status_code == 400

    client.post("/api/auth/logout")
    assert login(client).status_code == 401
    assert login(client, password="a new password").status_code == 200


def test_reset_links_expire_and_are_limited(client, mailbox):
    from app import account, db

    signup(client)
    mailbox.sent.clear()  # the link that confirms the address
    client.post("/api/auth/logout")
    for _ in range(account.RESETS_PER_HOUR + 2):
        assert client.post("/api/auth/forgot", json={"email": "ada@example.com"}).status_code == 202
    assert len(mailbox.sent) == account.RESETS_PER_HOUR  # no flood of emails
    # Using one link spends the others.
    assert client.post("/api/auth/reset", json={"token": mailbox.reset_link(0), "password": "a new password"}).status_code == 204
    assert client.post("/api/auth/reset", json={"token": mailbox.reset_link(1), "password": "a new password"}).status_code == 400

    with db.connect() as conn:  # an hour later
        conn.execute("DELETE FROM password_resets")
        conn.execute("UPDATE email_sends SET created_at = '2000-01-01T00:00:00+00:00'")
    client.post("/api/auth/logout")
    client.post("/api/auth/forgot", json={"email": "ada@example.com"})
    assert len(mailbox.sent) == account.RESETS_PER_HOUR + 1
    with db.connect() as conn:
        conn.execute("UPDATE password_resets SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert client.post("/api/auth/reset", json={"token": mailbox.reset_link(), "password": "a new password"}).status_code == 400


def test_reset_needs_email_set_up(client):
    signup(client)
    r = client.post("/api/auth/forgot", json={"email": "ada@example.com"})
    assert r.status_code == 409 and "isn't set up" in r.json()["detail"]


def test_securing_the_account_cancels_reset_links(client, mailbox):
    signup(client)
    client.post("/api/auth/forgot", json={"email": "ada@example.com"})
    assert client.post("/api/me/password", json={"current_password": "correct horse", "new_password": "a new password"}).status_code == 204
    assert client.post("/api/auth/reset", json={"token": mailbox.reset_link(), "password": "attacker's choice"}).status_code == 400

    client.post("/api/auth/forgot", json={"email": "ada@example.com"})
    assert client.post("/api/me/sign-out-others", json={}).status_code == 204
    assert client.post("/api/auth/reset", json={"token": mailbox.reset_link(), "password": "attacker's choice"}).status_code == 400
    client.post("/api/auth/logout")
    assert login(client, password="a new password").status_code == 200


def test_change_password_and_sign_out_other_devices(client, monkeypatch):
    from app.config import settings

    signup(client)
    other = other_device(client)
    assert client.get("/api/me/sessions").json() == {"others": 1}
    assert client.post("/api/me/password", json={"current_password": "wrong", "new_password": "a new password"}).status_code == 403
    assert client.post("/api/me/password", json={"current_password": "correct horse", "new_password": "short"}).status_code == 422
    assert client.post("/api/me/password", json={"current_password": "correct horse", "new_password": "a new password"}).status_code == 204
    assert client.get("/api/me").status_code == 200 and not session_works(client, other)
    assert client.get("/api/me/sessions").json() == {"others": 0}

    # Signing out the others keeps this device's notifications and stops theirs.
    monkeypatch.setattr(settings, "push_contact", "mailto:team@aieverytime.example")
    here, there = Device("https://fcm.googleapis.com/fcm/send/here"), Device("https://fcm.googleapis.com/fcm/send/there")
    subscribe(client, here)
    subscribe(client, there)
    others = [other_device(client), other_device(client)]
    assert client.post("/api/me/sign-out-others", json={"keep_device": here.endpoint}).status_code == 204
    assert not any(session_works(client, t) for t in others) and client.get("/api/me").status_code == 200
    assert [d["endpoint"] for d in client.get("/api/push/devices").json()] == [here.endpoint]
    client.post("/api/auth/logout")
    assert login(client, password="a new password").status_code == 200


def test_wrong_passwords_are_slowed_down(client):
    from app import account, db

    signup(client)
    client.post("/api/auth/logout")
    for _ in range(account.GUESSES):
        assert login(client, password="guess").status_code == 401
    r = login(client)  # even the right password waits now
    assert r.status_code == 429 and "Too many wrong passwords" in r.json()["detail"]
    assert "reset" not in r.json()["detail"]  # email isn't set up here, so resetting isn't offered
    # Unknown emails behave the same, so this doesn't reveal who has an account.
    for _ in range(account.GUESSES):
        login(client, email="nobody@example.com", password="guess")
    assert login(client, email="nobody@example.com", password="guess").status_code == 429

    with db.connect() as conn:  # the window passes
        conn.execute("UPDATE login_failures SET created_at = '2000-01-01T00:00:00+00:00'")
    assert login(client).status_code == 200
    # Confirming the password for account changes counts too.
    for _ in range(account.GUESSES):
        client.post("/api/me/password", json={"current_password": "guess", "new_password": "a new password"})
    assert client.request("DELETE", "/api/me", json={"password": "correct horse"}).status_code == 429


def test_download_my_data(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{mine['id']}/chat", json={"message": "My name is Ada and I want to learn reinforcement learning."})
    for _ in range(50):
        if client.get(f"/api/companions/{mine['id']}/memories").json():
            break
        time.sleep(0.05)
    client.get("/api/lessons/what-is-ai")
    quiz = client.post("/api/assessments", json={"source_type": "lesson", "source_ref": "what-is-ai"}).json()
    # Answers stay on the server until the quiz is submitted, in the download too.
    [unsubmitted] = client.get("/api/me/export").json()["quizzes_and_exams"]
    assert "answer_index" not in json.dumps(unsubmitted) and "explanation" not in json.dumps(unsubmitted)
    client.post(f"/api/assessments/{quiz['id']}/submit", json={"answers": [0] * quiz["total"]})
    [submitted] = client.get("/api/me/export").json()["quizzes_and_exams"]
    assert all("answer_index" in q and "your_answer" in q for q in submitted["questions"])
    as_user(client, "bo@example.com", "Bo")
    assert client.post("/api/companions/1/chat", json={"message": "Bo's private question"}).status_code == 200
    as_user(client, "ada@example.com", "Ada")

    r = client.get("/api/me/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    data = r.json()
    assert data["account"]["email"] == "ada@example.com" and data["account"]["role"] == "learner"
    [designed] = data["companions_you_designed"]
    assert designed["persona"]["custom_instructions"] == persona()["custom_instructions"] and designed["versions"]
    assert [m["role"] for m in data["conversations"]] == ["user", "assistant"] and data["conversations"][0]["companion"] == "Zara"
    assert any(m["fact"] == "Prefers to be called Ada" for m in data["memories"])
    assert any(l["lesson_id"] == "what-is-ai" for l in data["lessons"]) and data["study_days"]
    text = r.text
    for secret in ("password_hash", "scrypt$", "unsubscribe_token", "Bo's private question", client.cookies.get("aiet_session")):
        assert secret not in text, secret


def test_delete_account(client):
    from app import db

    signup(client)
    zara = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{zara['id']}/publish")
    client.post(f"/api/companions/{zara['id']}/chat", json={"message": "Hello"})
    ada_id = client.get("/api/me").json()["id"]
    as_user(client, "bo@example.com", "Bo")
    client.post(f"/api/companions/{zara['id']}/primary")
    as_user(client, "ada@example.com", "Ada")

    assert client.request("DELETE", "/api/me", json={"password": "wrong"}).status_code == 403
    assert client.request("DELETE", "/api/me", json={"password": "correct horse"}).status_code == 204
    assert client.get("/api/me").status_code == 401
    assert login(client).status_code == 401
    with db.connect() as conn:
        for table, col in (("users", "id"), ("companions", "owner_id"), ("messages", "user_id"), ("sessions", "user_id")):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (ada_id,)).fetchone()[0] == 0, table

    # Bo's account is fine; the community companion Ada designed is gone with it.
    as_user(client, "bo@example.com", "Bo")
    assert client.get("/api/me").json()["primary_companion_id"] is None
    assert client.get(f"/api/companions/{zara['id']}").status_code == 404
    # The email address is free again.
    client.post("/api/auth/logout")
    assert signup(client)["email"] == "ada@example.com"
