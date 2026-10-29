"""Invite links, invites that count once the friend confirms their address, and the free month of Plus that every
three earn."""

import pytest

from tests.test_app import signup
from tests.test_billing import FakeStripe, _paying, paywall, plan, send_event, sql, stripe_fake  # noqa: F401  (fixtures)
from tests.test_community import as_user
from tests.test_email import mailbox  # noqa: F401  (fixture)


def invite_code(client) -> str:
    r = client.get("/api/referrals").json()
    assert r["link"].endswith(f"/r/{r['code']}")
    return r["code"]


def join(client, code, email, name="Friend", box=None):
    """A friend signs up with the code, and with a mailbox, confirms their address."""
    client.post("/api/auth/logout")
    r = client.post("/api/auth/signup", json={"email": email, "password": "correct horse", "display_name": name,
                                              "referral_code": code})
    assert r.status_code == 201, r.text
    if box is not None:
        client.post("/api/email/confirm", data={"token": box.link_token("confirm")})


def test_three_friends_earn_a_free_month_of_plus(client, paywall, mailbox):
    signup(client, email="ada@example.com", name="Ada")
    code = invite_code(client)
    assert invite_code(client) == code  # made once, then kept
    for i in range(3):
        join(client, code, f"friend{i}@example.com", f"Friend{i} Surname", mailbox)
    join(client, "NOTACODE2", "stranger@example.com", box=mailbox)  # an unknown code is ignored
    as_user(client, "ada@example.com", "Ada")
    r = client.get("/api/referrals").json()
    assert r["counted"] == 3 and r["toward_next"] == 0 and r["earned"] == 1 and r["to_claim"] == 1
    assert [i["status"] for i in r["invites"]] == ["joined"] * 3
    assert r["invites"][0]["name"].startswith("Friend") and "Surname" not in r["invites"][0]["name"]
    assert "email" not in r["invites"][0]  # their first name only
    assert any("free month of Plus" in n["text"] for n in sql("SELECT text FROM notices"))
    claimed = client.post("/api/referrals/claim").json()
    assert claimed["to_claim"] == 0 and "Plus" in claimed["message"]
    p = plan(client)
    assert p["tier"] == "plus" and p["reward"]["tier"] == "plus" and p["reward"]["until"] and p["grant"] is None
    assert client.post("/api/referrals/claim").status_code == 409
    assert sql("SELECT signup_source FROM users WHERE email = 'friend0@example.com'")[0]["signup_source"] == "referral"


def test_your_own_code_doesnt_count(client, paywall):
    signup(client)
    from app import db, referrals

    code = invite_code(client)
    with db.connect() as conn:
        assert referrals.attribute(conn, sql("SELECT id FROM users")[0]["id"], code) is False


def test_an_invite_counts_once_the_friend_confirms_their_address(client, paywall, mailbox):
    signup(client, email="ada@example.com", name="Ada")
    code = invite_code(client)
    join(client, code, "kofi@example.com", "Kofi")
    token = mailbox.link_token("confirm")
    as_user(client, "ada@example.com", "Ada")
    assert client.get("/api/referrals").json()["invites"][0]["status"] == "unconfirmed"
    client.post("/api/email/confirm", data={"token": token})
    r = client.get("/api/referrals").json()
    assert r["invites"][0]["status"] == "joined" and r["counted"] == 1 and r["toward_next"] == 1


def test_the_invite_link_remembers_the_code_for_sign_up(client, paywall):
    signup(client, email="ada@example.com", name="Ada")
    code = invite_code(client)
    client.post("/api/auth/logout")
    r = client.get(f"/r/{code.lower()}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == f"/signup?ref={code}"
    assert client.cookies.get("aiet_ref") == code
    client.post("/api/auth/signup", json={"email": "kofi@example.com", "password": "correct horse", "display_name": "Kofi"})
    assert sql("SELECT COUNT(*) AS n FROM referral_invites")[0]["n"] == 1
    assert client.get("/r/<script>", follow_redirects=False).headers["location"] == "/signup"


def test_a_friend_who_pays_shows_as_subscribed(client, stripe_fake):
    signup(client, email="ada@example.com", name="Ada")
    code = invite_code(client)
    client.post("/api/auth/logout")
    client.post("/api/auth/signup", json={"email": "kofi@example.com", "password": "correct horse", "display_name": "Kofi",
                                          "referral_code": code})
    from tests.test_billing import me_id, subscription

    stripe_fake.subs["sub_k"] = subscription("sub_k", "cus_k", me_id(client), "digital")
    send_event(client, "customer.subscription.created", stripe_fake.subs["sub_k"])
    as_user(client, "ada@example.com", "Ada")
    assert client.get("/api/referrals").json()["invites"][0]["status"] == "subscribed"


def test_members_already_paying_for_plus_get_it_off_their_bill(client, stripe_fake):
    uid = _paying(client, stripe_fake, "pro")
    from app import db

    with db.connect() as conn:
        conn.execute("INSERT INTO referral_rewards (user_id, created_at) VALUES (?, ?)", (uid, db.now()))
    stripe_fake.fail.add("balance.create")
    assert client.post("/api/referrals/claim").status_code == 502
    assert client.get("/api/referrals").json()["to_claim"] == 1  # still there to claim
    stripe_fake.fail.clear()
    r = client.post("/api/referrals/claim").json()
    assert "$2.99 comes off your next bill" in r["message"]
    (_, args, kw) = stripe_fake.named("balance.create")[-1]
    assert args[0] == "cus_1" and kw["params"]["amount"] == -299 and kw["options"]["idempotency_key"].endswith(
        str(sql("SELECT id FROM referral_rewards")[0]["id"]))
    assert sql("SELECT applied_as FROM referral_rewards")[0]["applied_as"] == "credit"
    assert plan(client)["grant"] is None


def test_a_reward_month_leaves_a_given_plan_as_it_is(client, paywall):
    from tests.test_billing import me_id, run

    signup(client)
    uid = me_id(client)
    run("UPDATE subscriptions SET grant_tier = 'digital', grant_until = NULL, grant_note = 'Staff' WHERE user_id = ?", uid)
    run("INSERT INTO referral_rewards (user_id, created_at) VALUES (?, '2026-01-01T00:00:00+00:00'), "
        "(?, '2026-01-01T00:00:00+00:00')", uid, uid)
    client.post("/api/referrals/claim")
    first = plan(client)["reward"]["until"]
    client.post("/api/referrals/claim")  # a second month follows the first
    p = plan(client)
    assert p["tier"] == "plus" and p["reward"]["until"] > first
    assert p["grant"] == {"tier": "digital", "name": "Digital", "until": None, "note": "Staff"}  # for good, still
    run("UPDATE subscriptions SET reward_until = '2001-01-01T00:00:00+00:00' WHERE user_id = ?", uid)  # the months ran out
    assert plan(client)["tier"] == "digital"


def test_invites_need_a_confirmed_mailbox_of_their_own(client, paywall, mailbox):
    signup(client, email="ada@gmail.com", name="Ada")
    client.post("/api/email/confirm", data={"token": mailbox.link_token("confirm")})
    code = invite_code(client)
    join(client, code, "ad.a+1@gmail.com", "Ada Again", mailbox)  # the inviter's own mailbox
    join(client, code, "kofi@example.com", "Kofi", mailbox)
    join(client, code, "kofi+2@example.com", "Kofi Again", mailbox)  # Kofi's again
    as_user(client, "ada@gmail.com", "Ada")
    assert client.get("/api/referrals").json()["counted"] == 1


def test_without_email_invites_dont_count(client, paywall):
    signup(client, email="ada@example.com", name="Ada")
    code = invite_code(client)
    join(client, code, "kofi@example.com", "Kofi")
    as_user(client, "ada@example.com", "Ada")
    r = client.get("/api/referrals").json()
    assert r["counted"] == 0 and r["invites"][0]["status"] == "unconfirmed"
