"""News and offers by email, only with a yes; the reminder after an unfinished checkout; the weekly series for the
free plan; admins' emails; and the admin Analytics page."""

from datetime import datetime, timedelta, timezone

from tests.test_app import signup
from tests.test_billing import paywall, run, sql, stripe_fake  # noqa: F401  (fixtures)
from tests.test_community import as_user
from tests.test_email import mailbox  # noqa: F401  (fixture)
from tests.test_staff import admin_email  # noqa: F401  (fixture)


def news_reader(client, mailbox, email="ada@example.com", name="Ada", yes=True):
    client.post("/api/auth/logout")
    r = client.post("/api/auth/signup", json={"email": email, "password": "correct horse", "display_name": name,
                                              "marketing_emails": yes})
    assert r.status_code == 201
    client.post("/api/email/confirm", data={"token": mailbox.link_token("confirm")})
    return r.json()["id"]


def later(days=0, hours=0):
    return datetime.now(timezone.utc) + timedelta(days=days, hours=hours)


def test_the_weekly_series_goes_only_to_free_learners_who_said_yes(client, stripe_fake, mailbox):
    from app import campaigns

    yes = news_reader(client, mailbox, "ada@example.com", "Ada")
    news_reader(client, mailbox, "kofi@example.com", "Kofi", yes=False)
    before = len(mailbox.sent)
    assert campaigns.nudges(later(days=1)) == 0  # not in their first few days
    assert campaigns.nudges(later(days=4)) == 1
    msg = mailbox.sent[-1]
    assert msg["To"] == "ada@example.com" and msg["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    text = msg.get_body(("plain",)).get_content()
    assert "utm_source=email" in text and "utm_campaign=nudge-1" in text and "list=news" in msg["List-Unsubscribe"]
    assert campaigns.nudges(later(days=5)) == 0  # a week apart
    run("UPDATE email_log SET sent_at = '2000-01-01T00:00:00+00:00'")
    assert campaigns.nudges(later(days=12)) == 1
    # Paying stops it.
    run("UPDATE subscriptions SET tier = 'plus', status = 'active' WHERE user_id = ?", yes)
    run("UPDATE email_log SET sent_at = '2000-01-01T00:00:00+00:00'")
    assert campaigns.nudges(later(days=20)) == 0
    assert len(mailbox.sent) == before + 2


def test_one_click_stops_news_but_not_reminders(client, stripe_fake, mailbox):
    from app import campaigns

    news_reader(client, mailbox)
    client.patch("/api/me", json={"email_reminders": True})
    campaigns.nudges(later(days=4))
    token = mailbox.sent[-1]["List-Unsubscribe"].split("token=")[1].split("&")[0]
    assert "news and offers" in client.get(f"/api/email/unsubscribe?token={token}&list=news").text
    assert client.post(f"/api/email/unsubscribe?token={token}&list=news").status_code == 200
    me = client.get("/api/me").json()
    assert me["marketing_emails"] is False and me["email_reminders"] is True
    run("UPDATE email_log SET sent_at = '2000-01-01T00:00:00+00:00'")
    assert campaigns.nudges(later(days=12)) == 0


def test_a_reminder_a_day_after_an_unfinished_checkout(client, stripe_fake, mailbox):
    from app import campaigns

    news_reader(client, mailbox)
    client.post("/api/billing/checkout", json={"tier": "pro"})
    assert campaigns.follow_ups(later(hours=2)) == 0
    assert campaigns.follow_ups(later(hours=30)) == 1
    msg = mailbox.sent[-1]
    assert msg["Subject"] == "Still thinking about Pro?" and "7 days free" in msg.get_body(("plain",)).get_content()
    assert campaigns.follow_ups(later(hours=31)) == 0  # once per choice
    assert campaigns.follow_ups(later(hours=60)) == 0  # and not once it's old


def test_no_selling_while_payments_are_off(client, paywall, mailbox):
    from app import campaigns

    news_reader(client, mailbox)
    assert campaigns.nudges(later(days=4)) == 0 and campaigns.follow_ups(later(hours=30)) == 0


def test_admins_email_learners_who_said_yes_in_batches(client, paywall, mailbox, admin_email, monkeypatch):
    from app import campaigns

    for i in range(3):
        news_reader(client, mailbox, f"learner{i}@example.com", f"Learner {i}")
    news_reader(client, mailbox, "quiet@example.com", "Quiet", yes=False)
    news_reader(client, mailbox, admin_email, "Boss", yes=False)  # admins named in the config confirm their address first
    start = len(mailbox.sent)
    test = client.post("/api/staff/campaigns", json={"subject": "New courses", "body": "Hello!\n\nThere are new courses.", "test": True})
    assert test.status_code == 201 and mailbox.sent[-1]["To"] == admin_email and mailbox.sent[-1]["Subject"] == "[Test] New courses"
    made = client.post("/api/staff/campaigns", json={"subject": "New courses", "body": "Hello!\n\nThere are new courses."}).json()
    assert made["audience"] == 3
    monkeypatch.setattr(campaigns, "BATCH", 2)
    assert campaigns.campaign_batch() == 2 and campaigns.campaign_batch() == 1 and campaigns.campaign_batch() == 0
    sent = mailbox.sent[start + 1:]
    assert sorted(m["To"] for m in sent) == [f"learner{i}@example.com" for i in range(3)]
    assert "There are new courses." in sent[0].get_body(("plain",)).get_content()
    listed = client.get("/api/staff/campaigns").json()
    assert listed["campaigns"][0]["sent"] == 3 and listed["campaigns"][0]["finished_at"]
    as_user(client, "learner0@example.com", "Learner 0")
    assert client.post("/api/staff/campaigns", json={"subject": "Hi all", "body": "Hello everyone."}).status_code == 403


def test_analytics_for_admins_without_what_anyone_wrote(client, paywall, admin_email):
    from tests.test_usage import chat, starter

    signup(client, email="kofi@example.com", name="Kofi")
    starter_id = next(c["id"] for c in client.get("/api/companions").json() if c["is_starter"])
    client.post(f"/api/companions/{starter_id}/primary")
    chat(client, starter(client), "my secret question about pancakes")
    run("UPDATE subscriptions SET tier = 'pro', status = 'active', stripe_subscription_id = 'sub_a', amount_cents = 6999, "
        "billing_interval = 'yearly'")
    assert client.get("/api/staff/analytics").status_code == 403
    as_user(client, admin_email, "Boss")
    r = client.get("/api/staff/analytics?days=7")
    assert r.status_code == 200 and "pancakes" not in r.text and "sub_a" not in r.text
    a = r.json()
    assert a["overview"]["users"] == 2 and a["overview"]["paying"] == 1 and a["overview"]["mrr"] == round(6999 / 12)
    assert {t["tier"]: t["users"] for t in a["subscriptions"]["tiers"]}["pro"] == 1
    assert a["conversations"]["messages_in_period"] == 1 and len(a["growth"]) == 7
    assert client.get("/api/staff/analytics?days=3").status_code == 422
