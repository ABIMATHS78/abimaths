"""Plans, the paywall, and paying with Stripe: checkout, webhooks, plan changes, cancelling, the billing portal,
given plans, and old-site passwords. Stripe is a fake here that keeps what it was asked; webhooks are signed
with a test secret, as Stripe signs them."""

import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest
import stripe

from tests.test_app import signup
from tests.test_community import as_user
from tests.test_courses import SLUG
from tests.test_email import mailbox  # noqa: F401  (fixture)
from tests.test_learning import onboard
from tests.test_staff import admin_email  # noqa: F401  (fixture)

SECRET = "whsec_test_secret"


@pytest.fixture()
def paywall(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "paywall_mode", "1")


def subscription(sid, customer, user_id, tier, interval="monthly", status="active", trial_end=None, cancel=False, amount=None):
    """A Stripe subscription as Stripe's API returns it (billing periods on the item, as since 2025)."""
    from app import plans

    return {"id": sid, "object": "subscription", "customer": customer, "status": status, "trial_end": trial_end,
            "cancel_at_period_end": cancel, "cancel_at": None, "canceled_at": None,
            "metadata": {"aiet_user_id": str(user_id), "aiet_tier": tier, "aiet_interval": interval},
            "items": {"data": [{"id": f"si_{sid}", "quantity": 1, "current_period_start": 1790000000,
                                "current_period_end": 1792592000, "price": {
                                    "id": f"price_{tier}_{interval}", "lookup_key": f"aiet_{tier}_{interval}",
                                    "unit_amount": plans.plan(tier).price(interval) if amount is None else amount,
                                    "recurring": {"interval": "year" if interval == "yearly" else "month"}}}]}}


class FakeStripe:
    """The parts of stripe.StripeClient the app uses, keeping what they were asked."""

    def __init__(self):
        self.calls, self.subs, self.sessions, self.fail = [], {}, {}, set()

        def call(name, fn):
            def run(*args, **kwargs):
                self.calls.append((name, args, kwargs))
                if name in self.fail:
                    raise stripe.APIConnectionError("Stripe is down (test)")
                return fn(*args, **kwargs)
            return run

        ns = SimpleNamespace
        self.v1 = ns(
            customers=ns(create=call("customers.create", lambda params, options=None: {"id": "cus_new"}),
                         balance_transactions=ns(create=call("balance.create", lambda cid, params, options=None: {"id": "cbtxn_1"}))),
            checkout=ns(sessions=ns(create=call("sessions.create", lambda params: {"id": "cs_test_1", "url": "https://checkout.stripe.test/cs_test_1"}),
                                    retrieve=call("sessions.retrieve", lambda sid, params=None: self.sessions[sid]))),
            subscriptions=ns(retrieve=call("subscriptions.retrieve", self._retrieve),
                             update=call("subscriptions.update", self._update),
                             cancel=call("subscriptions.cancel", self._cancel)),
            billing_portal=ns(sessions=ns(create=call("portal.create", lambda params: {"url": "https://billing.stripe.test/p_1"}))),
            prices=ns(list=call("prices.list", self._prices_list), retrieve=call("prices.retrieve", self._price),
                      create=call("prices.create", self._price_create)),
            products=ns(create=call("products.create", lambda params: {"id": f"prod_{params['metadata']['aiet_tier']}"})),
        )
        from app import plans

        # The prices in this Stripe account, by lookup key: to begin with, one for every plan and period.
        self.prices = {f"aiet_{p.slug}_{i}": {"id": f"price_aiet_{p.slug}_{i}", "lookup_key": f"aiet_{p.slug}_{i}",
                                              "unit_amount": p.price(i)} for p in plans.PLANS.values() if p.paid
                       for i in plans.INTERVALS}

    def _prices_list(self, params):
        return {"data": [self.prices[k] for k in params["lookup_keys"] if k in self.prices][:params.get("limit", 10)]}

    def _price(self, pid, params=None):
        return next(p for p in self.prices.values() if p["id"] == pid)

    def _price_create(self, params):
        self.prices[params["lookup_key"]] = {"id": f"price_{params['lookup_key']}", "lookup_key": params["lookup_key"],
                                             "unit_amount": params["unit_amount"]}
        return self.prices[params["lookup_key"]]

    def _retrieve(self, sid, params=None):
        if sid not in self.subs:
            raise stripe.InvalidRequestError("No such subscription", "id")
        return self.subs[sid]

    def _update(self, sid, params):
        sub = self.subs[sid]
        if "items" in params:
            tier, interval = params["items"][0]["price"].removeprefix("price_aiet_").rsplit("_", 1)
            sub["items"]["data"][0]["price"] = subscription(sid, "", 0, tier, interval)["items"]["data"][0]["price"]
        if "cancel_at_period_end" in params:
            sub["cancel_at_period_end"] = params["cancel_at_period_end"]
        return sub

    def _cancel(self, sid, params=None):
        self.subs[sid]["status"] = "canceled"
        return self.subs[sid]

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture()
def stripe_fake(monkeypatch, paywall):
    from app import billing
    from app.config import settings

    fake = FakeStripe()
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_123")
    monkeypatch.setattr(settings, "stripe_webhook_secret", SECRET)
    monkeypatch.setattr(settings, "public_url", "https://aieverytime.example")
    monkeypatch.setattr(billing, "_client", lambda: fake)
    monkeypatch.setattr(billing, "_price_ids", {})
    return fake


def send_event(client, kind, obj, event_id=None, secret=SECRET):
    payload = json.dumps({"id": event_id or f"evt_{time.time_ns()}", "object": "event", "type": kind, "created": int(time.time()),
                          "data": {"object": obj}})
    ts = int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.{payload}".encode(), hashlib.sha256).hexdigest()
    return client.post("/api/billing/webhook", content=payload,
                       headers={"stripe-signature": f"t={ts},v1={sig}", "content-type": "application/json"})


def plan(client) -> dict:
    return client.get("/api/billing").json()


def me_id(client) -> int:
    return client.get("/api/me").json()["id"]


def sql(query, *args):
    from app import db

    with db.connect() as conn:
        return [dict(r) for r in conn.execute(query, args)]


def run(query, *args):
    from app import db

    with db.connect() as conn:
        conn.execute(query, args)


def pays_for(client, tier, status="active"):
    """The signed-in learner's plan, as if paid for (sign-up starts everyone on the free plan)."""
    run("UPDATE subscriptions SET tier = ?, status = ? WHERE user_id = ?", tier, status, me_id(client))


# ------------------------------------------------------------------- the plans


def test_the_plans_are_public_and_match_the_pricing_page(client):
    d = client.get("/api/plans").json()
    assert [p["slug"] for p in d["plans"]] == ["free", "digital", "plus", "pro", "insider", "teams"]
    by = {p["slug"]: p for p in d["plans"]}
    assert [by[s]["monthly"] for s in ("digital", "plus", "pro", "insider", "teams")] == [199, 299, 699, 999, 899]
    assert by["plus"]["badge"] == "Most popular" and by["insider"]["badge"] == "Premium"
    assert by["teams"]["per_seat"] and not by["teams"]["self_serve"] and by["teams"]["trial_days"] == 0
    assert all(by[s]["trial_days"] == 7 and 0 < by[s]["yearly_saving"] < 30 for s in ("digital", "plus", "pro", "insider"))
    assert [by[s]["ai_allowance"] for s in by] == sorted(by[s]["ai_allowance"] for s in by)  # more as plans go up
    assert "All 103 courses in full: every module, lab and project" in by["plus"]["highlights"] or any(
        "courses in full" in h for h in by["plus"]["highlights"])
    assert d["teams_email"] and d["payments"] is False and d["paywall"] is False


def test_without_the_paywall_everything_is_open(client):
    onboard(client)
    from app import catalog

    me = client.get("/api/me").json()
    assert me["plan"]["tier"] == "free" and me["plan"]["paywall"] is False and "courses" in me["plan"]["unlocks"]
    assert client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 3, 1)}").status_code == 200
    assert client.post("/api/custom-paths", json={"topic": "Rust for data engineers"}).status_code == 201
    assert me["ai_allowance"]["limit"] == 100  # the site's allowance, as before plans


def test_the_free_plan_opens_the_first_module_of_every_course(client, paywall):
    onboard(client)
    from app import catalog

    client.post(f"/api/courses/{SLUG}/enroll")
    page = client.get(f"/api/courses/{SLUG}").json()
    assert page["access"] == {"full": False, "free_modules": 1, "plan": "Plus"}
    assert [m["locked"] for m in page["syllabus"]][:2] == [False, True]
    assert client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 1, 1)}").status_code == 200
    assert client.get(f"/api/courses/{SLUG}/labs/1").status_code == 200
    for r in (client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 2, 1)}"),
              client.post(f"/api/lessons/{catalog.lesson_id(SLUG, 2, 1)}/complete"),
              client.get(f"/api/courses/{SLUG}/labs/3"), client.get(f"/api/courses/{SLUG}/labs/3/files"),
              client.post(f"/api/courses/{SLUG}/labs/3/submit", json={"content": "my work"}),
              client.get(f"/api/courses/{SLUG}/project"),
              client.post(f"/api/courses/{SLUG}/project/submit", json={"content": "my project"}),
              client.post(f"/api/courses/{SLUG}/certificate", json={}),
              client.post("/api/assessments", json={"kind": "quiz", "source_type": "path", "source_ref": catalog.module_id(SLUG, 2)}),
              client.post("/api/decks", json={"source_type": "lesson", "source_ref": catalog.lesson_id(SLUG, 2, 1)})):
        assert r.status_code == 402, r.text
        assert r.headers["X-Plan-Needed"] == "plus"
    assert "Plus" in client.get(f"/api/courses/{SLUG}/project").json()["detail"]
    # Asking about a locked lesson in chat is locked too, so the instructor doesn't teach it there instead.
    teacher = page["instructors"][0]["id"]
    assert client.post(f"/api/companions/{teacher}/chat", json={"message": "Explain it",
                                                                "lesson_id": catalog.lesson_id(SLUG, 2, 1)}).status_code == 402
    # A module 1 quiz is fine; a mock exam needs Plus; a path designed for them needs Pro.
    quiz = client.post("/api/assessments", json={"kind": "quiz", "source_type": "path", "source_ref": catalog.module_id(SLUG, 1)})
    assert quiz.status_code == 201
    exam = client.post("/api/assessments", json={"kind": "exam", "source_type": "course", "source_ref": SLUG})
    assert exam.status_code == 402 and exam.headers["X-Plan-Needed"] == "plus"
    path = client.post("/api/custom-paths", json={"topic": "Rust for data engineers"})
    assert path.status_code == 402 and path.headers["X-Plan-Needed"] == "pro"
    assert client.get("/api/courses").json()["full_access"] is False


def test_a_plan_opens_what_it_includes(client, paywall):
    onboard(client)
    from app import catalog

    pays_for(client, "plus")
    assert client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 3, 1)}").status_code == 200
    assert client.post("/api/assessments", json={"kind": "exam", "source_type": "course", "source_ref": SLUG}).status_code == 201
    assert client.post("/api/custom-paths", json={"topic": "Rust"}).headers["X-Plan-Needed"] == "pro"
    me = client.get("/api/me").json()
    assert me["plan"]["tier"] == "plus" and me["ai_allowance"]["limit"] == 80
    # A plan whose payments Stripe gave up on is the free plan again.
    run("UPDATE subscriptions SET status = 'unpaid'")
    assert client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 3, 1)}").status_code == 402
    # One being retried after a failed payment carries on meanwhile.
    run("UPDATE subscriptions SET status = 'past_due'")
    assert client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 3, 1)}").status_code == 200


def test_admins_can_use_everything(client, paywall, admin_email):
    onboard(client, email=admin_email, name="Boss")
    assert client.post("/api/custom-paths", json={"topic": "Rust"}).status_code == 201
    assert client.get("/api/me").json()["ai_allowance"]["limit"] is None


def test_free_guides_each_month_and_the_start_of_the_rest(client, paywall, tmp_path, monkeypatch):
    from app.config import settings
    from tests.test_guides import guide, load, two_starters, write

    monkeypatch.setattr(settings, "free_guides", 2)
    a, _ = two_starters()
    body = "## Part one\n\n" + "A long paragraph about data. " * 40 + "\n\n## Part two\n\n" + "More text here. " * 80
    path = tmp_path / "guides.json.gz"
    write(path, [guide(i, a.slug, f"2026-09-0{i}", body=body) for i in range(1, 5)])
    load(monkeypatch, path)
    onboard(client)
    ids = sorted(g["id"] for g in client.get("/api/guides").json())
    first, second, third = (client.get(f"/api/posts/{i}").json() for i in ids[:3])
    assert first["body"] == body and first["guides"] == {"full": True, "left": 1, "limit": 2}
    assert second["guides"]["left"] == 0 and "locked" not in second
    assert third["locked"] is True and third["plan_needed"] == "digital" and len(third["body"]) < len(body) // 3
    assert client.get(f"/api/posts/{ids[0]}").json()["body"] == body  # one already read stays open all month
    r = client.post("/api/decks", json={"source_type": "post", "source_ref": str(ids[2])})
    assert r.status_code == 402 and r.headers["X-Plan-Needed"] == "digital"
    assert plan(client)["guides"] == {"limit": 2, "left": 0}
    # Digital and up read them all.
    pays_for(client, "digital")
    assert client.get(f"/api/posts/{ids[3]}").json()["body"] == body


def test_daily_ai_requests_follow_the_plan(client, paywall, monkeypatch):
    from app.config import settings
    from tests.test_usage import chat, starter

    monkeypatch.setattr(settings, "plan_allowances", {"free": 1})
    onboard(client)
    assert client.get("/api/me").json()["ai_allowance"]["limit"] == 1
    sid = starter(client)
    assert chat(client, sid).status_code == 200
    refused = chat(client, sid)
    assert refused.status_code == 429 and refused.headers["X-Plan-Needed"] == "digital"
    assert "Digital gives you 40 a day" in refused.json()["detail"]
    # An allowance an admin set for someone wins over their plan's.
    run("UPDATE users SET ai_allowance = 7")
    assert client.get("/api/me").json()["ai_allowance"]["limit"] == 7


def test_confirming_the_address_matters_only_on_the_free_plan(client, paywall, mailbox, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ai_allowance_unconfirmed", 5)
    onboard(client)
    me = client.get("/api/me").json()["ai_allowance"]
    assert me["limit"] == 5 and me["confirm_raises_to"] == 15
    pays_for(client, "pro")
    me = client.get("/api/me").json()["ai_allowance"]
    assert me["limit"] == 150 and me["confirm_raises_to"] is None


# --------------------------------------------------------------------- checkout


def test_checkout_needs_payments_set_up(client, paywall):
    signup(client)
    assert client.post("/api/billing/checkout", json={"tier": "plus"}).status_code == 503
    assert plan(client)["intended"] is None  # nothing to finish when nothing could be started


def test_checkout_starts_with_a_trial_and_remembers_the_plan_chosen(client, stripe_fake):
    signup(client)
    uid = me_id(client)
    r = client.post("/api/billing/checkout", json={"tier": "plus", "interval": "yearly"})
    assert r.status_code == 200 and r.json()["url"] == "https://checkout.stripe.test/cs_test_1"
    (_, _, made), = stripe_fake.named("customers.create")
    assert made["params"]["email"] == "ada@example.com" and made["params"]["metadata"] == {"aiet_user_id": str(uid)}
    assert made["options"]["idempotency_key"].startswith(f"aiet-customer-{uid}-")
    (_, _, s), = stripe_fake.named("sessions.create")
    p = s["params"]
    assert p["mode"] == "subscription" and p["customer"] == "cus_new" and p["client_reference_id"] == str(uid)
    assert p["line_items"] == [{"price": "price_aiet_plus_yearly", "quantity": 1}] and p["allow_promotion_codes"] is True
    assert p["subscription_data"]["trial_period_days"] == 7 and p["subscription_data"]["metadata"]["aiet_tier"] == "plus"
    assert p["success_url"] == "https://aieverytime.example/#/billing?checkout={CHECKOUT_SESSION_ID}"
    assert p["cancel_url"].startswith("https://aieverytime.example/#/pricing")
    assert plan(client)["intended"] == {"tier": "plus", "name": "Plus", "interval": "yearly",
                                        "at": sql("SELECT intended_at FROM subscriptions")[0]["intended_at"]}
    # Again: the same customer, and no trial for someone who has had one.
    run("UPDATE subscriptions SET trial_used = 1")
    client.post("/api/billing/checkout", json={"tier": "pro"})
    assert len(stripe_fake.named("customers.create")) == 1
    assert "trial_period_days" not in stripe_fake.named("sessions.create")[-1][2]["params"]["subscription_data"]
    assert client.post("/api/billing/checkout", json={"tier": "teams"}).status_code == 422  # Teams is set up by us


def test_the_webhook_needs_stripes_signature(client, stripe_fake, monkeypatch):
    signup(client)
    assert send_event(client, "invoice.paid", {"id": "in_1"}, secret="whsec_wrong").status_code == 400
    r = client.post("/api/billing/webhook", content="{}", headers={"stripe-signature": "t=1,v1=00"})
    assert r.status_code == 400
    from app.config import settings

    monkeypatch.setattr(settings, "stripe_webhook_secret", "")
    assert send_event(client, "invoice.paid", {"id": "in_1"}).status_code == 503


def test_a_finished_checkout_starts_the_plan_once(client, stripe_fake, mailbox):
    signup(client)
    uid = me_id(client)
    client.post("/api/billing/checkout", json={"tier": "plus"})
    stripe_fake.subs["sub_1"] = subscription("sub_1", "cus_new", uid, "plus", status="trialing", trial_end=1791000000)
    done = {"id": "cs_test_1", "object": "checkout.session", "mode": "subscription", "status": "complete",
            "client_reference_id": str(uid), "customer": "cus_new", "subscription": "sub_1"}
    for _ in range(2):  # Stripe can send an event twice
        assert send_event(client, "checkout.session.completed", done, event_id="evt_same").status_code == 200
    p = plan(client)
    assert p["tier"] == "plus" and p["subscription"]["status"] == "trialing" and p["subscription"]["trial_end"]
    assert p["intended"] is None and p["trial_available"] is False and p["subscription"]["manageable"]
    assert [e["event_type"] for e in sql("SELECT * FROM subscription_events")] == ["new"]
    assert len(sql("SELECT * FROM stripe_events")) == 1
    welcome = [m for m in mailbox.sent if m["Subject"] == "Welcome to Plus"]
    assert len(welcome) == 1 and "free trial" in welcome[0].get_body(("plain",)).get_content()
    # The same event under a new id still sends one welcome: each email is noted.
    send_event(client, "checkout.session.completed", done)
    assert len([m for m in mailbox.sent if m["Subject"] == "Welcome to Plus"]) == 1


def test_events_arriving_out_of_order_use_stripes_current_state(client, stripe_fake):
    signup(client)
    uid = me_id(client)
    stripe_fake.subs["sub_1"] = subscription("sub_1", "cus_1", uid, "pro", status="canceled")
    stale = subscription("sub_1", "cus_1", uid, "pro", status="active")  # an update that happened before the cancel
    send_event(client, "customer.subscription.updated", stale)
    p = plan(client)
    assert p["tier"] == "free" and p["subscription"]["status"] == "canceled"


def test_a_subscription_matches_its_learner_by_customer_or_old_site_id(client, stripe_fake):
    signup(client)
    uid = me_id(client)
    run("UPDATE users SET legacy_id = '900001' WHERE id = ?", uid)
    sub = subscription("sub_9", "cus_9", 0, "insider")
    sub["metadata"] = {"userId": "900001"}  # as the old site wrote it
    stripe_fake.subs["sub_9"] = sub
    send_event(client, "customer.subscription.created", sub)
    assert plan(client)["tier"] == "insider"
    # Prices the app doesn't know are matched by amount.
    sub2 = subscription("sub_9", "cus_9", 0, "digital")
    sub2["items"]["data"][0]["price"] = {"id": "price_old_site", "unit_amount": 199, "recurring": {"interval": "month"}}
    stripe_fake.subs["sub_9"] = sub2
    send_event(client, "customer.subscription.updated", sub2)
    assert plan(client)["tier"] == "digital"
    assert [e["event_type"] for e in sql("SELECT * FROM subscription_events ORDER BY id")] == ["new", "downgrade"]


def test_payments_and_refunds_are_recorded_once(client, stripe_fake):
    signup(client)
    uid = me_id(client)
    stripe_fake.subs["sub_1"] = subscription("sub_1", "cus_1", uid, "pro")
    send_event(client, "customer.subscription.created", stripe_fake.subs["sub_1"])
    invoice = {"id": "in_1", "object": "invoice", "customer": "cus_1", "amount_paid": 699, "currency": "usd",
               "status_transitions": {"paid_at": 1790000000},
               "lines": {"data": [{"pricing": {"price_details": {"price": "price_aiet_pro_monthly"}}}]}}
    for _ in range(2):
        send_event(client, "invoice.paid", invoice)
    send_event(client, "invoice.paid", invoice | {"id": "in_trial", "amount_paid": 0})
    charge = {"id": "ch_1", "object": "charge", "customer": "cus_1", "currency": "usd", "amount_refunded": 200}
    send_event(client, "charge.refunded", charge)
    send_event(client, "charge.refunded", charge)
    send_event(client, "charge.refunded", charge | {"amount_refunded": 699})  # the rest, later
    rows = sql("SELECT kind, amount_cents, tier FROM revenue_events ORDER BY id")
    assert rows == [{"kind": "payment", "amount_cents": 699, "tier": "pro"}, {"kind": "refund", "amount_cents": -200, "tier": "pro"},
                    {"kind": "refund", "amount_cents": -499, "tier": "pro"}]


def test_a_failed_payment_is_emailed_and_the_plan_carries_on_while_retried(client, stripe_fake, mailbox):
    signup(client)
    uid = me_id(client)
    stripe_fake.subs["sub_1"] = subscription("sub_1", "cus_1", uid, "plus", status="past_due")
    send_event(client, "customer.subscription.updated", stripe_fake.subs["sub_1"])
    send_event(client, "invoice.payment_failed", {"id": "in_2", "customer": "cus_1"})
    assert plan(client)["tier"] == "plus"
    assert [m["Subject"] for m in mailbox.sent].count("Your payment didn't go through") == 1
    stripe_fake.subs["sub_1"]["status"] = "canceled"
    send_event(client, "customer.subscription.deleted", stripe_fake.subs["sub_1"])
    assert plan(client)["tier"] == "free"
    assert "Your Plus plan has ended" in [m["Subject"] for m in mailbox.sent]


def test_coming_back_from_checkout_starts_the_plan_without_waiting_for_the_webhook(client, stripe_fake):
    signup(client)
    uid = me_id(client)
    stripe_fake.subs["sub_1"] = subscription("sub_1", "cus_new", uid, "digital")
    stripe_fake.sessions["cs_test_9"] = {"id": "cs_test_9", "status": "complete", "client_reference_id": str(uid),
                                         "customer": "cus_new", "subscription": "sub_1"}
    stripe_fake.sessions["cs_test_other"] = {"id": "cs_test_other", "status": "complete", "client_reference_id": "999",
                                             "subscription": "sub_x"}
    assert client.post("/api/billing/checkout/confirm", json={"session_id": "cs_test_other"}).status_code == 404
    assert client.post("/api/billing/checkout/confirm", json={"session_id": "not-a-session"}).status_code == 422
    p = client.post("/api/billing/checkout/confirm", json={"session_id": "cs_test_9"}).json()
    assert p["tier"] == "digital" and p["subscription"]["live"]


def _paying(client, stripe_fake, tier="plus"):
    signup(client)
    uid = me_id(client)
    stripe_fake.subs["sub_1"] = subscription("sub_1", "cus_1", uid, tier)
    send_event(client, "customer.subscription.created", stripe_fake.subs["sub_1"])
    return uid


def test_moving_plans_up_is_charged_now_and_down_is_credited(client, stripe_fake):
    _paying(client, stripe_fake, "plus")
    assert client.post("/api/billing/checkout", json={"tier": "pro"}).status_code == 409  # no second subscription
    p = client.post("/api/billing/change", json={"tier": "pro", "interval": "monthly"}).json()
    assert p["tier"] == "pro"
    (_, args, kw), = stripe_fake.named("subscriptions.update")
    assert args[0] == "sub_1" and kw["params"]["items"] == [{"id": "si_sub_1", "price": "price_aiet_pro_monthly"}]
    assert kw["params"]["proration_behavior"] == "always_invoice" and kw["params"]["payment_behavior"] == "pending_if_incomplete"
    client.post("/api/billing/change", json={"tier": "digital", "interval": "monthly"})
    assert stripe_fake.named("subscriptions.update")[-1][2]["params"]["proration_behavior"] == "create_prorations"
    assert plan(client)["tier"] == "digital"
    assert [e["event_type"] for e in sql("SELECT * FROM subscription_events ORDER BY id")] == ["new", "upgrade", "downgrade"]


def test_cancelling_keeps_the_plan_to_the_end_of_the_period_and_can_be_undone(client, stripe_fake):
    _paying(client, stripe_fake)
    p = client.post("/api/billing/cancel").json()
    assert p["tier"] == "plus" and p["subscription"]["cancel_at_period_end"] is True
    p = client.post("/api/billing/resume").json()
    assert p["subscription"]["cancel_at_period_end"] is False
    assert client.post("/api/billing/portal").json()["url"].startswith("https://billing.stripe.test/")
    portal = stripe_fake.named("portal.create")[0][2]["params"]
    assert portal == {"customer": "cus_1", "return_url": "https://aieverytime.example/#/billing"}


def test_free_learners_have_nothing_to_change_or_cancel(client, stripe_fake):
    signup(client)
    for url in ("/api/billing/cancel", "/api/billing/resume"):
        assert client.post(url).status_code == 409
    assert client.post("/api/billing/change", json={"tier": "pro"}).status_code == 409
    assert client.post("/api/billing/portal").status_code == 409


def test_stripe_being_down_is_said_plainly(client, stripe_fake):
    signup(client)
    stripe_fake.fail.add("sessions.create")
    r = client.post("/api/billing/checkout", json={"tier": "plus"})
    assert r.status_code == 502 and "Try again" in r.json()["detail"]


def test_deleting_the_account_cancels_its_subscription(client, stripe_fake):
    _paying(client, stripe_fake)
    assert client.request("DELETE", "/api/me", json={"password": "correct horse"}).status_code == 204
    assert stripe_fake.subs["sub_1"]["status"] == "canceled"
    assert sql("SELECT * FROM stripe_cancellations") == []
    # The payment records stay, without the account.
    assert sql("SELECT user_id FROM subscription_events") == [{"user_id": None}]


def test_a_cancellation_stripe_couldnt_take_waits_and_is_retried(client, stripe_fake):
    from app import billing

    _paying(client, stripe_fake)
    stripe_fake.fail.add("subscriptions.cancel")
    assert client.request("DELETE", "/api/me", json={"password": "correct horse"}).status_code == 204
    assert [r["stripe_subscription_id"] for r in sql("SELECT * FROM stripe_cancellations")] == ["sub_1"]
    stripe_fake.fail.clear()
    assert billing.cancel_queued() == 1
    assert stripe_fake.subs["sub_1"]["status"] == "canceled" and sql("SELECT * FROM stripe_cancellations") == []


def test_reconcile_brings_imported_subscriptions_up_to_date(client, stripe_fake):
    from app import billing

    signup(client)
    uid = me_id(client)
    run("UPDATE subscriptions SET tier = 'insider', status = 'past_due', stripe_customer_id = 'cus_old', "
        "stripe_subscription_id = 'sub_old' WHERE user_id = ?", uid)
    stripe_fake.subs["sub_old"] = subscription("sub_old", "cus_old", 0, "insider", status="active")
    report = billing.reconcile(dry_run=True)
    assert report["changes"] == {"insider/past_due -> insider/active": 1} and plan(client)["subscription"]["status"] == "past_due"
    report = billing.reconcile()
    assert report["updated"] == 1 and plan(client)["subscription"]["status"] == "active"
    del stripe_fake.subs["sub_old"]
    assert billing.reconcile()["gone"] == 1  # not in this Stripe account: left for a person to look at


# ------------------------------------------------------------- plans given by admins


def test_admins_give_plans_until_a_day_or_for_good(client, paywall, admin_email):
    signup(client, email="kofi@example.com", name="Kofi")
    kofi = me_id(client)
    as_user(client, admin_email, "Boss")
    r = client.post(f"/api/staff/users/{kofi}/plan", json={"tier": "teams", "until": "2099-01-31", "note": "Acme team"})
    assert r.status_code == 200 and r.json()["tier"] == "teams" and r.json()["grant"]["until"].startswith("2099-02-01")
    assert client.post(f"/api/staff/users/{kofi}/plan", json={"tier": "pro", "until": "2001-01-01"}).status_code == 422
    people = client.get("/api/staff/users?q=kofi").json()
    assert people[0]["plan"]["tier"] == "teams" and people[0]["plan"]["grant"]["note"] == "Acme team"
    assert any(r["action"].startswith("plan:teams") for r in client.get("/api/staff/log").json())
    as_user(client, "kofi@example.com", "Kofi")
    assert client.post("/api/custom-paths", json={"topic": "Rust"}).status_code == 201
    assert any("Teams plan" in n["text"] for n in sql("SELECT text FROM notices"))
    run("UPDATE subscriptions SET grant_until = '2001-01-01T00:00:00+00:00'")  # it ran out
    assert plan(client)["tier"] == "free"
    as_user(client, admin_email, "Boss")
    assert client.post(f"/api/staff/users/{kofi}/plan", json={"tier": None}).json()["grant"] is None
    as_user(client, "kofi@example.com", "Kofi")
    assert client.post(f"/api/staff/users/{kofi}/plan", json={"tier": "pro"}).status_code == 403


# ------------------------------------------------------------------ old passwords


def test_old_site_passwords_work_and_are_rehashed(client):
    import bcrypt

    from app import db

    with db.connect() as conn:
        conn.execute("INSERT INTO users (email, display_name, password_hash, created_at, legacy_id) VALUES (?, ?, ?, ?, ?)",
                     ("old@example.com", "Old Member", bcrypt.hashpw(b"from the old site", bcrypt.gensalt(4)).decode(),
                      db.now(), "123"))
        conn.execute("INSERT INTO users (email, display_name, password_hash, created_at, legacy_id) VALUES (?, ?, '', ?, ?)",
                     ("google@example.com", "Google Member", db.now(), "124"))
    assert client.post("/api/auth/login", json={"email": "old@example.com", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "old@example.com", "password": "from the old site"}).status_code == 200
    stored = sql("SELECT password_hash, last_signed_in FROM users WHERE email = 'old@example.com'")[0]
    assert stored["password_hash"].startswith("scrypt$") and stored["last_signed_in"]
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"email": "old@example.com", "password": "from the old site"}).status_code == 200
    assert client.post("/api/auth/login", json={"email": "google@example.com", "password": ""}).status_code in (401, 422)
    assert client.post("/api/auth/login", json={"email": "google@example.com", "password": "anything"}).status_code == 401


def test_billing_in_the_data_export(client, stripe_fake):
    _paying(client, stripe_fake, "pro")
    data = json.loads(client.get("/api/me/export").content)
    assert data["plan"][0]["tier"] == "pro" and data["plan_changes"][0]["event_type"] == "new"
    assert "stripe_customer_id" not in data["plan"][0] and data["account"]["news_and_offers_emails"] is False


def test_insights_show_each_plans_allowance_while_the_paywall_is_on(client, paywall, admin_email):
    signup(client, email=admin_email, name="Boss")
    limits = client.get("/api/staff/insights").json()["ai"]["limits"]
    assert limits["plans"] == {"Free": 15, "Digital": 40, "Plus": 80, "Pro": 150, "Insider": 300, "Teams": 300}


def test_admins_set_up_stripe_from_the_setup_page(client, stripe_fake, admin_email):
    signup(client, email=admin_email, name="Boss")
    stripe_fake.prices.clear()  # a new Stripe account
    status = client.get("/api/staff/stripe").json()
    assert status["payments"] and status["webhook_secret"] and status["linked"] == 0
    assert status["webhook_url"] == "https://aieverytime.example/api/billing/webhook" and "invoice.paid" in status["events"]
    assert len(status["prices"]) == 10 and not any(r["ok"] for r in status["prices"])
    made = client.post("/api/staff/stripe/setup").json()
    assert len(made["made"]) == 10 and all(r["ok"] for r in made["prices"]) and len(stripe_fake.named("products.create")) == 5
    created = {c[2]["params"]["lookup_key"]: c[2]["params"] for c in stripe_fake.named("prices.create")}
    assert created["aiet_plus_yearly"]["unit_amount"] == 2999 and created["aiet_plus_yearly"]["recurring"] == {"interval": "year"}
    assert client.post("/api/staff/stripe/setup").json()["made"] == []  # again: nothing new
    stripe_fake.prices["aiet_pro_monthly"]["unit_amount"] = 799  # changed in Stripe's dashboard
    wrong = [r for r in client.get("/api/staff/stripe").json()["prices"] if not r["ok"]]
    assert [(r["tier"], r["interval"], r["amount"], r["expected"]) for r in wrong] == [("pro", "monthly", 799, 699)]
    assert client.post("/api/staff/stripe/reconcile", json={"dry_run": True}).json()["checked"] == 0
    client.post("/api/auth/logout")
    signup(client, email="kofi@example.com", name="Kofi")
    assert client.get("/api/staff/stripe").status_code == 403 and client.post("/api/staff/stripe/setup").status_code == 403
