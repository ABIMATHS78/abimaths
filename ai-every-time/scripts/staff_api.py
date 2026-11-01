"""Work the Staff pages from a terminal, or for a tool such as Claude Code, with a staff token (app/tokens.py).

An admin makes the token on the Staff page's Setup tab. Keep it in the environment as AIET_STAFF_TOKEN, never in a
file in this repository or in a chat. AIET_SITE is the site's address (https://www.aieverytime.org unless it's set).
The token is sent only to that address, and never printed.

    python scripts/staff_api.py whoami
    python scripts/staff_api.py find ada@example.com
    python scripts/staff_api.py give-plan ada@example.com insider --until 2026-12-01 --note "Complimentary"
    python scripts/staff_api.py take-plan ada@example.com
    python scripts/staff_api.py log 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

SITE = os.environ.get("AIET_SITE", "https://www.aieverytime.org").rstrip("/")
TIERS = ("digital", "plus", "pro", "insider", "teams")


def call(method: str, path: str, body: dict | None = None):
    token = os.environ.get("AIET_STAFF_TOKEN", "").strip()
    if not token:
        sys.exit("AIET_STAFF_TOKEN isn't set: make a token on the Staff page's Setup tab and put it in the environment.")
    request = urllib.request.Request(
        f"{SITE}{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "aiet-staff-api"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail")
        except ValueError:
            detail = None
        sys.exit(f"{method} {path}: {e.code} {detail or e.reason}")
    return json.loads(raw) if raw else None


def last_day(until: str) -> str:
    """The last day of a given plan: the site keeps the moment it ends, the midnight after that day."""
    return (datetime.fromisoformat(until) - timedelta(seconds=1)).date().isoformat()


def plan_text(user: dict) -> str:
    plan, grant = user["plan"], user["plan"].get("grant")
    given = f" (given: {grant['name']}, {'last day ' + last_day(grant['until']) if grant.get('until') else 'for good'})" if grant else ""
    return f"{plan['name']}{given}"


def find(query: str) -> list[dict]:
    return call("GET", f"/api/staff/users?{urllib.parse.urlencode({'q': query})}")


def one(email: str) -> dict:
    """The account with exactly this email address."""
    matches = [u for u in find(email) if u["email"].lower() == email.strip().lower()]
    if len(matches) != 1:
        sys.exit(f"No account has the email address {email}: they need to sign up first.")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("whoami", help="the admin the token is for")
    p = sub.add_parser("find", help="people whose email or name matches")
    p.add_argument("query")
    p = sub.add_parser("give-plan", help="give someone a plan, until a day (the last day it lasts) or for good")
    p.add_argument("email")
    p.add_argument("tier", choices=TIERS)
    p.add_argument("--until", help="the last day, as YYYY-MM-DD; leave it out for good")
    p.add_argument("--note", default="", help="shown to them, and in the decision log")
    p = sub.add_parser("take-plan", help="take back a plan someone was given")
    p.add_argument("email")
    p = sub.add_parser("log", help="the latest entries in the decision log")
    p.add_argument("count", nargs="?", type=int, default=20)
    args = parser.parse_args()

    if args.command == "whoami":
        me = call("GET", "/api/me")
        print(f"{me['display_name']} <{me['email']}>, {me['role']}, at {SITE}")
    elif args.command == "find":
        for u in find(args.query):
            print(f"#{u['id']}  {u['email']}  {u['display_name']}  {u['role']}  {plan_text(u)}")
    elif args.command in ("give-plan", "take-plan"):
        user = one(args.email)
        body = ({"tier": args.tier, "until": args.until, "note": args.note} if args.command == "give-plan"
                else {"tier": None, "until": None, "note": ""})
        call("POST", f"/api/staff/users/{user['id']}/plan", body)
        print(f"{user['email']} now has: {plan_text(one(args.email))}")
    elif args.command == "log":
        for e in call("GET", "/api/staff/log")[:args.count]:
            print(f"{e['created_at'][:16]}  {e['staff_name'] or '–'}  {e['action']}  {e['target_type']} #{e['target_id']}  {e['note']}")


if __name__ == "__main__":
    main()
