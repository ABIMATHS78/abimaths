"""Bring the old site's accounts and subscriptions over, keeping everyone's plan and how they sign in.

Run it on the server, with the two exports from the old site's database (they hold people's details, so they never go
in the repository, in a chat or anywhere else but the server):

    python scripts/import_legacy.py --users users.csv --subscriptions subscriptions.csv --dry-run
    python scripts/import_legacy.py --users users.csv --subscriptions subscriptions.csv --report import-report.json

--dry-run does everything and then undoes it, to see the report first. What it prints, and the report it writes,
hold counts and the old site's ids only: never an email address, a name, a password hash or a whole Stripe id.

For each account it keeps the old site's id, when it was made and last signed in, how it signed in, and its role, and
its password: the old site's bcrypt hashes work here as they are (and are stored this site's way at the next
sign-in). Accounts that signed in with Google, Apple or Microsoft have no password; they choose one with a reset
link (the address is taken as confirmed, as those services confirmed it). Sessions and reset links aren't brought
over, so everyone signs in again. Nobody is signed up for news and offers: that needs a yes here.

Each account keeps its plan, from its subscription: the plan, monthly or yearly, its status, its Stripe customer and
subscription, the period it's paid up to, and a plan chosen at checkout but not paid for. Paid plans carry on as they
are until Stripe says otherwise: run `python -m app.billing reconcile` once Stripe is set up, to bring each up to date
from Stripe. Accounts with no subscription start on the free plan. An account with two subscription rows keeps the
better one, and the report lists the other.

Where several accounts share an email address (the old site told case apart: Ada@ and ada@), the one with the best
plan becomes the account here (then a password, then the latest sign-in); the others are kept in legacy_accounts,
linked to it, and listed in the report. They are never merged silently. An address already used by an account made
here is left alone and reported, unless --link-existing links the two (keeping the password chosen here).

Running it again adds accounts that weren't there before and leaves the ones already brought over as they are;
--refresh-subscriptions updates their plans from a newer export instead (for the final export, before switching over).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.legacy import BadExport, run, summary  # noqa: E402,F401  (run: for tests and other scripts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--users", required=True, help="the old site's users export (CSV)")
    parser.add_argument("--subscriptions", required=True, help="the old site's subscriptions export (CSV)")
    parser.add_argument("--dry-run", action="store_true", help="do everything, then undo it, and report")
    parser.add_argument("--link-existing", action="store_true", help="link old accounts to accounts made here with the same address")
    parser.add_argument("--refresh-subscriptions", action="store_true", help="update plans of accounts already brought over")
    parser.add_argument("--report", help="write the full report (counts and the old site's ids only) to this JSON file")
    args = parser.parse_args(argv)
    try:
        result = run(args.users, args.subscriptions, args.dry_run, args.link_existing, args.refresh_subscriptions)
    except BadExport as e:
        print(e, file=sys.stderr)
        return 2
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary(result), indent=2))
    if args.dry_run:
        print("Dry run: nothing was saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
