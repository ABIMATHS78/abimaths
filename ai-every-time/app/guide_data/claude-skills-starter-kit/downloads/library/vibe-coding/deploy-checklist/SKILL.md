---
name: deploy-checklist
description: Runs a launch checklist for a web app, covering secrets, settings, backups, security, privacy, monitoring and a rollback plan. Use before putting an app live or shipping a big change.
---

# Deploy checklist

Go live without the classic mistakes, and know exactly what to do if something goes wrong.

## Before you start

Ask for: what the app is and who uses it; where it's hosted; the database, and where its backups go; whether it handles accounts, payments or personal data; how it's deployed now; and who's watching after launch. Ask in one short message.

## Steps

1. **Secrets**: no keys or passwords in the code or its history; each one set in the host's secret settings; separate keys for testing and for the live app; and anything that was ever exposed is replaced.
2. **Settings**: debug mode off, detailed errors hidden from users, and the right addresses, time zone and email sender.
3. **Data**: a recent backup exists, and restoring it has been tested; database changes were tried on a copy first.
4. **Security**: HTTPS everywhere; passwords stored with a proper hashing method; checks on who can see and change what; limits on login attempts; and up-to-date dependencies.
5. **Privacy**: a privacy notice that matches what the app collects; only the personal data the app needs; and a way for users to delete their data where the law requires it.
6. **Monitoring**: error tracking, uptime checks and logs, without personal data or secrets in the logs.
7. **Test it like a user**: sign up, log in, the main paths, payments in test mode, and on a phone.
8. **Plan the rollback**: how to return to the previous version and data, how long it takes, and who decides.
9. **Plan the launch**: a time when people are around to watch it, and who to tell if something breaks.

## Output

```
# Launch checklist: <app>, <date>
| Area | Check | Status | Owner | Note |
|---|---|---|---|---|

Blockers: <must be fixed before launch>
Rollback: <steps, time needed, who decides>
After launch: <what to watch, and for how long>
```

## Rules

- **Never paste real secrets into the conversation.** Check that they're set, not what they are.
- Don't mark a check done without evidence, such as a backup that was restored successfully.
- A known security or data-loss problem is a blocker, not a note.
- If the app handles payments, health information or children's data, recommend a review by an experienced person before launch.

## Checklist before you finish

- [ ] Secrets are out of the code and set in the host.
- [ ] A backup has been restored successfully.
- [ ] There's a rollback plan with an owner.
- [ ] Blockers are clearly separated from nice-to-haves.
