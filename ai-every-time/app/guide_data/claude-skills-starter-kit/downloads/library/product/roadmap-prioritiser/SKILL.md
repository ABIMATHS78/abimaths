---
name: roadmap-prioritiser
description: Prioritises a product backlog with a scoring method such as RICE, shows every score and explains the trade-offs behind the order. Use when choosing what to build next or explaining roadmap choices.
---

# Roadmap prioritiser

Rank a backlog in a way the team can see, question and repeat. The score starts the decision; it doesn't make it.

## Before you start

You need the backlog items, each with a short description, the goal they serve and whatever estimates exist. If the goal or the time period is missing, ask in one short message. If there are no estimates, propose rough ones and label them as yours.

## Steps

1. **Agree the method.** RICE suits most backlogs. Use value against effort when estimates are very rough, and must, should, could, won't (MoSCoW) for a release with fixed scope.
2. **Define the RICE scales before scoring anything:**
   - **Reach**: people or events affected in a set period, such as customers per quarter, from data where possible.
   - **Impact**: the effect on the goal for each person reached. 3 is massive, 2 high, 1 medium, 0.5 low, 0.25 minimal.
   - **Confidence**: how well the estimates are supported. 100% is high, 80% medium, 50% low.
   - **Effort**: total team time, in person-months or person-weeks, the same unit for every item.
   - **Score** = Reach × Impact × Confidence ÷ Effort.
3. **Score each item.** Use the same scales for every item, with a short reason and source for each estimate.
4. **Rank, then test the ranking.** Sort by score, then ask whether a reasonable change to one estimate would change the order. Items that swap easily are close calls; say so.
5. **Apply judgement.** Adjust for what scores miss: dependencies, deadlines, commitments, risk, technical debt and strategic bets. Record every move away from the score order, with the reason.
6. **Explain the trade-offs.** For the top items, say what the team gains. For items below the line, say what it gives up, and when they might come back.

## Output

```
| Rank | Item | Reach | Impact | Confidence | Effort | Score | Notes |
|---|---|---|---|---|---|---|---|
| 1 | <item> | <n per period> | <0.25 to 3> | <%> | <person-months> | <score> | <reason, source> |

**Moved by judgement**: <item, from which rank to which, and why>
**Close calls**: <items whose order depends on uncertain estimates>
**Trade-offs**: <what we gain, what we give up>
**Estimates to confirm**: <list, with who can confirm each>
```

## Rules

- Never present a guess as data. Label each estimate's source: data, expert view or assumption, and mark assumptions "to confirm".
- Don't let precise-looking scores hide rough inputs. Round the scores, and keep confidence honest.
- Score against one goal at a time. Mixing goals makes the scores meaningless.
- Treat legal, security and privacy obligations as constraints, not as items to score. If the laws that apply require something, schedule it and say why.
- Keep customer names and personal data out of backlog summaries.
- Keep everything visible: the method, the scales and every change made by judgement.

## Checklist before you finish

- [ ] Every item is scored on the same scales, with sources.
- [ ] Close calls and changes made by judgement are explained.
- [ ] The trade-offs are clear to someone who wasn't in the room.
