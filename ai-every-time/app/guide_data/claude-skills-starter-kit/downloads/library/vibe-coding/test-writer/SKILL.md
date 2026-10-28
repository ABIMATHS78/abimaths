---
name: test-writer
description: Writes automated tests in the project's own framework, covering normal cases, edge cases and failures, and checks they would catch a real bug. Use when adding tests for new code or code that has none.
---

# Test writer

Write tests that catch real bugs, run fast, and are easy to understand when they fail.

## Before you start

You need the code to test, and what it's supposed to do. Ask for the language, the test framework and some existing tests, so the new ones match. Ask in one short message.

## Steps

1. **List the behaviours** the code promises, from its name, its documentation and the code that calls it, rather than from how it's written.
2. **Choose the cases:**
   - normal inputs, and the results they should give;
   - edge cases: empty, zero, one item, very large, missing values, duplicates, unusual characters, time zones and dates;
   - failures: invalid input, errors from other services, and missing permissions.
3. **Write one test per behaviour**, named so a failure explains itself, such as "rejects an expired discount code".
4. **Arrange, act, assert**: set up the inputs, run the code, and check the result. Keep each test independent of the others.
5. **Replace slow or outside things only where needed**: networks, clocks, randomness and payment services. Test the real code wherever that's practical.
6. **Check that the tests can fail.** Break the code on purpose, or reason it through, and confirm a test catches it. A test that can't fail is worthless.
7. **Run the tests**, and report the results honestly.

## Output

```
Behaviours covered: <list>

<test code, in the project's framework>

How to run: <command>
Not covered: <what still needs a test, and why>
```

## Rules

- **Never change the code under test to make a test pass** without saying so. If a test exposes a bug, report it.
- Never weaken or delete an existing test to get a passing result.
- Use made-up test data. Never put real customers' personal data or real secrets in tests.
- Don't say tests pass unless they were run.

## Checklist before you finish

- [ ] Normal, edge and failure cases are covered.
- [ ] The test names describe the behaviour.
- [ ] Each test would fail if the code were wrong.
- [ ] The command to run them, and the results, are given.
