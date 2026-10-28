---
name: appointment-reminders
description: Writes appointment reminder messages for text, email, letter or voicemail that share the least personal information needed. Use when a clinic needs reminder templates or wording.
---

# Appointment reminders

Write reminders that help patients attend, or rebook in good time, without telling anyone else more than they need to know.

## Ask first

If any of this is missing, ask for it in one short message:
- the channels: text message, email, letter, voicemail or app notification;
- when reminders go out, such as a week before and the day before;
- anything the patient must do, such as arrive early, bring something or fill in a form;
- how to confirm, cancel or rebook, and the contact details;
- the sender name patients will see, and the clinic's rules on what reminders may contain.

## Steps

1. **Start with the minimum.** Most reminders need only a greeting, the fact of the appointment, the date and time, the place, and how to change it.
2. **Leave out anything that reveals health information**: the condition, the test, the department or specialty, and anything about results. If the clinic's name itself reveals the service, suggest a neutral sender name for the clinic to approve.
3. **Assume someone else may see it.** Texts show on lock screens, letters get opened by others and voicemails are played aloud. A stranger should learn almost nothing from any message.
4. **Use the least identity data**: a first name or no name, and never a date of birth, record number or address.
5. **Make the action easy**: one clear way to confirm or rebook. Mention a missed-appointment fee only if the clinic asks you to.
6. **Write a version for each channel.** Keep texts to 160 plain characters where possible, so they arrive as one message. Keep voicemails free of anything a family member shouldn't hear.
7. **Use preparation instructions only in the clinic's approved wording**, such as fasting before a test, and only in the channels the clinic allows for them.

## Output

```
| Channel | When sent | Message | Characters |
|---|---|---|---|
| Text | <timing> | Hi <first name>, a reminder of your appointment on <date> at <time> at <place>. To change it, call <number>. | <n> |
```

Then a short note on what each message leaves out on purpose, and a list of every detail marked "to confirm".

## Rules

- Never include diagnoses, test names, results, medicines or specialties unless the clinic confirms the patient has agreed to receive them that way.
- Send reminders only through channels the patient has agreed to, with an opt-out where the rules require one.
- Follow the data protection and health-privacy laws that apply where the clinic works, and the clinic's own policy. Its privacy lead approves the templates before use.
- Don't write clinical instructions yourself, and never add health advice. Use only wording a clinician has approved.
- Never invent phone numbers, addresses, fees or times. Mark them "to confirm".

## Checklist before you finish

- [ ] Could a stranger who saw this message learn anything about the patient's health? If so, cut it.
- [ ] Every message says how to change the appointment.
- [ ] Every text fits in one message where possible.
