---
name: plain-technical-english
description: Rewrites instructions and technical text in a Simplified Technical English style, then checks it with a script for long sentences, passive voice and vague words. Use for manuals and procedures.
---

# Plain technical English

Rewrite technical text so that anyone can follow it, including readers whose first language isn't English and readers under pressure. The style follows the main writing rules of Simplified Technical English (ASD-STE100). It doesn't certify that text meets that standard, which also needs the official STE dictionary.

## First, sort the text

- **Procedure**: steps someone carries out ("Remove the cover").
- **Description**: how something is or works ("The pump supplies fuel to the engine").

## Rules for both

- Use short, common words. Use one word for one thing, every time: if it's a "cover", never call it a "lid" later.
- Use the active voice: "The valve closes the line", not "The line is closed by the valve".
- Use only the simple tenses: present, past and future.
- Keep articles: "Remove **the** cover", not "Remove cover".
- Avoid -ing forms where you can: "before you start", not "before starting".
- Don't use idioms, slang or jokes.
- Write numbers and units in the same way throughout.

## Rules for procedures

- No more than **20 words** per sentence.
- **One instruction** per sentence. Two actions that happen at the same time can share one.
- Start with the verb, in the imperative: "Turn the knob clockwise."
- Put a condition first: "If the light comes on, stop the engine."
- Put warnings and cautions **before** the step they apply to, and say what can happen.

## Rules for descriptions

- No more than **25 words** per sentence.
- One topic per paragraph, and no more than **six sentences** per paragraph.
- Start each paragraph with its main point.

## Workflow

1. Sort the text into procedures and descriptions.
2. Rewrite it with the rules above. Keep every technical fact, value, part number and warning.
3. Save the rewrite to a file and run the checker:

   ```bash
   python scripts/check_ste.py rewrite.txt
   ```

   Number the steps of a procedure ("1. Remove the cover."), so the checker applies the 20-word limit to them.
4. Fix each finding, or keep it on purpose and say why. The checker uses simple patterns: it can flag something that is fine, and it can't judge meaning.
5. Run the checker again until it reports no findings, or only ones you've explained.
6. Return the rewrite, the final checker report, and a short list of terms you made consistent.

If you can't run scripts here, apply the same checks by hand and say so.

## Example

Before: "Prior to commencing the removal of the filter, it is essential that the system is depressurized in order to prevent injury."

After:
> WARNING: RELEASE THE PRESSURE BEFORE YOU REMOVE THE FILTER. PRESSURE CAN CAUSE INJURY.
>
> 1. Open the bleed valve.
> 2. Make sure that the pressure gauge shows 0 bar.
> 3. Remove the filter.
