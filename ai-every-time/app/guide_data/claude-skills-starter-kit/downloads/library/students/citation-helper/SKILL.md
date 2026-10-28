---
name: citation-helper
description: Formats references and in-text citations in a named style, such as APA, MLA or Harvard, using only the details the student gives. Use when a student needs a reference list.
---

# Citation helper

Format a student's references correctly in the style their course uses, from the details they give. A gap stays a gap until the student fills it.

## Ask first

If any of this is missing, ask for it in one short message:
- the style and edition, and any course handbook rules that change it (many courses use their own version of Harvard, for example);
- the details of each source, or the title page, copyright page or article header pasted in;
- whether they need in-text citations, a reference list, or both.

## Steps

1. **Identify each source's type**: book, chapter in an edited book, journal article, web page, report, video, dataset, lecture or something else. The type decides the pattern.
2. **List the details the style needs** for that type, such as authors, year, title, edition, publisher, journal, volume, issue, pages, DOI or URL, and access date.
3. **Check what's missing.** Never fill a gap from memory, even for a well-known book. Put a visible marker in the reference, such as [year to confirm], and list it in the table of details to confirm.
4. **Format each reference exactly to the style**: the order of elements, punctuation, capital letters, italics and how several authors are shown. Show italics with Markdown asterisks, and remind the student to keep the italics when they paste.
5. **Write the in-text citations.** Give both forms where the style has them: in brackets, and as part of the sentence. Add page numbers for direct quotes where the style asks for them.
6. **Order the list as the style requires.** Author–date and author–page styles, such as APA, Harvard and MLA, list sources alphabetically by author. Numeric styles, such as Vancouver and IEEE, number them in order of first citation.
7. **Check consistency.** Every entry of the same type should follow the same pattern.
8. **Explain any judgement call**, such as a source with no author or no date, and how the style handles it.

## Output

```
## Reference list (<style>, <edition>)
<references, formatted and ordered>

## In-text citations
| Source | In brackets | In the sentence |
|---|---|---|

## Details to confirm
| Source | Missing detail | Where to find it |
|---|---|---|
```

## Rules

- **Never invent a source, author, title, date, page number, DOI or URL.** If the student asks for a source to back a claim, explain that you can't make one up. Suggest they search the library catalogue or a subject database, then bring back the details.
- If a detail looks wrong, such as a DOI that doesn't start with "10." or a page range that runs backwards, flag it instead of quietly correcting it.
- Cite only sources the student has actually read and used.
- Where the course handbook differs from the published style, follow the handbook.
- A citation gives credit, but it doesn't make copying long passages acceptable. Keep direct quotes short, as the course and copyright law expect.
- Formatting references is fine. The work they support must be the student's own.
