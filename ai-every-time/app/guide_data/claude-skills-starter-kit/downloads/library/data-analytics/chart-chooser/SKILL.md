---
name: chart-chooser
description: Recommends the right chart for a message and a dataset, with a title that states the point, clean design settings and alt text. Use before building a chart for a report, slide or dashboard.
---

# Chart chooser

Pick the chart that makes the point obvious at a glance, then design it so nothing distracts from that point.

## Before you start

You need the data, or a description of its columns, and the one message the chart should get across. Also ask who will see it and where: a slide, a printed report, a dashboard or a phone. Ask for anything missing in one short message.

## Steps

1. **Write the message as a sentence**, such as "Sales in the north grew fastest this year". If there's no message yet, the data needs exploring first.
2. **Match the message to a chart:**
   - comparing categories: a bar chart, horizontal when labels are long, sorted unless the order means something;
   - change over time: a line chart; bars for a few periods;
   - parts of a whole: a stacked bar, or a simple bar of shares; a pie only for two or three parts;
   - distribution: a histogram, or box plots to compare groups;
   - relationship between two measures: a scatter plot;
   - a single number: a big number with its comparison, not a chart.
3. **Design for the message:**
   - make the title state the point, and use the subtitle for units and the period;
   - highlight what matters in one strong colour and keep everything else grey;
   - label lines and bars directly instead of using a legend, where possible;
   - start bar charts at zero; a line chart can start elsewhere if the axis is clearly labelled;
   - remove 3D effects, heavy gridlines, shadows and decoration.
4. **Make it accessible:** use colours that colour-blind readers can tell apart, never rely on colour alone, keep text large enough to read, and write alt text that gives the point and the key numbers.
5. **Offer the build steps** in the person's tool, such as Excel, Google Sheets, Python or a dashboard tool, if they ask.

## Output

```
**Message**: <the sentence the chart proves>
**Chart**: <type>, because <reason>
**Set-up**: x = <field>, y = <field>, sorted by <field>, highlight <series>
**Title**: <states the point>
**Subtitle**: <units, period, source>
**Alt text**: <the point and the key numbers>
**Avoid**: <the chart types that would mislead or confuse, and why>
```

## Rules

- Never invent or round data to make the point look stronger. Show what the data shows.
- Don't use two y-axes; they make any two lines look related. Use two charts side by side instead.
- Keep scales the same across charts that will be compared.
- Say so if the data can't support the message, rather than choosing a chart that hides the problem.

## Checklist before you finish

- [ ] The title states the point.
- [ ] The chart type suits the message.
- [ ] Bars start at zero, and scales are honest.
- [ ] Colours work for colour-blind readers, and alt text is written.
