"""scripts/import_guides.py: how it repairs the guides' text and describes their infographics."""

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "import_guides.py"
spec = importlib.util.spec_from_file_location("import_guides", SCRIPT)
ig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ig)


@pytest.mark.parametrize("damaged, repaired", [
    ("Comply with Gdpr and Hipaa using Onnx on Cuda", "Comply with GDPR and HIPAA using ONNX on CUDA"),
    ("a Readme, Yaml files and Https", "a README, YAML files and HTTPS"),
    ("call our Rest APIs and Restful services", "call our REST APIs and RESTful services"),
    ("explain it with SHAP or Lime", "explain it with SHAP or LIME"),
    ("standard Select and Insert statements", "standard SELECT and INSERT statements"),
    ("use Group BY and Union ALL", "use GROUP BY and UNION ALL"),
    ("run Explain Analyze first", "run EXPLAIN ANALYZE first"),
    ("send Post requests", "send POST requests"),
    ("images from Dall E", "images from DALL-E"),
    ("using the **With** keyword", "using the **WITH** keyword"),
    ("The Costar Framework and the Star Method", "The CO-STAR Framework and the STAR Method"),
])
def test_capitals_the_site_took_away_come_back(damaged, repaired):
    assert ig.fix_prose(damaged) == repaired


@pytest.mark.parametrize("text", ["Select a model that fits", "Where it helps most", "Lime green accents",
                                  "a well-rested team"])
def test_words_that_only_look_like_acronyms_are_left_alone(text):
    assert ig.fix_prose(text) == text


def test_sql_in_code_gets_its_keywords_back_but_not_its_strings_or_comments():
    sql = "Create Table users (\n  name Varchar(100) NOT Null Default 'Select me' -- Select is a word here\n);"
    assert ig.fix_code(sql, "sql") == (
        "CREATE TABLE users (\n  name VARCHAR(100) NOT NULL DEFAULT 'Select me' -- Select is a word here\n);")
    # In another language, SQL lives in strings; the program's own names stay.
    py = 'from sqlalchemy import Select, Table\ncur.execute("Select name From users Where id = ?")\nprint("Select a model")'
    assert ig.fix_code(py, "python") == (
        'from sqlalchemy import Select, Table\ncur.execute("SELECT name FROM users WHERE id = ?")\nprint("Select a model")')
    assert ig.fix_code("From python:3.12\nWorkdir /app\nCopy . .", "dockerfile") == "FROM python:3.12\nWORKDIR /app\nCOPY . ."
    assert ig.fix_code("curl -X Post https://x.test", "bash") == "curl -X POST https://x.test"
    assert ig.fix_prose("use `Alter Table Rebuild` or `Select`") == "use `ALTER TABLE Rebuild` or `Select`"


@pytest.mark.parametrize("damaged, repaired", [
    ("multi agent systems and pre trained models", "multi-agent systems and pre-trained models"),
    ("real time dashboards", "real-time dashboards"),
    ("process events in real time based on rules", "process events in real time based on rules"),
    ("an end to end, human in the loop workflow", "an end-to-end, human-in-the-loop workflow"),
    ("rule based routing and event driven design", "rule-based routing and event-driven design"),
    ("decisions based solely on data", "decisions based solely on data"),
    ("the price is based on usage", "the price is based on usage"),
    ("catch out of memory errors", "catch out-of-memory errors"),
    ("nodes run out of memory trying to load", "nodes run out of memory trying to load"),
    ("an in memory cache", "an in-memory cache"),
    ("keep data in memory simultaneously", "keep data in memory simultaneously"),
    ("built in functions", "built-in functions"),
    ("built in Python", "built in Python"),
    ("a key value store", "a key-value store"),
    ("a key value proposition", "a key value proposition"),
    ("no code platforms", "no-code platforms"),
    ("no code changes", "no code changes"),
    ("a step by step guide", "a step-by-step guide"),
    ("walk through step by step how", "walk through step by step how"),
    ("Your Step by Step Breakdown", "Your Step-by-Step Breakdown"),
    ("fine tuning a self hosted model", "fine-tuning a self-hosted model"),
    ("your future self will thank you", "your future self will thank you"),
])
def test_hyphens_come_back_where_they_belong(damaged, repaired):
    assert ig.fix_prose(damaged) == repaired


def test_formulas_become_readable_text():
    assert ig.latex_text(r"W_0 \in \mathbb{R}^{d \times k}") == "W₀ ∈ ℝ^(d × k)"
    assert ig.latex_text(r"\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V") == "softmax((Q Kᵀ) / (√d_k)) V"
    assert ig.fix_math(r"Chains ($X \rightarrow Z$) cost $5 and $10. Keep `a \to b`.") == \
        "Chains (X → Z) cost $5 and $10. Keep `a \\to b`."
    assert ig.fix_math("$$\\text{Cost} = V \\times C_{api}$$") .strip() == "```math\nCost = V × C_api\n```"
    # Escapes the export turned into control characters.
    assert ig.fix_escapes("tool.\n\rightarrow Create\n\rapi_key = 1") == "tool.\n\\rightarrow Create\n\napi_key = 1"


def test_a_guide_body_is_repaired_whole():
    body = ("# Asyncio for AI\n\n## Introduction\n\nIt Is Fast.**.\n\n## Core Concepts\n\nUse Cuda.\n\n"
            "```python\n## not a heading\nx = 'Gdpr'\n```\n\n## Core Concepts\n\nUse Cuda.\n\n## Key Takeaways With Details\n\n- One\n")
    fixed = ig.fix_markdown(body, "Asyncio for AI")
    assert not fixed.startswith("# ")  # the title isn't repeated as a heading
    assert "It Is Fast." in fixed and ".**." not in fixed
    assert fixed.count("## Core Concepts") == 1  # a section the site repeated is kept once
    assert "## not a heading\nx = 'GDPR'" in fixed  # code is code, even with # lines
    assert "## Key Takeaways with Details" in fixed  # headings: small words in lower case


@pytest.mark.parametrize("raw, cased", [
    ("Key Takeaways With Details", "Key Takeaways with Details"),
    ("Step 2: The Setup And The Rest", "Step 2: The Setup and the Rest"),
    ("1. The SaaS Dashboard Template", "1. The SaaS Dashboard Template"),
    ("IV. The Fourth Way Forward", "IV. The Fourth Way Forward"),
])
def test_headings_keep_small_words_small_except_first(raw, cased):
    assert ig.title_case(raw) == cased


@pytest.mark.parametrize("raw, shown", [
    ("Step 3: Configuring Statistical Drift Metrics", "Configuring Statistical Drift Metrics"),
    ("Minute 0 To 5: Setting Up Your Workspace", "Setting Up Your Workspace"),
    ("Mistake: Writing a recursive CTE without a limit", "Writing a recursive CTE without a limit"),
    ("Step 1 and 2 parse the handbook", "Step 1 and 2 parse the handbook"),
    ("3D Rendering Pipelines", "3D Rendering Pipelines"),
    ("Launch ComfyUI and clear the default workspace by selecting the clear option from the main menu",
     "Launch ComfyUI and clear the default workspace"),
])
def test_infographic_labels(raw, shown):
    assert ig.label(raw) == shown


def guide_body(extra=""):
    return ("## Introduction\n\nWhy it matters.\n\n## Core Concepts\n\n### Event Loop\n\nText.\n\n### Coroutines and Tasks\n\nText.\n\n"
            "## Step-by-Step Setup\n\n### Step 1: Install Python\n\nText.\n\n### Step 2: Write the Client\n\nText.\n\n"
            "### Step 3: Run It\n\nText.\n\n## Common Mistakes\n\n- **Blocking the loop**: text\n- **Forgetting await**: text\n"
            "- **Too many tasks**: text\n\n## Action Checklist\n\n- [ ] Install Python\n- [ ] Write the client\n- [ ] Run it\n\n"
            "## Key Takeaways\n\n- Asyncio runs many requests at once on a single thread.\n" + extra)


def test_the_infographic_is_built_from_the_guide_itself():
    g = {"id": "1", "kind": "tutorial", "title": "Asyncio for AI", "summary": "Concurrency, simply.", "body": guide_body()}
    flow = ig.graphic(g, "tool_guide")
    assert flow["layout"] == "flow" and flow["steps"] == ["Install Python", "Write the Client", "Run It"]
    assert flow["banner"] == "Asyncio runs many requests at once on a single thread."
    assert flow["sections"][0] == {"title": "Core Concepts", "items": ["Event Loop", "Coroutines and Tasks"], "icon": "layers"}
    assert flow["palette"] in ig.PALETTES and (flow["title"], flow["subtitle"]) == ("Asyncio for AI", "Concurrency, simply.")
    icons = [s["icon"] for s in flow["sections"]]
    assert len(icons) == len(set(icons))  # no two sections share an icon
    sheet = ig.graphic(g | {"kind": "cheat_sheet"}, "cheat_sheet")
    assert sheet["layout"] == "cheatsheet" and len(sheet["sections"]) >= 2
    no_steps = g | {"body": guide_body().replace("Step 1: ", "").replace("Step 2: ", "").replace("Step 3: ", "")}
    versus = ig.graphic(no_steps, "tool_guide")
    assert versus["layout"] == "versus"
    assert versus["avoid"]["items"] == ["Blocking the loop", "Forgetting await", "Too many tasks"]
    assert versus["do"] == {"title": "Action checklist", "items": ["Install Python", "Write the client", "Run it"]}


def test_the_import_reads_an_export(tmp_path):
    reporters = tmp_path / "reporters.json"
    reporters.write_text(json.dumps([{"result": {"data": {"json": [{"id": 7, "slug": "aoife-murphy"},
                                                                    {"id": 8, "slug": "someone-else"}]}}}]))
    export = tmp_path / "articles.csv"
    cols = ["id", "title", "summary", "content", "category", "authorId", "publishedAt", "contentMode", "format",
            "difficulty", "articleType"]
    rows = [
        ["1", "Asyncio for AI Building High Performance Apps", "Learn Cuda.", guide_body(), "Prompt Engineering", "7",
         "1788224796728", "education", "Tutorial", "advanced", "tool_guide"],
        ["2", "Too early", "x", guide_body(), "AI", "7", "1756000000000", "education", "Tutorial", "beginner", "x"],
        ["3", "News, not a guide", "x", guide_body(), "AI", "7", "1788224796728", "news", "", "beginner", "news"],
        ["4", "Not a starter", "x", guide_body(), "AI", "8", "1788224796728", "education", "Cheat Sheet", "beginner", "x"],
    ]
    with export.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    out = tmp_path / "guides.json.gz"
    ig.main([str(export), "--since", "2026-09-01", "--reporters", str(reporters), "--out", str(out)])
    data = json.load(gzip.open(out, "rt", encoding="utf-8"))
    assert [g["id"] for g in data["guides"]] == ["1"]
    g = data["guides"][0]
    assert (g["companion"], g["kind"], g["level"], g["published"]) == ("aoife-murphy", "tutorial", "advanced",
                                                                       "2026-09-01T01:06:36Z")
    assert g["title"] == "Asyncio for AI Building High-Performance Apps" and g["summary"] == "Learn CUDA."
    assert g["minutes"] >= 1 and g["graphic"]["layout"] == "flow"
