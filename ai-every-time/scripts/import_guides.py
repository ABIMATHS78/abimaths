"""Bring the education guides from aieverytime.org into the app as posts by the companions who wrote them.

The input is an export of the site's articles table (a CSV) and its list of reporters, the site's name for
the people the starter companions are. Guides are the articles in education mode: tutorials and cheat sheets.

    python scripts/import_guides.py articles.csv --since 2026-09-01

writes app/guides.json.gz, which the app loads when it starts (see app/guides.py). The reporters list comes
from the site's API unless --reporters names a saved copy.

On the way the text is repaired. The site's pipeline had damaged it in three ways, each reversed here:

- Words in capitals of four letters or more were put in title case, except a few such as JSON and HTML:
  "GDPR" became "Gdpr", and SQL in code examples read "Create Table … Varchar(100) NOT Null".
- Hyphens became spaces: "multi agent", "pre trained", "real time data", "end to end".
- A guide's last paragraph could end in a stray bold marker: "today.**."

Spelling slips are corrected from a reviewed list, and scripts/guide_edits.json holds editorial changes to
titles, summaries and topics, by article id. Each guide keeps the address of its graphic on the site; the app
shows the corrected copy of it in static/guides/ (see scripts/graphics/README.md).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app"
OUT = APP / "guides.json.gz"
EDITS = HERE / "guide_edits.json"
REPORTERS_URL = ("https://www.aieverytime.org/api/trpc/reporters.list?batch=1&input="
                 "%7B%220%22%3A%7B%22json%22%3Anull%2C%22meta%22%3A%7B%22values%22%3A%5B%22undefined%22%5D%7D%7D%7D")
KINDS = {"Tutorial": "tutorial", "Cheat Sheet": "cheat_sheet"}
WORDS_PER_MINUTE = 230

# ------------------------------------------------------------------------------------------------- capitals

# Acronyms and names the site turned into title case, as they should be written.
ACRONYMS = {
    "Accountadmin": "ACCOUNTADMIN", "Acwr": "ACWR", "Admet": "ADMET", "Ansi": "ANSI", "Arima": "ARIMA", "Asgi": "ASGI",
    "Ascap": "ASCAP", "Ascii": "ASCII", "Baai": "BAAI", "Bigint": "BIGINT", "Bleu": "BLEU", "Bytea": "BYTEA",
    "Captcha": "CAPTCHA", "Ccpa": "CCPA", "Cefr": "CEFR", "Cifar": "CIFAR", "Cmyk": "CMYK", "Cors": "CORS",
    "Ctas": "CTAS", "Ctgan": "CTGAN", "Cuda": "CUDA", "Cvat": "CVAT", "Dateadd": "DATEADD", "Datediff": "DATEDIFF",
    "Dcgan": "DCGAN", "Dcgm": "DCGM", "Ddim": "DDIM", "Desc": "DESC", "Dkim": "DKIM", "Dmarc": "DMARC",
    "Exif": "EXIF", "Faiss": "FAISS", "Fdir": "FDIR", "Ferpa": "FERPA", "Fgsm": "FGSM", "Fhir": "FHIR",
    "Fsdp": "FSDP", "Gdpr": "GDPR", "Gelu": "GELU", "Gexf": "GEXF", "Gguf": "GGUF", "Gltf": "glTF",
    "Gptq": "GPTQ", "Gtfs": "GTFS", "Gtsam": "GTSAM", "Hipaa": "HIPAA", "Hmac": "HMAC", "Hnsw": "HNSW",
    "Https": "HTTPS", "Httpx": "HTTPX", "Hvac": "HVAC", "Ieee": "IEEE", "Iops": "IOPS", "Isoxml": "ISOXML",
    "Isrc": "ISRC", "Jdbc": "JDBC", "Jpeg": "JPEG", "Jsonb": "JSONB", "Jsonl": "JSONL", "Khtml": "KHTML",
    "Lalal": "LALAL", "Landr": "LANDR", "Lcel": "LCEL", "Ldap": "LDAP", "Lstm": "LSTM", "Lufs": "LUFS",
    "Mape": "MAPE", "Mitre": "MITRE", "Mnist": "MNIST", "Mqtt": "MQTT", "Nasa": "NASA", "Ndcg": "NDCG",
    "Ndvi": "NDVI", "Nist": "NIST", "Nnapi": "NNAPI", "Ntcip": "NTCIP", "Ntile": "NTILE", "Numa": "NUMA",
    "Nvda": "NVDA", "Nvidia": "NVIDIA", "Olap": "OLAP", "Onnx": "ONNX", "Owasp": "OWASP", "Peft": "PEFT",
    "Posix": "POSIX", "Pragma": "PRAGMA", "Qgis": "QGIS", "Qsar": "QSAR", "Rbac": "RBAC", "Rdma": "RDMA",
    "Readme": "README", "Rmse": "RMSE", "Rtsp": "RTSP", "Saml": "SAML", "Sarimax": "SARIMAX", "Sasl": "SASL",
    "Sast": "SAST", "Scada": "SCADA", "Sdxl": "SDXL", "Sftp": "SFTP", "Shap": "SHAP", "Siem": "SIEM",
    "Simd": "SIMD", "Sram": "SRAM", "Toml": "TOML", "Tpot": "TPOT", "Ttft": "TTFT", "Tvae": "TVAE",
    "Undrop": "UNDROP", "Urdf": "URDF", "Usdz": "USDZ", "Uuid": "UUID", "Varchar": "VARCHAR", "Vram": "VRAM",
    "Wcag": "WCAG", "Workdir": "WORKDIR", "Wsgi": "WSGI", "Yaml": "YAML", "Yara": "YARA", "Yolo": "YOLO",
    "Yyyy": "YYYY", "Yyyymmdd": "YYYYMMDD", "Zstd": "ZSTD", "Rlhf": "RLHF", "Mteb": "MTEB", "Oidc": "OIDC",
    "Csrf": "CSRF", "Cvss": "CVSS", "Pii": "PII", "Isr": "ISR", "Bert": "BERT", "Smote": "SMOTE", "Rouge": "ROUGE",
}
_ACRONYM = re.compile(r"\b(" + "|".join(sorted(ACRONYMS, key=len, reverse=True)) + r")\b")
_DALL_E = re.compile(r"\bDall[- ]?E\b")
# Acronyms that are also words, restored only where they can only be the acronym.
_IN_CONTEXT = [
    (re.compile(r"\bRest(?= (?:APIs?|endpoints?|services?|calls?|interfaces?|architecture|clients?|requests?)\b)"), "REST"),
    (re.compile(r"\bRestful\b"), "RESTful"),
    (re.compile(r"\bSoar(?= (?:playbooks?|platforms?|tools?|solutions?|workflows?|systems?|integrations?)\b)"), "SOAR"),
    (re.compile(r"\bLime\b(?! [Gg]reen)"), "LIME"),
    (re.compile(r"\bCoco(?= (?:JSON|dataset|format|annotations?|benchmark|keypoints?)\b)"), "COCO"),
    (re.compile(r"\bCostar\b"), "CO-STAR"),  # the prompt framework
    (re.compile(r"\bStar(?= [Mm]ethod\b)"), "STAR"),  # the interview method
]

# SQL's words, for SQL in code (where the site's change broke it) and in prose ("Select statements").
SQL_WORDS = frozenset("""
ADD AFTER ALTER ALWAYS ANALYZE ARRAY ATTACH AUTOINCREMENT BEFORE BEGIN BETWEEN BIGINT BIGSERIAL BOOLEAN BUFFERS
CASCADE CASE CAST CHAR CHECK CLONE CLUSTER COALESCE COLLATE COLUMN COMMIT CONCURRENTLY CONFLICT CONSTRAINT COPY
COUNT CREATE CROSS CUBE CURRENT DATABASE DATE DATEADD DATEDIFF DECIMAL DECLARE DEFAULT DELETE DENSE_RANK DESC
DESCRIBE DETACH DISTINCT DOUBLE DROP EACH ELSE EXCEPT EXECUTE EXISTS EXPLAIN EXTENSION EXTRACT FALSE FETCH FILTER
FIRST FLATTEN FLOAT FLOAT64 FOLLOWING FOREIGN FROM FULL FUNCTION GENERATED GRANT GROUP GROUPING HAVING IDENTITY
ILIKE INDEX INNER INSERT INSTALL INT64 INTEGER INTERSECT INTERVAL INTO JOIN JSON JSONB LAST LATERAL LEAD LEFT
LIKE LIMIT LOAD MATCHED MATERIALIZED MERGE NATURAL NOTHING NTILE NULL NULLIF NULLS NUMERIC OFFSET ONLY ORDER
OUTER OVER PARTITION PIVOT PRAGMA PRECEDING PRECISION PRIMARY PROCEDURE QUALIFY RANGE RANK RECURSIVE REFERENCES
REFRESH REPLACE RESTRICT RETURNING RETURNS REVOKE RIGHT ROLLBACK ROWS SCHEMA SELECT SEQUENCE SERIAL SMALLINT
STRING STRUCT TABLE TABLESAMPLE TEMP TEMPORARY TEXT THEN TIME TIMESTAMP TIMESTAMPTZ TRIGGER TRUE TRUNCATE
UNBOUNDED UNDROP UNION UNIQUE UNNEST UNPIVOT UPDATE UPSERT USING VACUUM VALUES VARCHAR VARIANT VIEW WHEN WHERE
WINDOW WITH WITHIN ZONE TASK RESUME SUSPEND WAREHOUSE STAGE PIPE STREAM MODEL OPTIONS FORMAT FILE TYPE SCHEDULE
ROLE USAGE SHOW CALL RETURN LANGUAGE ACCOUNT GRANTS OWNERSHIP POLICY SECURITY LEVEL MASKING ENABLE DISABLE
CLUSTERING CLUSTERED PARTITIONED LOCATION EXTERNAL STORED PARQUET DELTA ICEBERG OPTIMIZE ZORDER PURGE DAYS HOUR
HOURS MINUTE MINUTES MONTH MONTHS YEAR YEARS WEEK WEEKS SECOND SECONDS QUARTER EPOCH
""".split())
# Clear enough to be SQL on their own (a line of code with two of them is a line of SQL).
_SQL_CORE = frozenset("SELECT FROM WHERE INSERT INTO VALUES CREATE TABLE UPDATE DELETE JOIN GROUP ORDER HAVING "
                      "UNION RETURNING VARCHAR NULL PRIMARY REFERENCES DISTINCT PARTITION OVER RECURSIVE ALTER "
                      "DROP MERGE LIMIT".split())
# A string in code: SQL in a Python or JavaScript program is written in one of these.
_STRING = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|"((?:[^"\\\n]|\\.)*)"|\'((?:[^\'\\\n]|\\.)*)\'|`([^`]*)`', re.S)
PLAIN_LANGS = frozenset({"", "text", "txt", "plaintext", "plain", "none", "output", "console"})
_TITLE_SQL = {w.title(): w for w in SQL_WORDS if w.isalpha() or w[-2:].isdigit()}
_SQL_TOKEN = re.compile(r"\b[A-Z][a-z]+(?:64)?\b|\b[A-Z]{2,}\b")
_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'|\"[^\"\n]*\"|--[^\n]*|/\*.*?\*/", re.S)
SQL_LANGS = frozenset({"sql", "postgresql", "postgres", "pgsql", "psql", "mysql", "sqlite", "plsql", "tsql",
                       "snowflake", "bigquery", "duckdb", "sparksql", "hiveql"})
DOCKERFILE = frozenset("ADD ARG CMD COPY ENTRYPOINT ENV EXPOSE FROM HEALTHCHECK LABEL ONBUILD RUN SHELL "
                       "STOPSIGNAL USER VOLUME WORKDIR".split())
_DOCKER_LINE = re.compile(r"^(\s*)(" + "|".join(w.title() for w in DOCKERFILE) + r")\b", re.M)
_HTTP_VERB = re.compile(r"(-X\s+|--request\s+)(Post|Patch|Delete|Head)\b|^(\s*)(Post|Patch|Delete|Head)(\s+/)", re.M)
_HTTP_PROSE = re.compile(r"\b(Post|Patch|Delete)(\s+(?:requests?|endpoints?|methods?|calls?|routes?|handlers?)\b|\s+/)")
_SQL_PHRASE = re.compile(
    r"\b(Select|Insert|Update|Delete|Where|From|Join|With|Returning|Having|Union|Merge|Upsert|Explain|Analyze|Vacuum|"
    r"Truncate|Pragma|Qualify|Pivot|Unpivot|Window|Over|Partition|Coalesce|Lateral|Distinct|Cast|Grant|Revoke)"
    r"(\*\*|`)?(\s+(?:statements?|clauses?|keywords?|quer(?:y|ies)|commands?|operations?|operators?|syntax|blocks?|"
    r"functions?|subquer(?:y|ies)|expressions?|conditions?|filters?|joins?|privileges?)\b)")
# "Select and Insert statements": both keywords, before the noun they share.
_SQL_PAIR = re.compile(r"\b(Select|Insert|Update|Delete|Merge|Upsert|Create|Alter|Drop|Truncate|Grant|Revoke|Where|"
                       r"Having|Join|Union|From)((?:,| and| or|,? and|,? or)\s+)(?=(?:Select|Insert|Update|Delete|Merge|"
                       r"Upsert|Create|Alter|Drop|Truncate|Grant|Revoke|Where|Having|Join|Union|From)\s+(?:statements?|"
                       r"clauses?|keywords?|quer(?:y|ies)|commands?|operations?|privileges?)\b)")
_SQL_FUNC = re.compile(r"\b(Coalesce|Nullif|Lead|Rank|Ntile|Cast|Count|Unnest|Flatten|Extract|Exists|Over)(?=\()")
_AFTER_PAREN = re.compile(r"(\)\s+)(Over|Filter|Within)\b")


def _sql_run(match: re.Match) -> str:
    """A run of SQL words in prose, like "Group BY" or "Insert Into": all capitals if it has one in capitals
    already or reads as a statement's opening."""
    words = match.group(0).split()
    upper = [w.upper() for w in words]
    if any(w.isupper() for w in words) or " ".join(upper[:2]) in _SQL_OPENINGS:
        return " ".join(upper)
    return match.group(0)


_SQL_OPENINGS = frozenset({"CREATE TABLE", "CREATE INDEX", "CREATE VIEW", "CREATE MATERIALIZED", "CREATE EXTENSION",
                           "CREATE FUNCTION", "CREATE TRIGGER", "CREATE SEQUENCE", "CREATE SCHEMA", "INSERT INTO",
                           "DELETE FROM", "ALTER TABLE", "DROP TABLE", "DROP INDEX", "LEFT JOIN", "INNER JOIN",
                           "OUTER JOIN", "CROSS JOIN", "FULL OUTER", "LATERAL JOIN", "WITH RECURSIVE", "GROUP BY",
                           "ORDER BY", "PARTITION BY", "UNION ALL", "EXPLAIN ANALYZE", "SELECT DISTINCT",
                           "ON CONFLICT", "IS NULL", "IS NOT", "NOT NULL", "PRIMARY KEY", "FOREIGN KEY",
                           "COPY INTO", "MERGE INTO", "REFRESH MATERIALIZED", "UNION DISTINCT"})
_SQL_WORD_RE = "(?:" + "|".join(sorted({w.title() for w in SQL_WORDS} | SQL_WORDS | {"BY", "ON", "IS", "NOT", "AS",
                                                                                     "ALL", "KEY", "OR", "AND", "IN"},
                                       key=len, reverse=True)) + ")"
_SQL_RUN = re.compile(rf"\b{_SQL_WORD_RE}(?:\s+{_SQL_WORD_RE})+\b")


def sql_upper(code: str) -> str:
    """SQL with its keywords back in capitals, leaving strings, quoted names and comments as they are."""
    out, last = [], 0
    for m in _SQL_LITERAL.finditer(code):
        out.append(_sql_words(code[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_sql_words(code[last:]))
    return "".join(out)


def _sql_words(text: str) -> str:
    return _SQL_TOKEN.sub(lambda m: _TITLE_SQL.get(m.group(0), m.group(0)), text)


def _sql_like(text: str, at_least: int = 2) -> bool:
    """Whether text holds SQL in the site's title case: that many of SQL's plainest keywords, one or more of them
    in title case."""
    titled = {_TITLE_SQL[w] for w in re.findall(r"\b[A-Z][a-z]+\b", text) if _TITLE_SQL.get(w) in _SQL_CORE}
    upper = {w for w in re.findall(r"\b[A-Z]{2,}\b", text) if w in _SQL_CORE}
    return bool(titled) and len(titled | upper) >= at_least


def _sql_in_strings(code: str) -> str:
    """SQL in another language's code is in its strings: those that hold SQL are put right."""
    def fix(m: re.Match) -> str:
        whole = m.group(0)
        inner = next(g for g in m.groups() if g is not None)
        if not _sql_like(inner):
            return whole
        start = whole.index(inner) if inner else 0
        return whole[:start] + _sql_words(inner) + whole[start + len(inner):]
    return _STRING.sub(fix, code)


def fix_code(code: str, lang: str) -> str:
    lang = lang.strip().lower()
    code = _ACRONYM.sub(lambda m: ACRONYMS[m.group(1)], code)
    if lang in SQL_LANGS:
        return sql_upper(code)
    if lang in ("dockerfile", "docker"):
        code = _DOCKER_LINE.sub(lambda m: m.group(1) + m.group(2).upper(), code)
    code = _HTTP_VERB.sub(lambda m: (m.group(1) + m.group(2).upper()) if m.group(1) else
                          (m.group(3) + m.group(4).upper() + m.group(5)), code)
    if (lang in PLAIN_LANGS and _sql_like(code, 3)) or (lang == "inline" and _sql_like(code)):  # unlabelled SQL
        return sql_upper(code)
    return _sql_in_strings(code)


def fix_capitals(text: str) -> str:
    """Prose: acronyms, SQL and HTTP methods back in capitals."""
    text = _ACRONYM.sub(lambda m: ACRONYMS[m.group(1)], text)
    text = _DALL_E.sub("DALL-E", text)
    for pattern, word in _IN_CONTEXT:
        text = pattern.sub(word, text)
    text = _SQL_RUN.sub(_sql_run, text)
    text = _SQL_PAIR.sub(lambda m: m.group(1).upper() + m.group(2), text)
    text = _SQL_PHRASE.sub(lambda m: m.group(1).upper() + (m.group(2) or "") + m.group(3), text)
    text = _SQL_FUNC.sub(lambda m: m.group(1).upper(), text)
    text = _AFTER_PAREN.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    return _HTTP_PROSE.sub(lambda m: m.group(1).upper() + m.group(2), text)


# -------------------------------------------------------------------------------------------------- hyphens

_PREFIX = re.compile(r"\b(multi|semi|non|pre|anti|hyper|ultra)[ ](?=[A-Za-z])", re.I)
_SELF = re.compile(r"\b(self)[ ](?=(?:attention|heal\w*|host\w*|join\w*|serv\w*|contain\w*|paced|correct\w*|"
                   r"supervis\w*|optimi[sz]\w*|document\w*|driving|improv\w*|protect\w*|consisten\w*|referenc\w*|"
                   r"harden\w*|learning|evaluat\w*|reflect\w*|critique|check\w*|managed|explanatory|sufficien\w*|"
                   r"assess\w*|regulat\w*|replicat\w*|signed|study|directed|guided|taught|aware\w*|organi[sz]\w*|"
                   r"verif\w*|play|describ\w*|report\w*|tun\w*|debug\w*|repair\w*|updat\w*)\b)", re.I)
_CROSS = re.compile(r"\b(cross)[ ](?=(?:validat\w*|functional|attention|referenc\w*|encod\w*|entropy|check\w*|"
                    r"platform|modal\w*|border|department\w*|channel|devices?|site|silo\w*|joins?|domains?|"
                    r"cloud|teams?|regions?|accounts?|lingual|sell\w*|training|origin|cutting|pollinat\w*|"
                    r"tenant|chain|environment\w*|browser|disciplinary|correlat\w*)\b)", re.I)
# Hyphenated wherever they appear.
_ALWAYS = [re.compile(p, re.I) for p in (
    r"\b(end) (to) (end)\b", r"\b(human) (in) (the) (loop)\b", r"\b(fine) (tun(?:e|ed|es|ing|able))\b",
    r"\b(zero|few) (shot)\b", r"\b(chain) (of) (thought)\b", r"\b(retrieval) (augmented)\b", r"\b(peer) (to) (peer)\b",
    r"\b(trade) (offs)\b", r"\b(plug) (and) (play)\b", r"\b(pay) (as) (you) (go)\b", r"\b(open) (ended)\b",
    r"\b(top) ([kp])\b", r"\b(k) (means|nearest|fold)\b", r"\b(n) (grams?)\b", r"\b([xyz]) (axis)\b",
    r"\b(e) (commerce)\b", r"\b(wi) (fi)\b", r"\b(sci) (fi)\b", r"\b(t) (SNE)\b",
    r"\b(production|enterprise|industrial|research|commercial|consumer|military|investment|audit) (grade)\b",
    r"\b(production|enterprise|audit|future|cloud|deployment|interview|market) (ready)\b",
    r"\b(user|beginner|developer|mobile|budget|eco|SEO|privacy|family|printer|search) (friendly)\b",
    r"\b(context|cost|privacy|latency|energy|hardware|schema|quantization|risk|security|location) (aware)\b",
    r"\b(domain|task|language|model|platform|vendor|region|industry|user|context|hardware|application|"
    r"framework|role|use|case|site) (specific)\b",
    r"\b(compute|memory|data|resource|labor|labour|time|CPU|GPU|knowledge|capital) (intensive)\b",
    r"\b(customer|user|public|client|internet|externally) (facing)\b",
    r"\b(latency|cost|privacy|time|context|case|price|safety|security) (sensitive)\b",
    r"\b(cost|energy|memory|parameter|resource|data|compute|token|sample|space|time) (efficient)\b",
    r"\b(cost) (effective)\b", r"\b(CPU|GPU|IO|memory|compute|network) (bound)\b",
    r"\b(detail|goal|object|service|action|results|task|outcome) (oriented)\b",
    r"\b(user|data|human|customer|developer|model|learner|patient|people|privacy|mobile|cloud) (centric)\b",
    r"\b(model|vendor|platform|framework|cloud|language|hardware|device|provider) (agnostic)\b",
    r"\b(AI|LLM|GPT|machine|data|model|human|computer) (generated|assisted|enabled|native|focused|led|written|"
    r"guided)\b(?! by)",
    r"\b(text|speech|image|sequence) (to) (speech|text|image|video|sequence|SQL|3D|code|audio|music)"
    r"(?= (?:models?|systems?|tools?|engines?|services?|APIs?|pipelines?|generation|conversion|synthesis|"
    r"translation|workflows?|apps?|applications?|tasks?|capabilities|features?|platforms?|generators?)\b)",
    r"\b(built) (in)(?= (?:functions?|tools?|support|features?|types?|methods?|modules?|librar(?:y|ies)|"
    r"commands?|templates?|integrations?|connectors?|capabilit(?:y|ies)|operators?|plugins?|protections?|"
    r"safeguards?|validation|security|guardrails?|retr(?:y|ies)|caching|logging|monitoring|metrics|dashboards?|"
    r"components?|nodes?|actions?|evaluators?|datasets?|encryption|versioning|scheduling|governance)\b)",
    r"\b(key) (value)(?= (?:stores?|pairs?|cache|caches|databases?|lookups?|formats?|mappings?|structures?|"
    r"data|storage)\b)",
    r"\b(no|low) (code)(?= (?:platforms?|tools?|builders?|solutions?|automations?|apps?|workflows?|"
    r"interfaces?|AI|options?|environments?|approach|editors?|integrations?|agents?)\b)",
    r"\b(on) (device)(?= (?:inference|models?|processing|AI|LLMs?|learning|training|deployment|execution|"
    r"computation|intelligence|storage|transcription|speech|vision)\b)",
    r"\b(hands) (on)(?= (?:experience|practice|labs?|projects?|exercises?|tutorials?|guides?|learning|training|"
    r"workshops?|approach|sessions?|work|coding|demos?|examples?|activit(?:y|ies)|skills?|time|testing)\b)",
    r"\b(single) (page|tenant|threaded|shot)(?= [a-z])",
    r"\b(in) (memory)(?= (?:databases?|stores?|caches?|caching|data|storage|processing|computing|tables?|index(?:es)?|"
    r"buffers?|queues?|structures?|representations?|cop(?:y|ies)|state|vectors?|graphs?|engines?|analytics|"
    r"approach|mode|implementations?|solutions?|options?|layers?|lookups?|joins?|operations?)\b)",
    r"\b(in) (depth)(?= (?:guides?|analys[ie]s|look|reviews?|understanding|knowledge|explanations?|tutorials?|"
    r"coverage|exploration|discussions?|treatment|stud(?:y|ies)|breakdowns?|overviews?|walkthroughs?|examination|"
    r"investigations?|research|reporting|interviews?|insights?|assessments?|audits?|evaluations?|courses?)\b)",
    r"\b(in) (house)(?= (?:teams?|tools?|models?|solutions?|expertise|talent|development|capabilit(?:y|ies)|"
    r"infrastructure|systems?|data|resources|staff|engineers?|counsel|legal|lawyers?|ML|AI|platforms?|apps?|"
    r"applications?|builds?|software|research|production|training|labs?)\b)",
    r"\b(in) (context)(?= (?:learning|examples?|retrieval|prompting|windows?|demonstrations?)\b)",
    r"\b(in) (process)(?= (?:databases?|engines?|execution|analytics|OLAP|calls?|librar(?:y|ies)|mode|SQL|storage)\b)",
    r"\b(in) (place)(?= (?:updates?|operations?|editing|edits|upgrades?|modifications?|sorting|changes?|"
    r"migrations?|replacements?|mutations?)\b)",
    r"\b(in) (flight)(?= (?:requests?|transactions?|operations?|data|messages?|tasks?|work|batch(?:es)?|calls?|"
    r"jobs?|writes?|queries)\b)",
    r"\b(out) (of) (memory)(?= (?:errors?|exceptions?|crash(?:es)?|issues?|conditions?|kills?|failures?|"
    r"situations?|events?|problems?|states?|killer|scenarios?)\b)",
    r"\b(out) (of) (distribution)(?= (?:data|inputs?|samples?|detection|examples?|generalization|shifts?|"
    r"robustness|queries|scenarios?|cases?|points?)\b)",
    r"\b(out) (of) (the) (box)(?= (?:support|integrations?|solutions?|tools?|experience|functionality|features?|"
    r"configurations?|setups?|models?|templates?|capabilit(?:y|ies)|components?|connectors?|defaults?|settings?|"
    r"behavior|performance|options?|dashboards?)\b)",
    r"\b(out) (of) (band)(?= (?:communication|channels?|verification|alerts?|signals?|management|updates?|"
    r"methods?|access)\b)",
    r"\b(out) (of) (sample)(?= (?:data|testing|tests?|performance|evaluation|validation|predictions?|forecasts?|"
    r"accuracy|errors?|periods?|results?)\b)",
    r"\b(out) (of) (scope)(?= (?:items?|requests?|topics?|questions?|tasks?|features?|work|queries|"
    r"requirements?|areas?)\b)",
    r"\b(on) (demand)(?= (?:pricing|instances?|access|scaling|compute|capacity|services?|resources?|learning|"
    r"content|generation|inference|training|basis|models?|delivery|video|reports?|support|workers?|clusters?|"
    r"GPUs?|billing|tiers?|plans?)\b)",
    r"\b(on) (call)(?= (?:engineers?|rotations?|schedules?|teams?|staff|duty|shifts?|support|alerts?|"
    r"playbooks?|responders?|developers?)\b)",
    r"\b(on) (chain)(?= (?:data|transactions?|analytics|activity|identity|records?|verification|storage|"
    r"governance|metrics)\b)",
    r"\b(read|write) (only)(?= (?:access|mode|replicas?|users?|permissions?|roles?|tokens?|views?|cop(?:y|ies)|"
    r"connections?|transactions?|endpoints?|filesystems?|volumes?|mounts?|database|databases|queries|"
    r"credentials?|keys?|API|scopes?|accounts?|privileges?|attributes?|propert(?:y|ies)|fields?|snapshots?|"
    r"operations?|memory|storage|files?|tables?)\b)",
)]
# Hyphenated only before the word they describe ("real-time data", not "in real time").
_BEFORE_NOUN = re.compile(
    r"\b(real (?:time|world)|high (?:performance|quality|level|risk|volume|throughput|stakes|impact|value|"
    r"precision|resolution|dimensional|frequency|fidelity|signal|leverage|traffic|concurrency|availability|"
    r"priority|margin|density|speed|bandwidth|cardinality|confidence)|low (?:latency|level|cost|risk|"
    r"volume|rank|resource|power|precision|resolution|confidence|bandwidth|cardinality|effort|stakes|light|"
    r"maintenance|friction)|third party|open (?:source|weights?)|closed source|long (?:term|range|tail|"
    r"running|lived|context|horizon|form)|short (?:term|range|lived|form)|large scale|small scale|"
    r"full (?:stack|scale|text|time)|cutting edge|step by step|state of the art|point in time|"
    r"up to date|first (?:class|party)|world class|"
    r"mission critical|business critical|safety critical|decision making|problem solving|black box|white box|"
    r"well (?:defined|known|structured|documented|designed|tested|established|suited|organized|crafted|"
    r"maintained|formed|calibrated|tuned|engineered|written|understood|governed|paid)|one (?:time|off)|"
    r"two (?:way|step|stage|factor|tier|tower)|on premises?|off (?:the shelf|policy|chain)|data (?:heavy|rich|hungry)|"
    r"time (?:sensitive|consuming|bound|boxed|stamped|critical)|error (?:prone|free)|"
    r"lock free|rate limit(?:ed|ing)?|general purpose|special purpose|last mile|next generation|"
    r"trade off|follow up|opt in|opt out|sign up|check in|roll out|set up)"
    r"(?=\s+(?!(?:and|or|but|nor|in|on|at|for|to|with|without|within|inside|outside|across|by|from|of|is|are|"
    r"was|were|be|been|being|when|while|that|which|than|as|so|if|it|its|this|these|those|the|a|an|into|onto|"
    r"over|under|through|via|per|versus|vs|because|before|after|until|where|who|whom|whose|can|will|would|"
    r"should|could|may|might|must|do|does|did|has|have|had|not|rather|instead|also|too|more|less|only|just|"
    r"even|then|now|today|here|there|up|down|out|off|again|yet|still|about|like|unless|since|though|although|"
    r"whether|either|neither|both|each|every|all|any|some|no|such|very|really|you|your|we|our|they|their|he|"
    r"she|his|her|them|us|me|my|I|means|matters|using|against|among|between|toward|towards|during|including|"
    r"plus|except|given|until|unlike|despite|beyond|throughout|alongside|regardless|whenever|wherever|once|"
    r"immediately|automatically|quickly|easily|directly|properly|correctly|effectively|successfully|first|"
    r"next|later|soon|again|together|apart|back|away|anyway|instead|well|right|left|tomorrow|tonight|"
    r"yesterday|daily|weekly|monthly|annually|everywhere|anywhere|somewhere|nowhere|forever|ever|never|always|"
    r"often|usually|sometimes|rarely|frequently|regularly|constantly|continuously|manually|locally|globally|"
    r"internally|externally|online|offline|how|what|why|compared|based|relative|according|depending|instead|"
    r"thanks|due|prior|similar|equivalent|close|ahead|enough|alone|anymore|everything|anything|something)\b)[A-Za-z])",
    re.I)
# Before these the pair is a verb or a noun and keeps its space ("set up the server", "a follow up on").
_VERBAL = re.compile(r"^(?:trade off|follow up|opt in|opt out|sign up|check in|roll out|set up)$", re.I)


# "rule based", "event driven", "AI powered": hyphenated after any noun, but not in "based on", "is based" and the like.
_PARTICIPLE = re.compile(r"\b([A-Za-z][\w/]*) (based|driven|powered)\b(?! (?:\w+ly )?(?:on|upon|in|off|around|out|at|from|"
                         r"within|by|toward|towards|through)\b)", re.I)
_NOT_A_NOUN = frozenset("""
a an the and or but nor so yet is are was were be been being am not it its they them those these this that which who
whom whose what also all both each every either neither some any no more most less least very much many few one two
three own than then as of to for with without into onto from at by on in out over under about after before through
can could will would should may might must do does did has have had get gets got getting become becomes became
""".split())


def _participle(m: re.Match) -> str:
    word = m.group(1)
    if word.lower() in _NOT_A_NOUN or word.lower().endswith("ly"):
        return m.group(0)
    return f"{word}-{m.group(2)}"


def fix_hyphens(text: str) -> str:
    text = _PARTICIPLE.sub(_participle, text)
    text = _PREFIX.sub(r"\1-", text)
    text = _SELF.sub(r"\1-", text)
    text = _CROSS.sub(r"\1-", text)
    for pattern in _ALWAYS:
        text = pattern.sub(lambda m: "-".join(m.groups()), text)

    def before_noun(m: re.Match) -> str:
        phrase = m.group(1)
        if re.match(r"real time$", phrase, re.I) and re.search(r"\b(?:in|near)\s+$", m.string[max(0, m.start() - 6):m.start()], re.I):
            return phrase
        if _VERBAL.match(phrase):  # only as a noun before another ("a sign-up form", "follow-up questions")
            nxt = re.match(r"\s+([A-Za-z]+)", m.string[m.end():])
            if not nxt or nxt.group(1).lower() not in _AFTER_VERBAL:
                return phrase
        return phrase.replace(" ", "-")
    return _BEFORE_NOUN.sub(before_noun, text)


_AFTER_VERBAL = frozenset("""
analysis analyses questions question emails email messages message calls call meetings meeting tasks task
actions action items item reminders reminder prompts prompt forms form flows flow pages page process processes
rates rate steps step surveys survey sequences sequence workflows workflow notes note discussions discussion
plans plan time scripts script wizards wizard guides guide instructions checklists checklist costs cost
requirements screens screen consent consents mechanisms mechanism strategy strategies plan schedules schedule
""".split())


# ---------------------------------------------------------------------------------------------- whole guide

# Slips found by a dictionary check of every guide and reviewed by hand, as they're written and as they should be.
FIXES = (
    ("do and dont rules", "do's and don'ts"), ("misinterpretes", "misinterprets"),
    ("previous monthing raw", "previous month's raw"), ("up-opdate", "up-to-date"),
    ("latency percentagens", "latency percentiles"), ("ten year olds", "ten-year-olds"),
)
# Escapes in mathematical notation that the export decoded into control characters ("\times" became a tab and "imes").
_DECODED = re.compile(r"\t(?=imes\b|ext\{|ext=|heta\b|au\b|o\b)|\n(?=abla\b|eq\b|u\b)|\x0c(?=rac\{)|\x08(?=eta\b)|"
                      r"\r(?=ightarrow\b)")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _typos(text: str) -> str:
    for wrong, right in FIXES:
        text = text.replace(wrong, right)
    return text


def fix_escapes(text: str) -> str:
    text = _DECODED.sub(lambda m: {"\t": "\\t", "\n": "\\n", "\x0c": "\\f", "\x08": "\\b", "\r": "\\r"}[m.group(0)], text)
    text = text.replace("\r", "\n")  # a lone carriage return was a line break
    # Anything else of the kind, in a code example, is shown as the escape it was ("\x1f").
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group(0)):02x}", text)


# ------------------------------------------------------------------------------------------------------ math

# A few guides write formulas in LaTeX, which the app doesn't typeset: they're turned into plain characters.
SYMBOLS = {
    "times": "×", "cdot": "·", "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "leftrightarrow": "↔", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈",
    "sim": "~", "propto": "∝", "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "cup": "∪", "cap": "∩",
    "infty": "∞", "sum": "Σ", "prod": "Π", "int": "∫", "partial": "∂", "nabla": "∇", "pm": "±", "mid": "|",
    "ldots": "…", "cdots": "⋯", "dots": "…", "forall": "∀", "exists": "∃", "neg": "¬", "land": "∧", "lor": "∨",
    "oplus": "⊕", "otimes": "⊗", "circ": "∘", "star": "⋆", "ast": "*", "prime": "′", "degree": "°", "quad": " ",
    "qquad": "  ", "lVert": "‖", "rVert": "‖", "Vert": "‖", "lvert": "|", "rvert": "|", "langle": "⟨", "rangle": "⟩",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ",
    "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_SUP = str.maketrans("0123456789+-=()nTiK", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀⁱᴷ")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_SETS = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼", "P": "ℙ"}


def latex_text(expr: str) -> str:
    s = re.sub(r"\\\\+(?=[A-Za-z])", r"\\", expr.strip())  # "\\times", escaped twice on the way, is "\times"
    s = re.sub(r"\\(?:mathbb|mathbf|bm)\{([A-Z])\}", lambda m: _SETS.get(m.group(1), m.group(1)), s)
    for _ in range(4):  # from the innermost braces out
        s = re.sub(r"\\(?:text|mathrm|mathit|mathbf|mathcal|operatorname|textbf|textit|mbox|hat|bar|vec|tilde)"
                   r"\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}",
                   lambda m: f"{_group(m.group(1))} / {_group(m.group(2))}", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: f"√{_group(m.group(1))}", s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)(?=[()\[\]|.])\.?", "", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: SYMBOLS.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\\([,;:! ])", " ", s)
    s = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: _script(m.group(1) or m.group(2), _SUP, "^"), s)
    s = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: _script(m.group(1) or m.group(2), _SUB, "_"), s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _group(s: str) -> str:
    return s if re.fullmatch(r"[\w.]+", s.strip()) else f"({s.strip()})"


def _script(s: str, table: dict, mark: str) -> str:
    """A superscript or subscript in its own characters where Unicode has them all, else written out."""
    t = s.translate(table)
    if t.strip() and all(ord(c) > 127 or c.isspace() for c in t):
        return t
    return f"{mark}{s}" if re.fullmatch(r"\w+", s) else f"{mark}({s})"


_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE_MATH = re.compile(r"(?<![$\\\w])\$(?!\s)([^$\n]{1,200}?)(?<!\s)\$(?![$\w])")
_PROSE_LATEX = re.compile(r"\\+(rightarrow|leftarrow|Rightarrow|Leftarrow|leftrightarrow|times|to)\b")


def fix_math(text: str) -> str:
    """Formulas in prose: displayed ones on a line of their own, others in the sentence. Inline code is left be."""
    parts = re.split(r"(`+[^`]*`+)", text)
    return "".join(p if p.startswith("`") else _math(p) for p in parts)


def _math(text: str) -> str:
    text = _DISPLAY_MATH.sub(lambda m: f"\n\n```math\n{latex_text(m.group(1))}\n```\n\n", text)
    text = _INLINE_MATH.sub(lambda m: latex_text(m.group(1)) if re.search(r"[\\^_{}]", m.group(1)) else m.group(0),
                            text)
    return _PROSE_LATEX.sub(lambda m: SYMBOLS[m.group(1)], text)


_FENCE = re.compile(r"^(```|~~~)([^\n]*)\n(.*?)^\1[ \t]*$", re.M | re.S)
_INLINE = re.compile(r"(`+)(.+?)\1", re.S)


def fix_prose(text: str) -> str:
    """Text outside code. Inline code gets the code fixes, not the prose ones."""
    out, last = [], 0
    for m in _INLINE.finditer(text):
        out.append(_prose(text[last:m.start()]))
        out.append(m.group(1) + fix_code(m.group(2), "inline") + m.group(1))
        last = m.end()
    out.append(_prose(text[last:]))
    return "".join(out)


def _prose(text: str) -> str:
    return _typos(fix_hyphens(fix_capitals(text)))


def fix_markdown(body: str, title: str) -> str:
    """A guide's body with the damage undone."""
    body = fix_escapes(body.replace("\r\n", "\n")).strip()
    body = _math_outside_code(body)
    lines = body.split("\n")
    # A first heading that repeats the title is shown once, as the title.
    if lines and re.match(r"#\s", lines[0]) and _norm(lines[0][2:]) == _norm(title):
        body = "\n".join(lines[1:]).lstrip()
    out, last = [], 0
    for m in _FENCE.finditer(body):
        out.append(_fix_text_block(body[last:m.start()]))
        out.append(f"{m.group(1)}{m.group(2)}\n{fix_code(m.group(3), m.group(2))}{m.group(1)}")
        last = m.end()
    out.append(_fix_text_block(body[last:]))
    return dedupe_sections("".join(out)).strip() + "\n"


def _math_outside_code(body: str) -> str:
    out, last = [], 0
    for m in _FENCE.finditer(body):
        out.append(re.sub(r"\n{3,}", "\n\n", fix_math(body[last:m.start()])))
        out.append(m.group(0))
        last = m.end()
    out.append(re.sub(r"\n{3,}", "\n\n", fix_math(body[last:])))
    return "".join(out)


def _fix_text_block(text: str) -> str:
    text = re.sub(r"^# (?=\S)", "## ", _unpaired_bold(text), flags=re.M)  # the title is the only top heading
    text = fix_prose(text)
    return re.sub(r"^(#{2,6} )(.+)$", lambda m: m.group(1) + title_case(m.group(2)), text, flags=re.M)


def _unpaired_bold(text: str) -> str:
    """A paragraph whose bold markers don't pair up loses the stray one at its end ("today.**." → "today.")."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.count("**") % 2 == 1:
            lines[i] = re.sub(r"([.!?])\*\*[.!?]?\s*$", r"\1", line)
    return "\n".join(lines)


def fix_line(text: str) -> str:
    """A title, summary or heading."""
    text = re.sub(r"\s+", " ", text.replace("**", "")).strip()
    return fix_prose(text)


# Not "up", which belongs to the verb before it ("Setting Up", "Scale Up").
_SMALL = frozenset("a an and as at but by for from in into nor of on onto or over per the to via vs vs. with".split())
_LIST_NUMBER = re.compile(r"\(?(?:\d+[a-z]?|[ivxIVX]+|[a-zA-Z])[.)]")


def title_case(text: str) -> str:
    """A heading in title case with its small words ("And", "With", "Of") in lower case, as style guides have them,
    except first and after a colon."""
    words = text.split(" ")
    capped = [w for w in words if w[:1].isalpha()]
    if len(capped) < 3 or sum(w[0].isupper() for w in capped) < 0.7 * len(capped):
        return text  # not a title-case heading
    out = []
    for i, w in enumerate(words):
        # The first word, also after a colon or dash, or after the number of a numbered heading ("1. The Basics").
        start = i == 0 or out[-1].endswith((":", "–", "—", "?", "!")) or i == 1 and _LIST_NUMBER.fullmatch(out[0])
        out.append(w.lower() if not start and w.lower() in _SMALL and w[:1].isupper() and not w.isupper() else w)
    return " ".join(out)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def dedupe_sections(body: str) -> str:
    """A guide with a section repeated (some were, word for word) keeps its first copy."""
    sections, current, fence = [], [], None
    for line in body.split("\n"):
        if fence is None and re.match(r"(```|~~~)", line):
            fence = line[:3]
        elif fence and line.startswith(fence):
            fence = None
        elif fence is None and line.startswith("## "):
            sections.append(current)
            current = []
        current.append(line)
    sections.append(current)
    seen, kept = set(), []
    for lines in sections:
        if lines and lines[0].startswith("## "):
            key = _norm(lines[0])
            if key in seen:
                continue
            seen.add(key)
        kept.append("\n".join(lines))
    return "\n".join(kept)


# ---------------------------------------------------------------------------------------------------- import

def load_reporters(path: str | None) -> dict[str, str]:
    """The site's reporter ids, each with the slug of the starter companion they are."""
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(REPORTERS_URL, timeout=60) as r:
            data = json.load(r)
    rows = data[0]["result"]["data"]["json"] if isinstance(data, list) else data
    return {str(r["id"]): r["slug"] for r in rows}


def starter_slugs() -> set[str]:
    with (APP / "starter_companions.csv").open(encoding="utf-8", newline="") as f:
        return {row["slug"] for row in csv.DictReader(f)}


def minutes(body: str) -> int:
    words = len(re.findall(r"\w+", _FENCE.sub(lambda m: m.group(3), body)))
    return max(1, round(words / WORDS_PER_MINUTE))


def convert(row: dict, slug: str, edits: dict) -> dict:
    edit = edits.get(row["id"], {})
    title = title_case(edit.get("title") or fix_line(row["title"]))
    guide = {
        "id": row["id"],
        "companion": slug,
        "published": datetime.fromtimestamp(int(row["publishedAt"]) / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": KINDS[row["format"]],
        "level": row["difficulty"] or "beginner",
        "topic": edit.get("topic") or row["category"] or "AI",
        "title": title,
        "summary": edit.get("summary") or fix_line(row["summary"]),
        "body": fix_markdown(row["content"], row["title"]),
    }
    for old, new in edit.get("body", {}).items():
        guide["body"] = guide["body"].replace(old, new)
    guide["minutes"] = minutes(guide["body"])
    guide["image"] = row.get("imageUrl") or None
    return guide


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="the site's articles export (CSV)")
    ap.add_argument("--since", default="2026-09-01", help="the earliest publication date to bring in (UTC)")
    ap.add_argument("--reporters", help="a saved copy of the site's reporters list (JSON) instead of asking the site")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    since = int(datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp() * 1000)
    reporters, slugs = load_reporters(args.reporters), starter_slugs()
    edits = json.loads(EDITS.read_text(encoding="utf-8")) if EDITS.exists() else {}
    guides, skipped = [], {}
    with open(args.export, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["contentMode"] != "education" or row["format"] not in KINDS or int(row["publishedAt"]) < since:
                continue
            slug = reporters.get(row["authorId"])
            if slug not in slugs:
                skipped[row["authorId"]] = skipped.get(row["authorId"], 0) + 1
                continue
            guides.append(convert(row, slug, edits))
    guides.sort(key=lambda g: (g["published"], g["id"]))
    payload = {"since": args.since, "source": Path(args.export).name, "guides": guides}
    with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    by = {g["companion"] for g in guides}
    print(f"{len(guides)} guides by {len(by)} companions written to {args.out}")
    if skipped:
        print(f"Skipped {sum(skipped.values())} by authors who aren't starter companions: {skipped}")


if __name__ == "__main__":
    main()
