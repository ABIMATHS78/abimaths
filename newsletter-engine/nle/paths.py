"""Where everything lives. Every path is inside the engine folder, so the folder can move on its own."""

from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
SITE_CONFIG = ENGINE_DIR / "site.json"
CONTENT_DIR = ENGINE_DIR / "content"
ISSUES_DIR = CONTENT_DIR / "issues"
GUIDES_DIR = CONTENT_DIR / "guides"
MEDIA_DIR = CONTENT_DIR / "media"
TEMPLATES_DIR = ENGINE_DIR / "templates"
STATIC_DIR = ENGINE_DIR / "static"
DIST_DIR = ENGINE_DIR / "dist"
MODEL_DIR = ENGINE_DIR / "model"
PROFILE_PATH = MODEL_DIR / "style_profile.json"
STYLE_GUIDE_PATH = MODEL_DIR / "style_guide.md"
CORPUS_PATH = MODEL_DIR / "corpus.jsonl"          # the training issues' text: local only, never committed
TRAINING_DIR = ENGINE_DIR / "training"
DATA_DIR = TRAINING_DIR / "data"                  # downloaded training material: local only, never committed
SOURCES_PATH = TRAINING_DIR / "sources.local.json"
ANNOTATIONS_PATH = TRAINING_DIR / "annotations.json"
BRIEFS_DIR = ENGINE_DIR / "briefs"
