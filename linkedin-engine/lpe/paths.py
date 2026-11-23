"""Where the engine keeps its files. Everything is relative to the linkedin-engine folder."""

from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = ENGINE_DIR / "templates"
ASSETS_DIR = ENGINE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

TRAINING_DIR = ENGINE_DIR / "training"
DATA_DIR = TRAINING_DIR / "data"                       # downloaded training posts (git-ignored)
ANNOTATIONS_PATH = TRAINING_DIR / "annotations.json"   # what each training graphic is (committed)
SOURCES_PATH = TRAINING_DIR / "sources.local.json"     # the Drive folder you trained from (git-ignored)

MODEL_DIR = ENGINE_DIR / "model"
PROFILE_PATH = MODEL_DIR / "style_profile.json"        # the trained style, no third-party text (committed)
VOICE_GUIDE_PATH = MODEL_DIR / "voice_guide.md"        # the same, readable, used as Claude's brief (committed)
CORPUS_PATH = MODEL_DIR / "corpus.jsonl"               # full training texts for retrieval (git-ignored)

BRAND_PATH = ENGINE_DIR / "brand.json"
OUTPUT_DIR = ENGINE_DIR / "output"                     # generated posts (git-ignored)
