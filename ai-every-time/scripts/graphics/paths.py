"""Where the graphics work happens.

GRAPHICS_WORK: a folder for the original graphics, what the text reader found in them, the corrected graphics, the
views made for checking them and the fonts (about 2 GB in all). Default: ~/.cache/ai-every-time-graphics.
GUIDES_FILE: the guides, app/guides.json.gz in the app.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_beside = os.path.normpath(os.path.join(HERE, "..", "work"))  # where the corrections were first made
WORK = os.path.normpath(os.environ.get("GRAPHICS_WORK")
                        or (_beside if os.path.isdir(os.path.join(_beside, "orig"))
                            else os.path.expanduser("~/.cache/ai-every-time-graphics")))
_repo_guides = os.path.normpath(os.path.join(HERE, "..", "..", "app", "guides.json.gz"))
GUIDES = os.environ.get("GUIDES_FILE") or (_repo_guides if os.path.exists(_repo_guides)
                                           else "/home/user/abimaths/ai-every-time/app/guides.json.gz")
REPO = os.path.dirname(os.path.dirname(GUIDES))
