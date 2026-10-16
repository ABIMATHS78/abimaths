"""Download the fonts fix_text.py chooses from: get_fonts.py [out_dir]

The Google Fonts families below, in the weights the corrections were made with, as WOFF files (which Pillow reads)
in <out_dir> (default: fonts/ in the work folder, see paths.py). fix_text.py also uses the system's fonts: on Debian
or Ubuntu, apt install fonts-inter fonts-liberation fonts-crosextra-carlito fonts-crosextra-caladea
fonts-dejavu-core fonts-freefont-ttf.
"""
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK  # noqa: E402

# Asked for as an old browser, Google Fonts serves WOFF rather than WOFF2.
UA = ("Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.59.10 (KHTML, like Gecko) "
      "Version/5.1.9 Safari/534.59.10")
FAMILIES = {
    "Anton": [400],
    "Antonio": [300, 400, 500, 600, 700],
    "Archivo": [300, 400, 500, 600, 700, 800],
    "Archivo Black": [400],
    "Archivo Narrow": [400, 500, 600, 700],
    "Asap Condensed": [300, 400, 500, 600, 700],
    "Barlow": [300, 400, 500, 600, 700, 800],
    "Barlow Condensed": [300, 400, 500, 600, 700],
    "Barlow Semi Condensed": [300, 400, 500, 600, 700],
    "Bebas Neue": [400],
    "Cabin": [400, 500, 600, 700],
    "Crimson Pro": [400, 500, 600, 700],
    "DM Sans": [300, 400, 500, 600, 700, 800],
    "DM Serif Display": [400],
    "EB Garamond": [400, 500, 600],
    "Encode Sans Condensed": [300, 400, 500, 600, 700],
    "Exo 2": [300, 400, 500, 600, 700],
    "Figtree": [300, 400, 500, 600, 700, 800],
    "Fira Code": [400, 500],
    "Fira Sans": [300, 400, 500, 600, 700],
    "Fira Sans Condensed": [300, 400, 500, 600, 700],
    "Fjalla One": [400],
    "Fraunces": [500, 600, 700, 800],
    "Heebo": [300, 400, 500, 600, 700, 800],
    "Hind": [300, 400, 500, 600, 700],
    "IBM Plex Mono": [400, 500, 600],
    "IBM Plex Sans": [300, 400, 500, 600, 700],
    "IBM Plex Sans Condensed": [300, 400, 500, 600, 700],
    "Inconsolata": [400, 500, 700],
    "JetBrains Mono": [400, 500, 700],
    "Josefin Sans": [300, 400, 500, 600, 700],
    "Kanit": [300, 400, 500, 600],
    "Karla": [300, 400, 500, 600, 700],
    "Lato": [300, 400, 700, 900],
    "League Gothic": [400],
    "Lexend": [300, 400, 500, 600, 700],
    "Libre Baskerville": [400, 700],
    "Libre Franklin": [300, 400, 500, 600, 700, 800],
    "Lora": [400, 500, 600, 700],
    "Manrope": [300, 400, 500, 600, 700, 800],
    "Merriweather": [400, 700, 900],
    "Merriweather Sans": [300, 400, 700],
    "Montserrat": [300, 400, 500, 600, 700, 800, 900],
    "Mulish": [300, 400, 500, 600, 700, 800],
    "Noto Sans": [300, 400, 500, 600, 700],
    "Noto Serif": [400, 600, 700],
    "Nunito": [400, 600, 700, 800],
    "Nunito Sans": [300, 400, 600, 700, 800],
    "Open Sans": [300, 400, 500, 600, 700, 800],
    "Oswald": [300, 400, 500, 600, 700],
    "Outfit": [300, 400, 500, 600, 700],
    "PT Sans": [400, 700],
    "PT Sans Narrow": [400, 700],
    "PT Serif": [400, 700],
    "Pathway Gothic One": [400],
    "Playfair Display": [400, 500, 600, 700, 800],
    "Plus Jakarta Sans": [300, 400, 500, 600, 700, 800],
    "Poppins": [300, 400, 500, 600, 700, 800],
    "Prompt": [300, 400, 500, 600],
    "Public Sans": [300, 400, 500, 600, 700, 800],
    "Rajdhani": [300, 400, 500, 600, 700],
    "Raleway": [300, 400, 500, 600, 700, 800],
    "Roboto": [300, 400, 500, 700, 900],
    "Roboto Condensed": [300, 400, 500, 600, 700, 800],
    "Roboto Mono": [400, 500, 700],
    "Rubik": [400, 500, 600, 700],
    "Saira Condensed": [300, 400, 500, 600, 700],
    "Sora": [300, 400, 600, 700],
    "Source Code Pro": [400, 500, 600],
    "Source Sans 3": [300, 400, 500, 600, 700, 800],
    "Source Serif 4": [400, 600, 700],
    "Space Grotesk": [400, 500, 600, 700],
    "Teko": [300, 400, 500, 600],
    "Titillium Web": [300, 400, 600, 700],
    "Ubuntu": [300, 400, 500, 700],
    "Urbanist": [400, 500, 600, 700],
    "Work Sans": [300, 400, 500, 600, 700],
}


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    failed = []
    for fam, weights in FAMILIES.items():
        name = fam.replace(" ", "")
        todo = [w for w in weights if not os.path.exists(os.path.join(out, f"{name}-{w}.woff"))]
        if not todo:
            continue
        url = "https://fonts.googleapis.com/css2?family=" + fam.replace(" ", "+") + ":wght@" + ";".join(map(str, todo))
        try:
            css = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30).read().decode()
        except OSError as ex:
            failed.append(f"{fam}: {ex}")
            continue
        for block in css.split("@font-face")[1:]:
            w = re.search(r"font-weight:\s*(\d+)", block)
            u = re.search(r"url\((https://[^)]+\.woff)\)", block)
            if not (w and u):
                continue
            dest = os.path.join(out, f"{name}-{w.group(1)}.woff")
            if os.path.exists(dest):
                continue
            try:
                data = urllib.request.urlopen(urllib.request.Request(u.group(1), headers={"User-Agent": UA}), timeout=30).read()
            except OSError as ex:
                failed.append(f"{fam} {w.group(1)}: {ex}")
                continue
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
    have = [f for f in os.listdir(out) if f.endswith(".woff")]
    print(f"{len(have)} fonts in {out}")
    for line in failed:
        print("failed:", line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "fonts"))
