"""Read every line of text in each image, with its box, into ocr_dir/<id>.json: ocr_all.py in_dir ocr_dir worker_index workers"""
import glob, json, os, sys
from rapidocr_onnxruntime import RapidOCR
src, dst, k, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
engine = RapidOCR()
files = sorted(glob.glob(os.path.join(src, "*.png")))[k::n]
for path in files:
    gid = os.path.basename(path)[:-4]
    out = os.path.join(dst, f"{gid}.json")
    if os.path.exists(out):
        continue
    result, _ = engine(path)
    lines = []
    for box, text, score in result or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        lines.append({"box": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))], "text": text,
                      "score": round(float(score), 3)})
    lines.sort(key=lambda l: (l["box"][1] // 12, l["box"][0]))
    json.dump(lines, open(out + ".part", "w"), ensure_ascii=False)
    os.replace(out + ".part", out)
