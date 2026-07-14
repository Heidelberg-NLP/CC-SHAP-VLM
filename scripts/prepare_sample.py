"""Fetch a small, self-contained sample of the VALSE ``foil_it`` instrument.

The paper reads the full VALSE ``foil-it.json`` (1000 entries) plus the referenced COCO
val2014 images. For smoke tests and the before/after regression harness we only need a
handful, so this downloads the annotation json, keeps the first ``--num`` entries that pass
the same MTurk filter ``run-faithfulness.py`` applies (``mturk['caption'] >= 2``), fetches
their COCO images, and writes a reduced ``foil-it.json`` plus images under ``--root``
(default ``data/``, matching ``MULT_CHOICE_DATA['foil_it']`` in config.py).

The ``foil`` field is written as a one-element list because the runner reads
``foil['foil'][0]`` (VALSE ships it as a plain string).

Usage (inside the uv env):
    uv run python scripts/prepare_sample.py --num 5
"""
import argparse
import json
import urllib.request
from pathlib import Path

VALSE_FOIL_IT = "https://raw.githubusercontent.com/Heidelberg-NLP/VALSE/main/data/foil-it.json"
COCO_VAL2014 = "http://images.cocodataset.org/val2014"
MIN_CAPTION_VOTES = 2  # run-faithfulness.py keeps only annotator-accepted captions


def download(url: str, dest: Path, timeout: int = 120) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return len(data)


def as_foil_list(entry: dict) -> dict:
    entry = dict(entry)
    if isinstance(entry.get("foil"), str):
        entry["foil"] = [entry["foil"]]
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=5, help="number of samples to keep")
    ap.add_argument("--root", default="data")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "COCO" / "all_images"
    ann_cache = root / "foil-it.full.json"

    if not ann_cache.exists():
        print(f"Downloading annotations: {VALSE_FOIL_IT}")
        download(VALSE_FOIL_IT, ann_cache)
    full = json.loads(ann_cache.read_text())
    print(f"Full annotation set: {len(full)} entries")

    sample: dict[str, dict] = {}
    for key, entry in full.items():
        if len(sample) >= args.num:
            break
        if entry.get("mturk", {}).get("caption", 0) < MIN_CAPTION_VOTES:
            continue
        image_file = entry["image_file"]
        dest = img_dir / image_file
        if not dest.exists():
            size = download(f"{COCO_VAL2014}/{image_file}", dest)
            print(f"  {image_file}: {size} bytes")
        sample[key] = as_foil_list(entry)

    out_path = root / "foil-it.json"
    out_path.write_text(json.dumps(sample, indent=2))
    print(f"\nWrote {len(sample)} samples -> {out_path}")
    print(f"Images -> {img_dir}")


if __name__ == "__main__":
    main()
