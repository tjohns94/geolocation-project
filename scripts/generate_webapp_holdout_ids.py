"""Regenerate configs/webapp_holdout_ids.json from the committed data artifacts.

The 1,000 image IDs reserved for the human-vs-model webapp benchmark live in
two committed artifacts: ``data/original1000.zip`` (the actual JPEGs) and
``data/experiment_data.json`` (the merged record of model predictions and
human guesses). This script cross-verifies the two sources and emits the
canonical JSON list used by the training notebook to exclude these IDs from
model evaluation.

Run this whenever ``original1000.zip`` or ``experiment_data.json`` changes::

    python scripts/generate_webapp_holdout_ids.py

The output at ``configs/webapp_holdout_ids.json`` is deterministic (sorted
IDs) so diffs are meaningful.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "data" / "original1000.zip"
EXPERIMENT_DATA_PATH = REPO_ROOT / "data" / "experiment_data.json"
OUTPUT_PATH = REPO_ROOT / "configs" / "webapp_holdout_ids.json"


def _ids_from_zip(path: Path) -> set[str]:
    """Return the set of image IDs implied by JPG filenames in the zip."""
    with zipfile.ZipFile(path) as zf:
        return {Path(n).stem for n in zf.namelist() if n.lower().endswith(".jpg")}


def _ids_from_experiment_data(path: Path) -> set[str]:
    """Return the set of image IDs declared in experiment_data.json."""
    payload = json.loads(path.read_text())
    return {img["image_id"] for img in payload["images"]}


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found.", file=sys.stderr)
        return 1
    if not EXPERIMENT_DATA_PATH.exists():
        print(f"ERROR: {EXPERIMENT_DATA_PATH} not found.", file=sys.stderr)
        return 1

    zip_ids = _ids_from_zip(ZIP_PATH)
    exp_ids = _ids_from_experiment_data(EXPERIMENT_DATA_PATH)

    if zip_ids != exp_ids:
        only_zip = sorted(zip_ids - exp_ids)[:5]
        only_exp = sorted(exp_ids - zip_ids)[:5]
        print(
            "ERROR: original1000.zip and experiment_data.json disagree on image IDs.\n"
            f"  In zip not in experiment_data (first 5): {only_zip}\n"
            f"  In experiment_data not in zip (first 5): {only_exp}\n"
            "Resolve the mismatch before regenerating.",
            file=sys.stderr,
        )
        return 1

    ids_sorted = sorted(exp_ids)
    payload = {
        "description": (
            "Image IDs reserved for the human-vs-model webapp benchmark. "
            "Mirrors data/original1000.zip and data/experiment_data.json. "
            "Regenerate via scripts/generate_webapp_holdout_ids.py."
        ),
        "n_images": len(ids_sorted),
        "image_ids": ids_sorted,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH} ({len(ids_sorted)} IDs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
