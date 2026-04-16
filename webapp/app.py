"""
GeoGuessr-style Human vs. Model comparison webapp.

Flask backend serving:
  - Static images from data/images/
  - JSON manifest of images + pre-computed model predictions
  - SQLite database for logging human guesses
  - Smart image selection (prioritize images other users have seen)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, abort, g, jsonify, render_template, request, send_from_directory

# ── App Setup ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "guesses.db"
MANIFEST_PATH = DATA_DIR / "manifest.json"
IMAGES_DIR = DATA_DIR / "images"


def auto_setup() -> None:
    """Automatically extract images and generate manifest from repo data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Extract images from zip if not already present
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        zip_path = REPO_DIR / "data" / "original1000.zip"
        if zip_path.exists():
            import zipfile
            print(f"Extracting images from {zip_path} ...")
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(IMAGES_DIR)
            print(f"Extracted {len(list(IMAGES_DIR.rglob('*.jpg')))} images")
        else:
            print(f"WARNING: {zip_path} not found. Images will be unavailable.")

    # Generate manifest from experiment_data.json if not already present
    if not MANIFEST_PATH.exists():
        experiment_path = REPO_DIR / "data" / "experiment_data.json"
        if experiment_path.exists():
            print(f"Generating manifest from {experiment_path} ...")
            with open(experiment_path) as f:
                exp = json.load(f)
            manifest = {
                "images": [
                    {
                        "image_id": img["image_id"],
                        "filename": img["filename"],
                        "correct_country": img["correct_country"],
                        "model_prediction": img["efficientnet_b0_prediction"],
                        "model_confidence": img["efficientnet_b0_confidence"],
                    }
                    for img in exp["images"]
                ],
                "countries": exp["countries"],
            }
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f)
            print(f"Manifest created: {len(manifest['images'])} images, {len(manifest['countries'])} countries")
        else:
            print(f"WARNING: {experiment_path} not found. Run from the repo root.")

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600

# ── Security: Rate Limiting ──────────────────────────────────────────────────

RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 60  # max requests per window per IP
RATE_LIMIT_MAX_GUESSES = 20  # max guess submissions per window per IP

_request_log: dict[str, list[float]] = defaultdict(list)
_guess_log: dict[str, list[float]] = defaultdict(list)


def _is_rate_limited(log: dict[str, list[float]], ip: str, limit: int) -> bool:
    """Check if an IP has exceeded the rate limit."""
    now = time.time()
    # Prune old entries
    log[ip] = [t for t in log[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(log[ip]) >= limit:
        return True
    log[ip].append(now)
    return False


# ── Security: Input Validation ───────────────────────────────────────────────

MAX_USERNAME_LENGTH = 30
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\- ]+$")
EXPORT_SECRET = os.environ.get("GEOGUESS_EXPORT_KEY", "")


def sanitize_username(raw: str) -> str | None:
    """Validate and sanitize a username. Returns None if invalid."""
    name = raw.strip()[:MAX_USERNAME_LENGTH]
    if not name or not USERNAME_PATTERN.match(name):
        return None
    return name


@app.before_request
def security_checks() -> Response | None:
    """Apply rate limiting and security headers to every request."""
    ip = request.remote_addr or "unknown"
    if _is_rate_limited(_request_log, ip, RATE_LIMIT_MAX_REQUESTS):
        return jsonify({"error": "Too many requests. Please slow down."}), 429
    return None


@app.after_request
def add_security_headers(response: Response) -> Response:
    """Add security headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com; "
        "img-src 'self' https://*.tile.openstreetmap.org https://cartodb-basemaps-a.global.ssl.fastly.net "
        "https://cartodb-basemaps-b.global.ssl.fastly.net https://cartodb-basemaps-c.global.ssl.fastly.net "
        "https://cartodb-basemaps-d.global.ssl.fastly.net; "
        "connect-src 'self' https://cdn.jsdelivr.net; "
        "font-src 'self';"
    )
    return response

# ── Manifest (loaded once at startup) ────────────────────────────────────────

_manifest: dict = {}  # image_id -> {filename, correct_country, model_prediction, model_confidence}
_country_list: list[str] = []  # sorted unique country names from the dataset


def load_manifest() -> None:
    """Load the image manifest and country list from disk."""
    global _manifest, _country_list
    if not MANIFEST_PATH.exists():
        print(f"WARNING: {MANIFEST_PATH} not found. Run prepare_data.py first.")
        return
    with open(MANIFEST_PATH) as f:
        raw = json.load(f)
    _manifest = {img["image_id"]: img for img in raw["images"]}
    _country_list = sorted(raw.get("countries", []))
    print(f"Loaded manifest: {len(_manifest)} images, {len(_country_list)} countries")


# ── Database ─────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS guesses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    image_id TEXT NOT NULL,
    guessed_country TEXT NOT NULL,
    correct_country TEXT NOT NULL,
    is_correct INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    sequence_num INTEGER NOT NULL,
    UNIQUE(username, image_id)
);

CREATE TABLE IF NOT EXISTS model_guesses (
    image_id TEXT PRIMARY KEY,
    predicted_country TEXT NOT NULL,
    correct_country TEXT NOT NULL,
    confidence REAL,
    is_correct INTEGER NOT NULL
);
"""


def get_db() -> sqlite3.Connection:
    """Get a database connection (one per request)."""
    if "db" not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
        g.db.executescript(SCHEMA)
    return g.db


@app.teardown_appcontext
def close_db(exception: BaseException | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def ensure_model_guesses_populated() -> None:
    """Populate model_guesses table from manifest (idempotent)."""
    db = get_db()
    existing = db.execute("SELECT COUNT(*) FROM model_guesses").fetchone()[0]
    if existing >= len(_manifest):
        return
    for img in _manifest.values():
        db.execute(
            """INSERT OR IGNORE INTO model_guesses
               (image_id, predicted_country, correct_country, confidence, is_correct)
               VALUES (?, ?, ?, ?, ?)""",
            (
                img["image_id"],
                img["model_prediction"],
                img["correct_country"],
                img.get("model_confidence", 0.0),
                1 if img["model_prediction"] == img["correct_country"] else 0,
            ),
        )
    db.commit()


# ── Routes: Pages ────────────────────────────────────────────────────────────

@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/images/<path:filename>")
def serve_image(filename: str) -> Response:
    return send_from_directory(str(IMAGES_DIR), filename)


# ── Routes: API ──────────────────────────────────────────────────────────────

@app.route("/api/countries")
def api_countries() -> Response:
    """Return the list of valid country names."""
    return jsonify({"countries": _country_list})


@app.route("/api/next-image")
def api_next_image() -> Response:
    """Return the next image for a user.

    Priority:
      1. Images that OTHER users have guessed but this user hasn't
      2. Random image this user hasn't seen
    Returns 204 if user has completed all images.
    """
    username = sanitize_username(request.args.get("username", ""))
    if not username:
        return jsonify({"error": "Invalid username. Use letters, numbers, spaces, hyphens, or underscores (max 30 chars)."}), 400

    db = get_db()
    ensure_model_guesses_populated()

    # Priority 1: images done by others but not by this user
    row = db.execute(
        """SELECT DISTINCT g.image_id
           FROM guesses g
           WHERE g.username != ?
             AND g.image_id NOT IN (
                 SELECT image_id FROM guesses WHERE username = ?
             )
           ORDER BY RANDOM()
           LIMIT 1""",
        (username, username),
    ).fetchone()

    if row is None:
        # Priority 2: any image not yet done by this user
        done_ids = {
            r[0]
            for r in db.execute(
                "SELECT image_id FROM guesses WHERE username = ?", (username,)
            ).fetchall()
        }
        remaining = [iid for iid in _manifest if iid not in done_ids]
        if not remaining:
            return jsonify({"done": True}), 204

        import random
        image_id = random.choice(remaining)
    else:
        image_id = row[0]

    img = _manifest[image_id]

    # Count how many this user has done
    user_count = db.execute(
        "SELECT COUNT(*) FROM guesses WHERE username = ?", (username,)
    ).fetchone()[0]

    return jsonify(
        {
            "image_id": image_id,
            "image_url": f"images/{img['filename']}",
            "total_images": len(_manifest),
            "user_completed": user_count,
        }
    )


@app.route("/api/submit-guess", methods=["POST"])
def api_submit_guess() -> Response:
    """Submit a human guess and return the result + model's guess."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Rate limit guess submissions more strictly
    ip = request.remote_addr or "unknown"
    if _is_rate_limited(_guess_log, ip, RATE_LIMIT_MAX_GUESSES):
        return jsonify({"error": "Too many guesses. Please slow down."}), 429

    username = sanitize_username(data.get("username", ""))
    image_id = data.get("image_id", "").strip()[:20]
    guessed_country = data.get("country", "").strip()[:100]

    if not all([username, image_id, guessed_country]):
        return jsonify({"error": "username, image_id, and country required"}), 400

    if guessed_country not in _country_list:
        return jsonify({"error": "Invalid country name"}), 400

    if image_id not in _manifest:
        return jsonify({"error": "unknown image_id"}), 404

    img = _manifest[image_id]
    correct_country = img["correct_country"]
    is_correct = 1 if guessed_country == correct_country else 0

    db = get_db()

    # Get user's sequence number
    seq = db.execute(
        "SELECT COUNT(*) FROM guesses WHERE username = ?", (username,)
    ).fetchone()[0] + 1

    # Insert human guess (ignore if duplicate)
    try:
        db.execute(
            """INSERT INTO guesses
               (username, image_id, guessed_country, correct_country, is_correct, timestamp, sequence_num)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                username,
                image_id,
                guessed_country,
                correct_country,
                is_correct,
                datetime.now(timezone.utc).isoformat(),
                seq,
            ),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "already guessed this image"}), 409

    # Get model's prediction
    model_pred = img["model_prediction"]
    model_correct = 1 if model_pred == correct_country else 0

    # Get user's running stats
    stats = db.execute(
        """SELECT COUNT(*) as total,
                  SUM(is_correct) as correct
           FROM guesses WHERE username = ?""",
        (username,),
    ).fetchone()

    return jsonify(
        {
            "correct_country": correct_country,
            "human_guess": guessed_country,
            "human_correct": bool(is_correct),
            "model_guess": model_pred,
            "model_correct": bool(model_correct),
            "user_stats": {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / stats["total"] * 100, 1)
                if stats["total"]
                else 0,
            },
        }
    )


@app.route("/api/stats")
def api_stats() -> Response:
    """Return aggregate stats for all users and the model."""
    db = get_db()

    # Per-user stats
    user_stats = db.execute(
        """SELECT username, COUNT(*) as total, SUM(is_correct) as correct
           FROM guesses
           GROUP BY username
           ORDER BY correct DESC"""
    ).fetchall()

    # Model stats (on images that have been guessed by at least one human)
    model_stats = db.execute(
        """SELECT COUNT(*) as total, SUM(mg.is_correct) as correct
           FROM model_guesses mg
           WHERE mg.image_id IN (SELECT DISTINCT image_id FROM guesses)"""
    ).fetchone()

    return jsonify(
        {
            "users": [
                {
                    "username": r["username"],
                    "total": r["total"],
                    "correct": r["correct"],
                    "accuracy": round(r["correct"] / r["total"] * 100, 1)
                    if r["total"]
                    else 0,
                }
                for r in user_stats
            ],
            "model": {
                "total": model_stats["total"] if model_stats["total"] else 0,
                "correct": model_stats["correct"] if model_stats["correct"] else 0,
                "accuracy": round(
                    model_stats["correct"] / model_stats["total"] * 100, 1
                )
                if model_stats["total"]
                else 0,
            },
        }
    )


@app.route("/api/export")
def api_export() -> Response:
    """Export all data as JSON for analysis. Protected by secret key."""
    key = request.args.get("key", "")
    if not EXPORT_SECRET or key != EXPORT_SECRET:
        return jsonify({"error": "Unauthorized. Provide ?key=YOUR_SECRET"}), 403
    db = get_db()
    guesses = db.execute("SELECT * FROM guesses ORDER BY timestamp").fetchall()
    model = db.execute("SELECT * FROM model_guesses").fetchall()

    return jsonify(
        {
            "human_guesses": [dict(r) for r in guesses],
            "model_predictions": [dict(r) for r in model],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    auto_setup()
    load_manifest()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    auto_setup()
    load_manifest()
