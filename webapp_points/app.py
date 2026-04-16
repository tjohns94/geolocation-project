"""
GeoGuessr 5000-point scoring webapp — Human vs. Model comparison.

Flask backend serving:
  - Static images from data/images/
  - JSON manifest of images + pre-computed model predictions
  - SQLite database for logging human guesses with coordinates and scores
  - Smart image selection (prioritize images other users have seen)
  - GeoGuessr 5000-point scoring: score = 5000 * exp(-distance_km / 1492.7)
"""

from __future__ import annotations

import json
import math
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
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "guesses.db"
MANIFEST_PATH = DATA_DIR / "manifest.json"
IMAGES_DIR = DATA_DIR / "images"

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600

# ── Constants ────────────────────────────────────────────────────────────────

# GeoGuessr scoring: score = 5000 * exp(-distance_km / 1492.7)
# At 0 km: 5000 points
# At ~1035 km: 2500 points
# At ~2400 km: 1000 points
MAX_SCORE = 5000
DISTANCE_SCALING_FACTOR = 1492.7  # empirically derived constant

# Earth radius in km (for haversine)
EARTH_RADIUS_KM = 6371.0

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


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate latitude and longitude are in valid ranges."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
        return -90 <= lat_f <= 90 and -180 <= lon_f <= 180
    except (ValueError, TypeError):
        return False


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

_manifest: dict = {}  # image_id -> {filename, correct_country, model_prediction, model_confidence, lat, lon}
_country_centroids: dict[str, tuple[float, float]] = {}  # country_code -> (lat, lon)


def load_manifest() -> None:
    """Load the image manifest and compute country centroids from disk."""
    global _manifest, _country_centroids
    if not MANIFEST_PATH.exists():
        print(f"WARNING: {MANIFEST_PATH} not found. Run prepare_data.py first.")
        return
    with open(MANIFEST_PATH) as f:
        raw = json.load(f)
    _manifest = {img["image_id"]: img for img in raw["images"]}

    # Compute country centroids by averaging lat/lon per country
    country_coords: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for img in _manifest.values():
        country = img.get("correct_country")
        if country and "lat" in img and "lon" in img:
            country_coords[country].append((img["lat"], img["lon"]))

    for country, coords in country_coords.items():
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lon = sum(c[1] for c in coords) / len(coords)
        _country_centroids[country] = (avg_lat, avg_lon)

    print(f"Loaded manifest: {len(_manifest)} images, {len(_country_centroids)} country centroids")


# ── Scoring Functions ────────────────────────────────────────────────────────

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance between two points in kilometers.

    Args:
        lat1, lon1: First point (latitude, longitude in degrees)
        lat2, lon2: Second point (latitude, longitude in degrees)

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c


def compute_score(distance_km: float) -> int:
    """Compute GeoGuessr 5000-point score from distance.

    Formula: score = 5000 * exp(-distance_km / 1492.7)

    Args:
        distance_km: Distance from guess to true location in kilometers

    Returns:
        Score as integer (0-5000)
    """
    if distance_km < 0:
        distance_km = 0
    score = MAX_SCORE * math.exp(-distance_km / DISTANCE_SCALING_FACTOR)
    return max(0, int(round(score)))


# ── Database ─────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS guesses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    image_id TEXT NOT NULL,
    guess_lat REAL NOT NULL,
    guess_lon REAL NOT NULL,
    true_lat REAL NOT NULL,
    true_lon REAL NOT NULL,
    distance_km REAL NOT NULL,
    score INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    sequence_num INTEGER NOT NULL,
    UNIQUE(username, image_id)
);

CREATE TABLE IF NOT EXISTS model_scores (
    image_id TEXT PRIMARY KEY,
    predicted_country TEXT NOT NULL,
    correct_country TEXT NOT NULL,
    model_lat REAL NOT NULL,
    model_lon REAL NOT NULL,
    true_lat REAL NOT NULL,
    true_lon REAL NOT NULL,
    distance_km REAL NOT NULL,
    score INTEGER NOT NULL
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


def ensure_model_scores_populated() -> None:
    """Populate model_scores table from manifest (idempotent)."""
    db = get_db()
    existing = db.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0]
    if existing >= len(_manifest):
        return

    for img in _manifest.values():
        predicted_country = img.get("model_prediction")
        correct_country = img.get("correct_country")

        # Get model's location (centroid of predicted country)
        if predicted_country in _country_centroids:
            model_lat, model_lon = _country_centroids[predicted_country]
        else:
            # Fallback: use true location if country not found
            model_lat = img.get("lat", 0)
            model_lon = img.get("lon", 0)

        true_lat = img.get("lat", 0)
        true_lon = img.get("lon", 0)

        distance_km = haversine_distance(model_lat, model_lon, true_lat, true_lon)
        score = compute_score(distance_km)

        db.execute(
            """INSERT OR IGNORE INTO model_scores
               (image_id, predicted_country, correct_country, model_lat, model_lon, true_lat, true_lon, distance_km, score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                img["image_id"],
                predicted_country,
                correct_country,
                model_lat,
                model_lon,
                true_lat,
                true_lon,
                distance_km,
                score,
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

@app.route("/api/country-centroids")
def api_country_centroids() -> Response:
    """Return the centroid coordinates for all countries."""
    return jsonify(_country_centroids)


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
    ensure_model_scores_populated()

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
    """Submit a human guess (lat/lon) and return the result + model's score.

    Request body:
        {
            "username": str,
            "image_id": str,
            "lat": float,
            "lon": float
        }

    Returns:
        {
            "score": int,
            "distance_km": float,
            "true_lat": float,
            "true_lon": float,
            "model_score": int,
            "model_distance_km": float,
            "model_lat": float,
            "model_lon": float,
            "correct_country": str,
            "stats": {
                "total": int,
                "mean_score": float,
                "sum_score": int
            }
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Rate limit guess submissions more strictly
    ip = request.remote_addr or "unknown"
    if _is_rate_limited(_guess_log, ip, RATE_LIMIT_MAX_GUESSES):
        return jsonify({"error": "Too many guesses. Please slow down."}), 429

    username = sanitize_username(data.get("username", ""))
    image_id = data.get("image_id", "").strip()[:20]
    guess_lat = data.get("lat")
    guess_lon = data.get("lon")

    if not all([username, image_id, guess_lat is not None, guess_lon is not None]):
        return jsonify({"error": "username, image_id, lat, and lon required"}), 400

    if not validate_coordinates(guess_lat, guess_lon):
        return jsonify({"error": "Invalid coordinates. lat must be in [-90, 90], lon in [-180, 180]"}), 400

    if image_id not in _manifest:
        return jsonify({"error": "unknown image_id"}), 404

    img = _manifest[image_id]
    true_lat = img.get("lat", 0)
    true_lon = img.get("lon", 0)
    correct_country = img.get("correct_country")

    # Compute human score
    distance_km = haversine_distance(float(guess_lat), float(guess_lon), true_lat, true_lon)
    score = compute_score(distance_km)

    db = get_db()

    # Get user's sequence number
    seq = db.execute(
        "SELECT COUNT(*) FROM guesses WHERE username = ?", (username,)
    ).fetchone()[0] + 1

    # Insert human guess (ignore if duplicate)
    try:
        db.execute(
            """INSERT INTO guesses
               (username, image_id, guess_lat, guess_lon, true_lat, true_lon, distance_km, score, timestamp, sequence_num)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                username,
                image_id,
                guess_lat,
                guess_lon,
                true_lat,
                true_lon,
                distance_km,
                score,
                datetime.now(timezone.utc).isoformat(),
                seq,
            ),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "already guessed this image"}), 409

    # Get model's scores for this image
    model_row = db.execute(
        "SELECT model_lat, model_lon, distance_km, score FROM model_scores WHERE image_id = ?",
        (image_id,)
    ).fetchone()

    if model_row:
        model_lat = model_row["model_lat"]
        model_lon = model_row["model_lon"]
        model_distance_km = model_row["distance_km"]
        model_score = model_row["score"]
    else:
        # Fallback (shouldn't happen if ensure_model_scores_populated ran)
        model_lat = true_lat
        model_lon = true_lon
        model_distance_km = 0
        model_score = MAX_SCORE

    # Get user's running stats
    stats_row = db.execute(
        """SELECT COUNT(*) as total, SUM(score) as sum_score
           FROM guesses WHERE username = ?""",
        (username,),
    ).fetchone()

    total = stats_row["total"]
    sum_score = stats_row["sum_score"] or 0
    mean_score = sum_score / total if total > 0 else 0

    return jsonify(
        {
            "score": score,
            "distance_km": round(distance_km, 2),
            "true_lat": true_lat,
            "true_lon": true_lon,
            "model_score": model_score,
            "model_distance_km": round(model_distance_km, 2),
            "model_lat": model_lat,
            "model_lon": model_lon,
            "correct_country": correct_country,
            "stats": {
                "total": total,
                "sum_score": sum_score,
                "mean_score": round(mean_score, 1),
            },
        }
    )


@app.route("/api/stats")
def api_stats() -> Response:
    """Return aggregate stats for all users and the model.

    Returns scores instead of accuracy percentages.
    """
    db = get_db()

    # Per-user stats
    user_stats = db.execute(
        """SELECT username, COUNT(*) as total, SUM(score) as sum_score, AVG(score) as mean_score
           FROM guesses
           GROUP BY username
           ORDER BY sum_score DESC"""
    ).fetchall()

    # Model stats (on images that have been guessed by at least one human)
    model_stats = db.execute(
        """SELECT COUNT(*) as total, SUM(ms.score) as sum_score, AVG(ms.score) as mean_score
           FROM model_scores ms
           WHERE ms.image_id IN (SELECT DISTINCT image_id FROM guesses)"""
    ).fetchone()

    return jsonify(
        {
            "users": [
                {
                    "username": r["username"],
                    "total": r["total"],
                    "sum_score": r["sum_score"] or 0,
                    "mean_score": round(r["mean_score"], 1) if r["mean_score"] else 0,
                }
                for r in user_stats
            ],
            "model": {
                "total": model_stats["total"] if model_stats["total"] else 0,
                "sum_score": model_stats["sum_score"] if model_stats["sum_score"] else 0,
                "mean_score": round(model_stats["mean_score"], 1) if model_stats["mean_score"] else 0,
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
    model = db.execute("SELECT * FROM model_scores").fetchall()

    return jsonify(
        {
            "human_guesses": [dict(r) for r in guesses],
            "model_predictions": [dict(r) for r in model],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_manifest()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    load_manifest()
