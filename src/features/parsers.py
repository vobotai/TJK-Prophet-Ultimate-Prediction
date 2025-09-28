"""Utility parsing helpers for TJK Prophet pipelines."""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from typing import Optional

DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]
TIME_FORMATS = ["%H:%M", "%H.%M"]


def parse_date(raw: str) -> Optional[str]:
    """Parse `gg/aa/yyyy` like strings to ISO date."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_time(raw: str) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%H:%M")
        except ValueError:
            continue
    if re.fullmatch(r"\d{2}:\d{2}", text):
        return text
    return None


def normalize_distance(raw: str) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw)
    text = text.replace(".", "").replace(",", "").lower()
    text = re.sub(r"[^0-9]", "", text)
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    if value < 800 or value > 3400:
        return None
    return value


def parse_float(raw: str) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("%", "").replace(" ", "")
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(raw: str) -> Optional[int]:
    value = parse_float(raw)
    if value is None or math.isnan(value):
        return None
    return int(round(value))


def parse_agf(raw: str) -> Optional[float]:
    value = parse_float(raw)
    if value is None or math.isnan(value):
        return None
    if value > 1.5:  # likely expressed in percentage
        value = value / 100.0
    return max(min(value, 1.0), 0.0)


BEST_TIME_PATTERN = re.compile(
    r"^(?:(?:(?P<h>\d+)[:.])?(?P<m>\d+)[:.])?(?P<s>\d+)(?:[.,](?P<ms>\d+))?$"
)


def parse_best_time(raw: str) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    match = BEST_TIME_PATTERN.match(text)
    if not match:
        return None
    h = match.group("h")
    m = match.group("m")
    s = match.group("s")
    ms = match.group("ms")
    total = 0.0
    if h:
        total += int(h) * 3600
    if m:
        total += int(m) * 60
    total += int(s)
    if ms:
        frac = ms
        if len(frac) >= 3:
            total += int(frac[:3]) / 1000.0
        else:
            total += int(frac) / (10 ** len(frac))
    return total


def slugify(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    ascii_text = ascii_text.strip("-")
    return ascii_text or "n-a"


def genealogy_token(raw: str) -> Optional[str]:
    if raw is None:
        return None
    norm = unicodedata.normalize("NFKD", raw)
    ascii_text = norm.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = ascii_text.replace(" ", "_")
    ascii_text = re.sub(r"[^a-z0-9_]+", "", ascii_text)
    return ascii_text or None


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
