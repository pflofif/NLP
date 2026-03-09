from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_RES = _ROOT / "resources"


def _load_json(name: str) -> dict:
    with open(_RES / name, encoding="utf-8") as f:
        return json.load(f)


_CURRENCIES: dict = _load_json("currencies.json")
_MONTHS: dict = _load_json("months_ua.json")
_WEEKDAYS: dict = _load_json("weekdays_ua.json")
_LOCATIONS: dict = _load_json("locations_lviv.json")

_CURRENCY_PATTERN = (
    r"(?:"
    + r"|".join(
        re.escape(k)
        for k in sorted(_CURRENCIES["normalize"].keys(), key=len, reverse=True)
    )
    + r")"
)
_AMOUNT_RE = re.compile(
    r"(\d[\d\s\u00a0,.]*\d|\d)\s*(" + _CURRENCY_PATTERN + r")(?=[^\w]|$)",
    re.IGNORECASE | re.UNICODE,
)

_AMOUNT_ANTI_RE = re.compile(r"\d\s*%|\d\s*/\s*\d|\d\s+з\s+\d", re.IGNORECASE)


def _normalize_value(raw_num: str) -> float:
    cleaned = re.sub(r"[\s\u00a0]", "", raw_num)
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def extract_amounts(text: str) -> list[dict[str, Any]]:
    results = []
    for m in _AMOUNT_RE.finditer(text):
        raw = m.group(0).strip()
        num_part = m.group(1)
        cur_token = m.group(2).strip()

        context_before = text[max(0, m.start() - 5) : m.start()]
        if _AMOUNT_ANTI_RE.search(context_before + raw):
            continue

        currency = _CURRENCIES["normalize"].get(cur_token, "UNKNOWN")
        currency = _CURRENCIES["normalize"].get(cur_token.lower(), currency)

        results.append(
            {
                "field_type": "AMOUNT",
                "value": _normalize_value(num_part),
                "currency": currency,
                "raw": raw,
                "start_char": m.start(),
                "end_char": m.start() + len(raw),
                "method": "regex_amount_v1",
            }
        )
    return results

_DATE_ABS_RE = re.compile(
    r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b"
)

_MONTH_WORDS = "|".join(
    re.escape(k)
    for k in sorted(
        list(_MONTHS["months"].keys()) + list(_MONTHS["abbreviations"].keys()),
        key=len,
        reverse=True,
    )
)
_DATE_MONTH_RE = re.compile(
    r"\b(?:(\d{1,2})\s+)?(" + _MONTH_WORDS + r")(?:\s+(\d{4}))?\b",
    re.IGNORECASE | re.UNICODE,
)

_WEEKDAY_WORDS = "|".join(
    re.escape(k)
    for k in sorted(_WEEKDAYS["weekdays"].keys(), key=len, reverse=True)
)
_RELATIVE_WORDS = "|".join(
    re.escape(k)
    for k in sorted(_WEEKDAYS["relative"].keys(), key=len, reverse=True)
)
_DATE_WEEKDAY_RE = re.compile(
    r"\b(" + _WEEKDAY_WORDS + r")\b", re.IGNORECASE | re.UNICODE
)
_DATE_RELATIVE_RE = re.compile(
    r"\b(" + _RELATIVE_WORDS + r")\b", re.IGNORECASE | re.UNICODE
)

_DATE_ANTI_DURATION_RE = re.compile(
    r"\d+\s*(?:хвилин|годин|секунд|місяц|тижн|день|дні|днів)\b",
    re.IGNORECASE | re.UNICODE,
)


def _normalize_date(day: str | None, month: int, year: str | None) -> str | None:
    if year and day:
        y = int(year) if len(year) == 4 else 2000 + int(year)
        return f"{y:04d}-{month:02d}-{int(day):02d}"
    if year:
        y = int(year) if len(year) == 4 else 2000 + int(year)
        return f"{y:04d}-{month:02d}"
    return None


def extract_dates(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    used: set[int] = set()  

    for m in _DATE_ABS_RE.finditer(text):
        raw = m.group(0)
        d, mo, y = m.group(1), m.group(2), m.group(3)
        y_int = int(y) if len(y) == 4 else 2000 + int(y)
        normalized = f"{y_int:04d}-{int(mo):02d}-{int(d):02d}"
        results.append(
            {
                "field_type": "DATE",
                "subtype": "DATE_ABS",
                "value": normalized,
                "raw": raw,
                "normalized": normalized,
                "start_char": m.start(),
                "end_char": m.end(),
                "method": "regex_date_abs",
            }
        )
        used.update(range(m.start(), m.end()))

    for m in _DATE_MONTH_RE.finditer(text):
        if any(i in used for i in range(m.start(), m.end())):
            continue
        context = text[max(0, m.start() - 10) : m.end() + 10]
        if _DATE_ANTI_DURATION_RE.search(context):
            continue
        raw = m.group(0)
        day_str = m.group(1)
        month_word = m.group(2).lower()
        year_str = m.group(3)
        month_num = _MONTHS["months"].get(month_word) or _MONTHS["abbreviations"].get(month_word)
        if month_num is None:
            continue
        normalized = _normalize_date(day_str, month_num, year_str)
        results.append(
            {
                "field_type": "DATE",
                "subtype": "DATE_MONTH",
                "value": month_word,
                "raw": raw,
                "normalized": normalized,
                "start_char": m.start(),
                "end_char": m.end(),
                "method": "dict_month_ua",
            }
        )
        used.update(range(m.start(), m.end()))

    for m in _DATE_WEEKDAY_RE.finditer(text):
        if any(i in used for i in range(m.start(), m.end())):
            continue
        raw = m.group(1)
        value = _WEEKDAYS["weekdays"].get(raw.lower(), raw.lower())
        results.append(
            {
                "field_type": "DATE",
                "subtype": "DATE_WEEKDAY",
                "value": value,
                "raw": raw,
                "normalized": None,
                "start_char": m.start(),
                "end_char": m.end(),
                "method": "dict_weekday_ua",
            }
        )
        used.update(range(m.start(), m.end()))

    for m in _DATE_RELATIVE_RE.finditer(text):
        if any(i in used for i in range(m.start(), m.end())):
            continue
        raw = m.group(1)
        value = _WEEKDAYS["relative"].get(raw.lower(), raw.lower())
        results.append(
            {
                "field_type": "DATE",
                "subtype": "DATE_RELATIVE",
                "value": value,
                "raw": raw,
                "normalized": None,
                "start_char": m.start(),
                "end_char": m.end(),
                "method": "dict_relative_ua",
            }
        )
        used.update(range(m.start(), m.end()))

    results.sort(key=lambda x: x["start_char"])
    return results


_LOC_ENTRIES: list[tuple[str, str]] = []

for loc in _LOCATIONS["city_variants"]:
    canon = "Львів"
    _LOC_ENTRIES.append((loc, canon))

for loc in _LOCATIONS["landmarks"]:
    canon = _LOCATIONS["all_normalized"].get(loc.lower(), loc)
    _LOC_ENTRIES.append((loc, canon))

for loc in _LOCATIONS["squares_and_streets"]:
    canon = _LOCATIONS["all_normalized"].get(loc.lower(), loc)
    _LOC_ENTRIES.append((loc, canon))

for loc in _LOCATIONS["districts"]:
    canon = _LOCATIONS["all_normalized"].get(loc.lower(), loc)
    _LOC_ENTRIES.append((loc, canon))

_LOC_ENTRIES.sort(key=lambda x: len(x[0]), reverse=True)

_LOC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<!\w)" + re.escape(raw) + r"(?!\w)", re.IGNORECASE | re.UNICODE), canon)
    for raw, canon in _LOC_ENTRIES
]


def extract_locations(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    used: set[int] = set()

    for pattern, canon in _LOC_PATTERNS:
        for m in pattern.finditer(text):
            if any(i in used for i in range(m.start(), m.end())):
                continue
            raw = m.group(0)
            results.append(
                {
                    "field_type": "LOCATION",
                    "value": canon,
                    "raw": raw,
                    "start_char": m.start(),
                    "end_char": m.end(),
                    "method": "dict_lviv_landmarks",
                }
            )
            used.update(range(m.start(), m.end()))

    results.sort(key=lambda x: x["start_char"])
    return results

def extract_all(text: str) -> dict[str, list[dict[str, Any]]]:
    """
    Run all extractors on text. Returns dict with keys: AMOUNT, DATE, LOCATION.
    """
    return {
        "AMOUNT": extract_amounts(text),
        "DATE": extract_dates(text),
        "LOCATION": extract_locations(text),
    }

if __name__ == "__main__":
    samples = [
        "Замовляла каву флет вайт. 85 грн.не вартує. Більше не замовлю.",
        "Були в суботу ввечері на площі Ринок, у Львові.",
        "Сьогодні зайшли в заклад на вулиці Дорошенка, заплатили 240 грн.",
        "Таких пельменів за 300 грн я не їв. Знайшов на проспекті Свободи.",
        "100% рекомендую! Чекали 30 хвилин, але воно того варте.",
    ]
    for s in samples:
        print("TEXT:", s)
        r = extract_all(s)
        for field, items in r.items():
            if items:
                for item in items:
                    print(f"  [{field}] {item['raw']!r} -> {item.get('value')} {item.get('currency', '')}")
        print()
