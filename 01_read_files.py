#!/usr/bin/env python3
# Reads the EX-19 policies, asks the model when the quarterly trading window
# closes, joins the metadata csv, dumps blackout_summary.csv.
#
#   pip install openai python-dotenv beautifulsoup4
#   OPENAI_API_KEY goes in a .env next to this file
#   python 01_read_files.py

import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from openai import OpenAI # pyright: ignore[reportMissingImports]

EX19_DIR = Path("ex19_policies")
METADATA_FILE = EX19_DIR / "ex19_metadata.csv"
OUTPUT_CSV = "blackout_summary.csv"

MODEL_NAME = "gpt-5.4-nano"  # stay with this model
MAX_CONTEXT_CHARS = 30000
MAX_RETRIES = 2

# only run part of the folder (testing / re-runs after a crash)
TEST_MODE = False
TEST_FILE_LIMIT = 3
RANGE_MODE = False
RANGE_START = 90
RANGE_END = 95

META_FIELDS = ["cik", "ticker", "company_name", "sector", "market_value",
               "accession", "filing_date", "url"]


def extract_text(path):
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            raw = path.read_bytes().decode("latin-1", errors="ignore")
        except Exception:
            raw = path.read_bytes().decode("cp1252", errors="ignore")

    low = raw.lower()
    if "<html" not in low and "<body" not in low:
        return raw

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def lightly_filter(text):
    # whole exhibits blow past the context limit, so score paragraphs on
    # window/blackout language and keep the top ones
    paragraphs = re.split(r"\n\s*\n", re.sub(r"\r\n?", "\n", text))

    scored = []
    for p in paragraphs:
        lower = p.lower()
        score = 0
        if "ending" in lower and ("week" in lower or "day" in lower):
            score += 10
        if "window period" in lower or "trading window" in lower:
            score += 15
        if "period" in lower:
            score += 10
        if "prior to" in lower and "quarter" in lower:
            score += 5
        if "blackout" in lower or "black-out" in lower  or "black out" in lower:
            score += 10
        if "trading days" in lower:
            score += 5
        if "financial" in lower and "release" in lower:
            score += 5

        if score:
            scored.append((score, p))

    scored.sort(reverse=True, key=lambda x: x[0])
    joined = "\n\n".join(p for _, p in scored[:50])

    if len(joined) > 200:
        return joined[:MAX_CONTEXT_CHARS]
    return text[:MAX_CONTEXT_CHARS]


def load_metadata():
    if not METADATA_FILE.exists():
        logging.warning(f"no metadata at {METADATA_FILE}, continuing without it")
        return {}

    meta = {}
    with METADATA_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row.get("filename", "").strip()
            if not fname:
                continue
            meta[fname] = {k: row.get(k, "").strip() for k in META_FIELDS}

    logging.info(f"loaded metadata for {len(meta)} exhibits")
    return meta


def create_client():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logging.error(f"OPENAI_API_KEY not found in {env_path}")
        sys.exit(1)

    return OpenAI(api_key=key)


SYSTEM_PROMPT = """
Analyze insider trading policies to find when trading windows close before quarter end.

Output JSON:
{
  "has_recurring_blackout": true/false,
  "has_ad_hoc_blackout": true/false,
  "requires_preclearance": true/false,
  "preclearance_description": string,
  "prohibits_hedging": true/false,
  "hedging_description": string,
  "general_description": string,
  "groups": [{"name": string, "blackout_pattern": {"description": string, "blackout_start_days_before_quarter_end": integer|null}}]
}

CRITICAL: Trading windows have an END time. When window ENDS = blackout BEGINS. 
If policy mentions window ending or opening /beginning → has_recurring_blackout = TRUE.
"pre-announced each quarter" / "announced quarterly" (timing not specified) → has_recurring_blackout=true, days=null

PHRASES INDICATING WINDOW CLOSING:
- "window ending/generally ending X before/prior to [quarter end/end of quarter]"
- "open window ending X before quarter end"
- "closes X before quarter end"
- "closes on [day] of [month] of quarter"
- "closes first Friday of last month"


MONTH-BASED TIMING (assume 90-day quarter with 30-day months):
- "closes on last day of second month" → Day 60 of 90 = 30 days before quarter end
- "closes at end of second month" → 30 days before quarter end
- "closes on [any day] of second month" → calculate: 90 - (30 + day_number) = days before quarter end
- "closes first day/Friday of last month" → ~23 days before quarter end
- "closes mid last month" → ~15 days before quarter end
- "closes on last day of fiscal quarter" → 0 days
- "Blackout periods begin on the first day of the third month of a fiscal quarter." → 30 days

CONVERSIONS:
- "two weeks" = 14, "one week" = 7
- "prior to" = "before"
- Second month ends = day 60 of 90-day quarter = 30 days before quarter end
- Last day of quarter = 0 days

AD-HOC BLACKOUTS - Flag has_ad_hoc_blackout=true if mentions ANY of or similar phrases:
- "event-specific blackout"
- "event-driven blackout"
- "special blackout"
- "discretionary blackout"
- "material events/information"
- "special circumstances"
- "mergers" / "M&A" / "acquisitions"
- "at the discretion of"
- "company may impose additional"

CALENDAR QUARTER EXCEPTION:
- If policy explicitly says "calendar quarter" (not just "quarter" or "fiscal quarter")
- → has_recurring_blackout = true
- → blackout_start_days_before_quarter_end = null
- → description should mention "based on calendar quarter"

EXAMPLES:
"window ending two weeks prior to quarter end" → 14
"window closes at end of second month of quarter" → 30
"closes on last day of second month" → 30
"prohibited transactions fifteen (15) trading days before financial release" → 15
"window closes at end of quarter" → 0
"event-specific blackouts may be imposed" → has_ad_hoc_blackout: true
"Quarterly Trading Windows" -> has_recurring_blackout: true

NULL only if: Zero timing relative to quarter end.

ADDITIONAL FEATURES (pre-clearance and hedging):

requires_preclearance:
- TRUE only if the policy explicitly requires designated insiders (directors, officers, or other listed persons) to obtain APPROVAL/AUTHORIZATION from a specified person (General Counsel, Compliance Officer, Legal Department, etc.) BEFORE executing any trade in company securities.
- FALSE if the policy only encourages consultation, requires notification after trading, or is silent on pre-clearance.
- preclearance_description: short sentence describing who must obtain approval and from whom.

prohibits_hedging:
- TRUE only if the policy explicitly prohibits hedging or monetizing instruments such as: prepaid variable forwards, equity swaps, collars, exchange funds, puts, calls, short sales, or similar derivatives designed to offset or limit decreases in company stock value.
- FALSE if the policy only discourages hedging, requires disclosure of hedging, or is silent.
- FALSE if it only prohibits short-term/speculative trading without specifically mentioning hedging instruments.
- hedging_description: short sentence describing the scope of the prohibition (which instruments, which groups).

Be CONSERVATIVE on both: when language is ambiguous, vague, or only implied, mark FALSE. We want to capture only explicit requirements/prohibitions.

Output ONLY valid JSON.
"""


def blank_result(note=""):
    return {
        "has_recurring_blackout": False,
        "has_ad_hoc_blackout": False,
        "requires_preclearance": False,
        "preclearance_description": "",
        "prohibits_hedging": False,
        "hedging_description": "",
        "general_description": note,
        "groups": [],
    }


def clean_days(data, filename):
    # model occasionally answers "14 days" or something out of range
    for group in data.get("groups", []):
        pattern = group.get("blackout_pattern", {})
        days = pattern.get("blackout_start_days_before_quarter_end")
        if days is None:
            continue
        try:
            days = int(days)
        except (ValueError, TypeError):
            logging.warning(f"{filename}: days value not an integer, dropping it")
            days = None
        if days is not None and not 0 <= days <= 90:
            logging.warning(f"{filename}: days out of range ({days}), dropping it")
            days = None
        pattern["blackout_start_days_before_quarter_end"] = days
    return data


def analyze_policy(client, filename, text):
    user_prompt = (
        f"Policy file: {filename}\n\n"
        f"POLICY TEXT:\n{text}\n\n"
        "Return JSON only. Determine exact blackout_start_days_before_quarter_end (no rounding)."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content.strip())
            for k, v in blank_result().items():
                data.setdefault(k, v)
            data = clean_days(data, filename)
        except Exception as e:
            logging.warning(f"{filename}: attempt {attempt}/{MAX_RETRIES} failed ({e})")
            continue

        return data

    logging.error(f"giving up on {filename}")
    return blank_result("Analysis failed after multiple attempts.")


def oneline(s):
    return str(s or "").replace("\n", " ")


FIELDNAMES = ["filename", "accession", "cik", "ticker", "company_name", "sector",
              "market_value", "filing_date", "has_recurring_blackout",
              "has_ad_hoc_blackout", "requires_preclearance", "preclearance_description",
              "prohibits_hedging", "hedging_description", "general_description",
              "group_name", "blackout_description",
              "blackout_start_days_before_quarter_end", "url", "groups_raw_json"]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not EX19_DIR.exists():
        logging.error(f"folder not found: {EX19_DIR.resolve()}")
        sys.exit(1)

    all_files = sorted(p for p in EX19_DIR.iterdir()
                       if p.is_file() and p.name != METADATA_FILE.name)

    if RANGE_MODE:
        files = all_files[RANGE_START:RANGE_END]
    elif TEST_MODE:
        files = all_files[:TEST_FILE_LIMIT]
    else:
        files = all_files

    logging.info(f"processing {len(files)} of {len(all_files)} files in {EX19_DIR.resolve()}")
    if not files:
        return

    metadata = load_metadata()
    client = create_client()
    rows = []

    for i, f in enumerate(files, 1):
        logging.info(f"[{i}/{len(files)}] {f.name}")
        result = analyze_policy(client, f.name, lightly_filter(extract_text(f)))

        groups = result.get("groups") or []
        primary = groups[0] if groups else {}  # first group is the broadest one
        pattern = primary.get("blackout_pattern") or {}
        days = pattern.get("blackout_start_days_before_quarter_end")

        meta = metadata.get(f.name, {})
        row = {"filename": f.name}
        row.update({k: meta.get(k, "") for k in META_FIELDS})
        row.update({
            "has_recurring_blackout": result["has_recurring_blackout"],
            "has_ad_hoc_blackout": result["has_ad_hoc_blackout"],
            "requires_preclearance": result["requires_preclearance"],
            "preclearance_description": oneline(result["preclearance_description"]),
            "prohibits_hedging": result["prohibits_hedging"],
            "hedging_description": oneline(result["hedging_description"]),
            "general_description": oneline(result["general_description"]),
            "group_name": str(primary.get("name", "")),
            "blackout_description": oneline(pattern.get("description")),
            "blackout_start_days_before_quarter_end": "" if days is None else days,
            "groups_raw_json": json.dumps(groups, ensure_ascii=False),
        })
        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
