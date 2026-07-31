#!/usr/bin/env python3
"""
01_read_files.py – recurring blackout analysis (one policy per API call)
python 01_read_files.py

Goal:
    For each policy file in ex19_policies/, analyze recurring quarterly
    trading windows and blackout periods and combine with metadata
    (CIK and filing_date) from ex19_metadata.csv.

Output:
    blackout_summary.csv

Requirements:
    pip install openai python-dotenv beautifulsoup4

.env file (same directory as this script):
    OPENAI_API_KEY=sk-...

"""

import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from openai import OpenAI # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

EX19_DIR = Path("ex19_policies")
METADATA_FILE = EX19_DIR / "ex19_metadata.csv"
OUTPUT_CSV = "blackout_summary.csv"

# Stay with this model
MODEL_NAME = "gpt-5.4-nano"

# Allow larger context for better accuracy
MAX_CONTEXT_CHARS = 30000  

# Number of retries for API calls
MAX_RETRIES = 2

# TEST MODE: Process only first N files
TEST_MODE = False
TEST_FILE_LIMIT = 3

# RANGE MODE: Process specific range of files
RANGE_MODE = False
RANGE_START = 90  # Start at file index 90 (0-based, so this is the 91st file)
RANGE_END = 95   # End at file index 100 (exclusive, so up to 100th file)


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


# ---------------------------------------------------------------------
# TEXT / FILE HELPERS
# ---------------------------------------------------------------------


def extract_text(path: Path) -> str:
    """
    Read file and return plain text.
    If it looks like HTML, strip tags with BeautifulSoup.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            raw = path.read_bytes().decode("latin-1", errors="ignore")
        except Exception:
            raw = path.read_bytes().decode("cp1252", errors="ignore")

    if "<html" in raw.lower() or "<body" in raw.lower():
        soup = BeautifulSoup(raw, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        
        return text

    return raw



def lightly_filter(text: str) -> str:
    """
    Extract paragraphs containing window/blackout information.
    Prioritize paragraphs with specific timing language.
    """
    text_norm = re.sub(r"\r\n?", "\n", text)
    paragraphs = re.split(r"\n\s*\n", text_norm)

    # Score each paragraph by relevance
    scored = []
    for p in paragraphs:
        lower = p.lower()
        score = 0
        
        # High priority phrases
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

        if score > 0:
            scored.append((score, p))
    
    # Sort by score, take top paragraphs
    scored.sort(reverse=True, key=lambda x: x[0])
    top_paragraphs = [p for score, p in scored[:50]]  # Top 50 paragraphs
    
    joined = "\n\n".join(top_paragraphs)
    
    if joined and len(joined) > 200:
        return joined[:MAX_CONTEXT_CHARS]
    
    # Fallback to full text
    return text[:MAX_CONTEXT_CHARS]




# ---------------------------------------------------------------------
# METADATA LOADING
# ---------------------------------------------------------------------


def load_metadata() -> Dict[str, Dict[str, str]]:
    """
    Load exhibit metadata from ex19_metadata.csv into a dict:
        filename -> {cik, ticker, company_name, sector, market_value, accession, filing_date, url}
    """
    meta: Dict[str, Dict[str, str]] = {}

    if not METADATA_FILE.exists():
        logging.warning(
            f"No metadata file found at {METADATA_FILE}; metadata will be empty."
        )
        return meta

    with METADATA_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "").strip()
            if not fname:
                continue
            meta[fname] = {
                'cik': row.get('cik', '').strip(),
                'ticker': row.get('ticker', '').strip(),
                'company_name': row.get('company_name', '').strip(),
                'sector': row.get('sector', '').strip(),
                'market_value': row.get('market_value', '').strip(),
                'accession': row.get('accession', '').strip(),
                'filing_date': row.get('filing_date', '').strip(),
                'url': row.get('url', '').strip(),
            }

    logging.info(f"Loaded metadata for {len(meta)} exhibits from {METADATA_FILE}")
    return meta


# ---------------------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------------------

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

def create_client() -> OpenAI:
    """Create OpenAI client, loading key from .env next to this script."""
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    load_dotenv(dotenv_path=env_path)

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logging.error(f"OPENAI_API_KEY not found in {env_path}")
        sys.exit(1)

    return OpenAI(api_key=key)


def validate_response(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Validate and clean the model response."""
    # Ensure required fields exist
    data.setdefault("has_recurring_blackout", False)
    data.setdefault("has_ad_hoc_blackout", False)
    data.setdefault("requires_preclearance", False)
    data.setdefault("preclearance_description", "")
    data.setdefault("prohibits_hedging", False)
    data.setdefault("hedging_description", "")
    data.setdefault("general_description", "")
    data.setdefault("groups", [])
    
    # Validate numeric fields
    for group in data.get("groups", []):
        pattern = group.get("blackout_pattern", {})
        
        # Validate blackout_start_days_before_quarter_end
        days_before = pattern.get("blackout_start_days_before_quarter_end")
        if days_before is not None:
            try:
                days_before = int(days_before)
                if days_before < 0 or days_before > 90:
                    logging.warning(
                        f"{filename}: Invalid blackout_start_days_before_quarter_end "
                        f"value {days_before}, setting to null"
                    )
                    pattern["blackout_start_days_before_quarter_end"] = None
                else:
                    # Ensure it's stored as int
                    pattern["blackout_start_days_before_quarter_end"] = days_before
            except (ValueError, TypeError):
                logging.warning(
                    f"{filename}: Non-integer blackout_start_days_before_quarter_end, "
                    f"setting to null"
                )
                pattern["blackout_start_days_before_quarter_end"] = None
    
    return data


def analyze_policy(client: OpenAI, filename: str, text: str) -> Dict[str, Any]:
    """
    Call the model for a single policy and parse its JSON answer.
    Includes retry logic and validation.
    """
    user_prompt = (
        f"Policy file: {filename}\n\n"
        f"POLICY TEXT:\n{text}\n\n"
        "Return JSON only. Determine exact blackout_start_days_before_quarter_end (no rounding)."
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,  # Deterministic output
                response_format={"type": "json_object"}  # Force JSON mode
            )

            raw = response.choices[0].message.content.strip()
            logging.debug(f"Raw model output for {filename}:\n{raw}")

            # Parse JSON
            data = json.loads(raw)
            
            # Validate and clean the response
            data = validate_response(data, filename)
            
            return data

        except json.JSONDecodeError as e:
            logging.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} - JSON parse error for "
                f"{filename}: {e}"
            )
            if attempt == MAX_RETRIES - 1:
                # Last attempt failed
                break
        
        except Exception as e:
            logging.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} - API error for "
                f"{filename}: {e}"
            )
            if attempt == MAX_RETRIES - 1:
                # Last attempt failed
                break

    # All attempts failed
    logging.error(f"Failed to analyze {filename} after {MAX_RETRIES} attempts")
    return {
        "has_recurring_blackout": False,
        "has_ad_hoc_blackout": False,
        "requires_preclearance": False,
        "preclearance_description": "",
        "prohibits_hedging": False,
        "hedging_description": "",
        "general_description": "Analysis failed after multiple attempts.",
        "groups": [],
    }




# ---------------------------------------------------------------------
# RESULT POST-PROCESSING
# ---------------------------------------------------------------------


def select_primary_group(groups: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    From the groups list, pick the primary group (broadest coverage).
    Typically this is the first group, which should be the least restrictive.
    """
    if not groups:
        return None
    return groups[0]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    setup_logging()

    if not EX19_DIR.exists():
        logging.error(f"Folder not found: {EX19_DIR.resolve()}")
        sys.exit(1)

    all_files = sorted(
        [
            p
            for p in EX19_DIR.iterdir()
            if p.is_file() and p.name != METADATA_FILE.name
        ]
    )
    
    # Apply range mode if enabled
    if RANGE_MODE:
        files = all_files[RANGE_START:RANGE_END]
        logging.info(
            f"RANGE MODE: Processing files {RANGE_START} to {RANGE_END} "
            f"({len(files)} files) out of {len(all_files)} total"
        )
    # Apply test mode limit if enabled
    elif TEST_MODE:
        files = all_files[:TEST_FILE_LIMIT]
        logging.info(
            f"TEST MODE: Processing only first {TEST_FILE_LIMIT} files "
            f"out of {len(all_files)} total files"
        )
    else:
        files = all_files
        logging.info(f"Found {len(files)} policy files in {EX19_DIR.resolve()}")

    if not files:
        logging.info("No files found. Exiting.")
        return

    metadata = load_metadata()
    client = create_client()

    rows: List[Dict[str, Any]] = []

    for i, f in enumerate(files, 1):
        logging.info(f"[{i}/{len(files)}] Analyzing {f.name}")
        full_text = extract_text(f)
        context = lightly_filter(full_text)

        result = analyze_policy(client, f.name, context)

        has_recurring_blackout = result.get("has_recurring_blackout", False)
        has_ad_hoc_blackout = result.get("has_ad_hoc_blackout", False)
        requires_preclearance = result.get("requires_preclearance", False)
        preclearance_description = result.get("preclearance_description", "")
        prohibits_hedging = result.get("prohibits_hedging", False)
        hedging_description = result.get("hedging_description", "")
        general_description = result.get("general_description", "")

        groups = result.get("groups") or []
        primary = select_primary_group(groups)

        # Defaults if no group info
        group_name = ""
        blackout_desc = ""
        blackout_start_days_before_q_end = ""

        if primary:
            group_name = str(primary.get("name", ""))
            pattern = primary.get("blackout_pattern") or {}

            blackout_desc = str(pattern.get("description", ""))

            bstart_val = pattern.get(
                "blackout_start_days_before_quarter_end", None
            )
            if bstart_val is not None:
                blackout_start_days_before_q_end = str(bstart_val)

        # Lookup metadata
        meta_row = metadata.get(f.name, {})
        cik = meta_row.get("cik", "")
        ticker = meta_row.get("ticker", "")
        company_name = meta_row.get("company_name", "")
        sector = meta_row.get("sector", "")
        market_value = meta_row.get("market_value", "")
        accession = meta_row.get("accession", "")
        filing_date = meta_row.get("filing_date", "")
        url = meta_row.get("url", "")

        # Optional: store full groups JSON as string for debugging
        groups_json = json.dumps(groups, ensure_ascii=False)

        row = {
            "filename": f.name,
            "cik": cik,
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "market_value": market_value,
            "accession": accession,
            "filing_date": filing_date,
            "url": url,
            "has_recurring_blackout": has_recurring_blackout,
            "has_ad_hoc_blackout": has_ad_hoc_blackout,
            "requires_preclearance": requires_preclearance,
            "preclearance_description": preclearance_description.replace("\n", " "),
            "prohibits_hedging": prohibits_hedging,
            "hedging_description": hedging_description.replace("\n", " "),
            "general_description": general_description.replace("\n", " "),
            "group_name": group_name,
            "blackout_description": blackout_desc.replace("\n", " "),
            "blackout_start_days_before_quarter_end": blackout_start_days_before_q_end,
            "groups_raw_json": groups_json,
        }
        rows.append(row)

    # Write CSV
    fieldnames = [
        "filename",
        "accession",
        "cik",
        "ticker",
        "company_name",
        "sector",
        "market_value",
        "filing_date",
        "has_recurring_blackout",
        "has_ad_hoc_blackout",
        "requires_preclearance",
        "preclearance_description",
        "prohibits_hedging",
        "hedging_description",
        "general_description",
        "group_name",
        "blackout_description",
        "blackout_start_days_before_quarter_end",
        "url",
        "groups_raw_json",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if TEST_MODE:
        logging.info(
            f"TEST MODE COMPLETE: Wrote {len(rows)} rows to {OUTPUT_CSV} "
            f"({len(all_files) - len(files)} files skipped)"
        )
    else:
        logging.info(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
