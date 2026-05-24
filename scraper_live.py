"""
scraper_live.py — Playwright-based headless scraper for Mahadevi Morning / Night markets.

These markets use Next.js SSR with client-side data hydration, meaning the result table
is injected via JavaScript after initial HTML load. Standard `requests` cannot see this data.
Playwright executes full JS in a real browser, waits for the table to appear, then extracts it.

Usage:
    python scraper_live.py                        # scrapes all 3 markets
    python scraper_live.py "Mahadevi Morning"     # scrapes one market
"""

import json
import os
import re
import sys
import asyncio
from datetime import datetime, timedelta

# ── Playwright import (graceful fallback if not installed) ────────────────────
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MARKET_CONFIG = {
    "Mahadevi Morning": {
        "url": "https://satkamatka.com.in/panel-chart-record/mahadevi-morning",
        "filename": "mahadevi_morning_history.json",
        "parser": "satkamatka",   # uses plain requests — no Playwright needed
    },
    "Mahadevi": {
        "url": "https://tara567.com/mrecords/mahadevi-panel-chart",
        "filename": "mahadevi_history.json",
        "parser": "tara567",       # uses Playwright (JS-rendered)
    },
    "Mahadevi Night": {
        "url": "https://satkamatka.com.in/panel-chart-record/mahadevi-night",
        "filename": "mahadevi_night_history.json",
        "parser": "satkamatka",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HTML parser for tara567.com (reused from scraper.py)
# ─────────────────────────────────────────────────────────────────────────────
def parse_html(html_content):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", class_="clsResultsTable") or soup.find("table")
    if not table:
        raise ValueError("No results table found in the rendered HTML.")


# ─────────────────────────────────────────────────────────────────────────────
# HTML parser for satkamatka.com.in (SSR — plain requests works fine)
#
# Table structure per row:
#   <td>DD/MM/YYYY<br>to<br>DD/MM/YYYY</td>   ← date range (start = Monday)
#   For each of 7 days (Mon-Sun), 3 consecutive <td>s:
#     <td>3<br>5<br>8</td>   ← open pana digits (join to get "358")
#     <td>66</td>            ← jodi
#     <td>4<br>5<br>7</td>   ← close pana digits
# ─────────────────────────────────────────────────────────────────────────────
def parse_satkamatka_html(html_content: str) -> list:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("No table found on satkamatka.com.in page.")

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    parsed_records = []

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        tds = row.find_all("td")
        # Expect 1 date cell + 7 days × 3 cells = 22 cells total
        if len(tds) < 22:
            continue

        # Parse date range from first cell — format "DD/MM/YYYY\nto\nDD/MM/YYYY"
        date_text = tds[0].get_text(separator=" ").strip()
        date_matches = re.findall(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", date_text)
        if not date_matches:
            # Try alternate format like "06-02-2023 to 12-02-2023"
            date_matches = re.findall(r"(\d{1,2}-\d{1,2}-\d{4})", date_text)
        if not date_matches:
            continue

        start_str = date_matches[0]
        # Parse start date (Monday)
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                start_date = datetime.strptime(start_str, fmt)
                break
            except ValueError:
                continue
        else:
            print(f"[satkamatka] Cannot parse date: '{start_str}'")
            continue

        # Parse 7 days — each day is 3 consecutive TDs starting at index 1
        for day_idx in range(7):
            td_base = 1 + day_idx * 3
            if td_base + 2 >= len(tds):
                break

            day_date = start_date + timedelta(days=day_idx)
            date_formatted = day_date.strftime("%Y-%m-%d")
            day_name = weekday_names[day_idx]

            # Open pana: digits separated by <br> — join without separator
            open_td = tds[td_base]
            open_digits = [t.strip() for t in open_td.get_text(separator="|").split("|") if t.strip()]
            open_pana = "".join(open_digits)

            # Jodi: plain 2-digit text
            jodi_td = tds[td_base + 1]
            jodi = jodi_td.get_text(strip=True)

            # Close pana: same as open
            close_td = tds[td_base + 2]
            close_digits = [t.strip() for t in close_td.get_text(separator="|").split("|") if t.strip()]
            close_pana = "".join(close_digits)

            # Validate
            is_valid = bool(
                open_pana and jodi and close_pana
                and open_pana not in ("***", "")
                and jodi not in ("**", "***", "")
                and len(jodi) == 2 and jodi.isdigit()
            )
            open_single = int(jodi[0]) if is_valid else None
            close_single = int(jodi[1]) if is_valid else None

            parsed_records.append({
                "date": date_formatted,
                "weekday": day_name,
                "weekday_num": day_idx,
                "open_pana": open_pana if is_valid else "",
                "jodi": jodi if is_valid else "",
                "close_pana": close_pana if is_valid else "",
                "open_single": open_single,
                "close_single": close_single,
                "is_valid": is_valid,
            })

    parsed_records.sort(key=lambda x: x["date"])
    return parsed_records



    rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")

    parsed_records = []
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 8:
            continue

        date_cell = tds[0]
        year_div = date_cell.find("div", class_="text-upright")
        year_str = year_div.text.strip() if year_div else ""
        if not year_str:
            year_match = re.search(r"\b(20\d{2})\b", date_cell.text)
            year_str = year_match.group(1) if year_match else str(datetime.now().year)

        top_div = date_cell.find("div", class_="dateDivTop")
        start_date_str = ""
        if top_div:
            start_date_str = top_div.text.replace("to", "").strip()
        else:
            text = date_cell.text.strip()
            match = re.search(r"(\d{1,2}-[A-Za-z]{3})", text)
            if match:
                start_date_str = match.group(1)

        if not start_date_str:
            continue

        try:
            clean_date_str = f"{start_date_str}-{year_str}"
            start_date = datetime.strptime(clean_date_str, "%d-%b-%Y")
        except Exception as e:
            print(f"[parser] Date parse error '{start_date_str}-{year_str}': {e}")
            continue

        for i in range(1, 8):
            day_cell = tds[i]
            day_name = weekday_names[i - 1]
            day_date = start_date + timedelta(days=i - 1)
            date_formatted = day_date.strftime("%Y-%m-%d")

            panna_div = day_cell.find("div", class_="divPanna")
            if panna_div:
                divs = panna_div.find_all("div")
                if len(divs) >= 3:
                    open_pana, jodi, close_pana = (divs[0].text.strip(),
                                                    divs[1].text.strip(),
                                                    divs[2].text.strip())
                else:
                    text_parts = [d.text.strip() for d in divs]
                    open_pana, jodi, close_pana = (
                        text_parts[0] if len(text_parts) > 0 else "***",
                        text_parts[1] if len(text_parts) > 1 else "**",
                        text_parts[2] if len(text_parts) > 2 else "***",
                    )
            else:
                text = day_cell.text.strip()
                parts = [p.strip() for p in re.split(r'\s+', text) if p.strip()]
                if len(parts) >= 3:
                    open_pana, jodi, close_pana = parts[0], parts[1], parts[2]
                else:
                    open_pana, jodi, close_pana = "***", "**", "***"

            is_valid = not (open_pana in ("***", "") or jodi in ("**", "") or close_pana in ("***", ""))
            open_single = None
            close_single = None
            if is_valid:
                if len(jodi) == 2 and jodi.isdigit():
                    open_single = int(jodi[0])
                    close_single = int(jodi[1])
                else:
                    is_valid = False

            parsed_records.append({
                "date": date_formatted,
                "weekday": day_name,
                "weekday_num": i - 1,
                "open_pana": open_pana if open_pana not in ("***", "") else "",
                "jodi": jodi if jodi not in ("**", "") else "",
                "close_pana": close_pana if close_pana not in ("***", "") else "",
                "open_single": open_single,
                "close_single": close_single,
                "is_valid": is_valid,
            })

    parsed_records.sort(key=lambda x: x["date"])
    return parsed_records


# ─────────────────────────────────────────────────────────────────────────────
# Playwright scraper — fetches page after full JS execution
# ─────────────────────────────────────────────────────────────────────────────
async def scrape_with_playwright(url: str, market_name: str) -> list:
    """
    Launches a headless Chromium browser, navigates to the URL, waits for the
    Next.js data to hydrate (resultData must be non-empty in __NEXT_DATA__),
    then extracts the rendered HTML and parses it.
    """
    print(f"[playwright] Opening headless browser for: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        # Navigate and wait for network to be idle (JS finished loading)
        await page.goto(url, wait_until="networkidle", timeout=45000)

        # Try to wait up to 15s for the result table to appear in DOM
        try:
            await page.wait_for_selector("table", timeout=15000)
            print(f"[playwright] Table detected in DOM.")
        except Exception:
            print(f"[playwright] Table not found — checking __NEXT_DATA__ for JSON...")

        # Capture HTML BEFORE closing the browser
        html = await page.content()

        # Also extract __NEXT_DATA__ JSON (fastest & most reliable)
        next_data_raw = await page.evaluate("""
            () => {
                const el = document.getElementById('__NEXT_DATA__');
                return el ? el.textContent : null;
            }
        """)

        # Done — close browser
        await browser.close()

    # First attempt: parse __NEXT_DATA__ JSON
    if next_data_raw:
        try:
            next_data = json.loads(next_data_raw)
            result_data = (
                next_data
                .get("props", {})
                .get("pageProps", {})
                .get("data", {})
                .get("resultData", [])
            )
            if result_data:
                print(f"[playwright] Found {len(result_data)} week-rows in __NEXT_DATA__.")
                return _parse_next_data(result_data)
            else:
                print(f"[playwright] __NEXT_DATA__.resultData is empty — falling back to HTML parse.")
        except Exception as e:
            print(f"[playwright] __NEXT_DATA__ JSON parse failed: {e}")

    # Second attempt: parse the fully-rendered HTML table
    return parse_html(html)


def _parse_next_data(result_data: list) -> list:
    """
    Parse the resultData array from tara567's __NEXT_DATA__ JSON.

    Actual format observed from tara567.com:
    resultData is a list of WEEKS, where each week is itself a LIST of day-objects:
    [
        [   # week 1
            {"date": "13-Apr-2026", "result": "89", "panna": "128-89-266", "displayValue": "..."},
            ...  (7 days)
        ],
        [   # week 2
            ...
        ]
    ]

    Each day-object has:
      - date:    "DD-Mon-YYYY"  e.g. "13-Apr-2026"
      - result:  "89"           (jodi, 2 digits)
      - panna:   "128-89-266"   (open_pana-jodi-close_pana)
    """
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    parsed_records = []

    for item in result_data:
        # Each item is a WEEK (list of day dicts), or could be a day dict directly
        if isinstance(item, list):
            days = item
        elif isinstance(item, dict):
            # Single day object at top level
            days = [item]
        else:
            continue

        for day in days:
            try:
                if not isinstance(day, dict):
                    continue

                date_str = day.get("date", "").strip()
                result = str(day.get("result", "") or "").strip()    # jodi e.g. "89"
                panna = str(day.get("panna", "") or "").strip()       # e.g. "128-89-266"

                if not date_str:
                    continue

                # Parse date — format "DD-Mon-YYYY" e.g. "13-Apr-2026"
                try:
                    day_date = datetime.strptime(date_str, "%d-%b-%Y")
                except ValueError:
                    # Try alternate formats
                    try:
                        day_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        print(f"[_parse_next_data] Cannot parse date: '{date_str}'")
                        continue

                date_formatted = day_date.strftime("%Y-%m-%d")
                day_name = weekday_names[day_date.weekday()]
                weekday_num = day_date.weekday()

                # Parse panna "128-89-266" → open_pana, jodi, close_pana
                parts = panna.split("-") if "-" in panna else []
                if len(parts) == 3:
                    open_pana, jodi, close_pana = parts[0].strip(), parts[1].strip(), parts[2].strip()
                elif result and len(result) == 2 and result.isdigit():
                    # Panna missing but jodi available
                    open_pana, jodi, close_pana = "", result, ""
                else:
                    open_pana, jodi, close_pana = "", "", ""

                # Fallback: use result field if jodi is empty
                if not jodi and result and len(result) == 2 and result.isdigit():
                    jodi = result

                is_valid = bool(
                    open_pana and jodi and close_pana
                    and len(jodi) == 2 and jodi.isdigit()
                )
                open_single = int(jodi[0]) if is_valid else None
                close_single = int(jodi[1]) if is_valid else None

                parsed_records.append({
                    "date": date_formatted,
                    "weekday": day_name,
                    "weekday_num": weekday_num,
                    "open_pana": open_pana,
                    "jodi": jodi,
                    "close_pana": close_pana,
                    "open_single": open_single,
                    "close_single": close_single,
                    "is_valid": is_valid,
                })
            except Exception as e:
                print(f"[_parse_next_data] Day parse error: {e} | day={day}")
                continue

    parsed_records.sort(key=lambda x: x["date"])
    return parsed_records



# ─────────────────────────────────────────────────────────────────────────────
# Public API — drop-in replacement for scraper.py's scrape_mahadevi_chart()
# ─────────────────────────────────────────────────────────────────────────────
def scrape_live(market_name: str = "Mahadevi") -> list:
    """
    Public scraper entry point. Routes to the correct scraper based on the
    'parser' field in MARKET_CONFIG:
      - 'satkamatka': uses plain requests (SSR HTML \u2014 no JS needed, fast)
      - 'tara567':    uses Playwright headless browser (JS-rendered)
    Falls back to local JSON cache if scraping fails.
    """
    import requests as req

    config = MARKET_CONFIG.get(market_name)
    if not config:
        raise ValueError(f"Unknown market: {market_name}. Valid: {list(MARKET_CONFIG.keys())}")

    local_cache_path = os.path.join(BASE_DIR, config["filename"])
    parser_type = config.get("parser", "tara567")
    url = config["url"]

    # ── Route: satkamatka.com.in \u2014 simple requests scrape ──────────────────────
    if parser_type == "satkamatka":
        try:
            print(f"[scraper_live] Fetching {market_name} from satkamatka.com.in...")
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            response = req.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            records = parse_satkamatka_html(response.text)
            if not records:
                raise ValueError("Parser returned 0 records from satkamatka.")
            print(f"[scraper_live] \u2705 Got {len(records)} records for {market_name}")
            return records
        except Exception as e:
            print(f"[scraper_live] satkamatka scrape failed: {e}")
            if os.path.exists(local_cache_path):
                print(f"[scraper_live] Falling back to local cache.")
                with open(local_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise

    # ── Route: tara567.com \u2014 Playwright headless browser ──────────────────────
    if not PLAYWRIGHT_AVAILABLE:
        print("[scraper_live] Playwright not installed \u2014 reading local cache.")
        if os.path.exists(local_cache_path):
            with open(local_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise RuntimeError("Playwright not installed and no local cache found.")

    try:
        records = asyncio.run(scrape_with_playwright(url, market_name))
        if not records:
            raise ValueError("Scraper returned 0 records.")
        return records
    except Exception as e:
        print(f"[scraper_live] Playwright scrape failed: {e}")
        if os.path.exists(local_cache_path):
            print(f"[scraper_live] Falling back to local cache: {local_cache_path}")
            with open(local_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    target_markets = sys.argv[1:] if len(sys.argv) > 1 else list(MARKET_CONFIG.keys())

    for market_name in target_markets:
        market_name = market_name.strip()
        if market_name not in MARKET_CONFIG:
            print(f"[main] Unknown market '{market_name}'. Skipping.")
            continue

        config = MARKET_CONFIG[market_name]
        output_path = os.path.join(BASE_DIR, config["filename"])
        print(f"\n{'='*60}")
        print(f" Scraping: {market_name}")
        print(f"{'='*60}")

        try:
            records = scrape_live(market_name)
            valid_count = len([r for r in records if r.get("is_valid")])
            print(f"✅ {len(records)} total records, {valid_count} valid draws.")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=4)
            print(f"💾 Saved → {output_path}")
        except Exception as e:
            print(f"❌ Failed for '{market_name}': {e}")


if __name__ == "__main__":
    main()
