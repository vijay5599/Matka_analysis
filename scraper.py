import json
import os
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

def get_url_from_api_config(market_name="Mahadevi"):
    default_urls = {
        "Mahadevi": "https://dpbossss.boston/panel-chart-record/mahadevi.php",
        "Mahadevi Night": "https://dpbossss.boston/panel-chart-record/mahadevi-night.php",
        "Mahadevi Morning": "https://dpbossss.boston/panel-chart-record/mahadevi-morning.php"
    }
    return default_urls.get(market_name, default_urls["Mahadevi"])

def scrape_mahadevi_chart(url=None, market_name="Mahadevi"):
    if not url:
        url = get_url_from_api_config(market_name)
    
    print(f"Scraping Matka data from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    filename = "mahadevi_history.json"
    if market_name == "Mahadevi Morning":
        filename = "mahadevi_morning_history.json"
    elif market_name == "Mahadevi Night":
        filename = "mahadevi_night_history.json"
        
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        print(f"Failed to fetch live URL: {e}. Checking local cache source...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_cache_path = os.path.join(base_dir, filename)
        if os.path.exists(local_cache_path):
            print(f"Reading from local cache: {local_cache_path}")
            try:
                with open(local_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as read_err:
                raise RuntimeError(f"Could not read local cache file: {read_err}")
        else:
            raise RuntimeError(f"Could not fetch URL and local backup not found. Error: {e}")
            
    return parse_html(html_content)

def parse_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("No results table found in the HTML content.")
            
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")
    
    parsed_records = []
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 4:
            continue
            
        date_full_str = tds[0].get_text(separator=" ", strip=True).replace("  ", " ")
        # format: "06-02-2023 to 12-02-2023"
        start_date_str = date_full_str.split("to")[0].strip()
        
        try:
            start_date_str = start_date_str.replace("/", "-")
            start_date = datetime.strptime(start_date_str, "%d-%m-%Y")
        except Exception as e:
            print(f"Error parsing date string '{start_date_str}': {e}")
            continue

        for i in range(7):
            day_name = weekday_names[i]
            day_date = start_date + timedelta(days=i)
            date_formatted = day_date.strftime("%Y-%m-%d")
            
            col_offset = 1 + (i * 3)
            open_pana = "***"
            jodi = "**"
            close_pana = "***"
            
            if col_offset + 2 < len(tds):
                open_pana = tds[col_offset].get_text(separator="").strip()
                jodi = tds[col_offset + 1].get_text(separator="").strip()
                close_pana = tds[col_offset + 2].get_text(separator="").strip()
                
            if jodi == "**" or jodi == "***" or not jodi:
                jodi = None
            if open_pana == "***" or not open_pana:
                open_pana = None
            if close_pana == "***" or not close_pana:
                close_pana = None
                
            is_valid = True
            if not open_pana or not jodi or not close_pana:
                is_valid = False
                
            open_single = None
            close_single = None
            if is_valid and jodi and len(jodi) == 2 and jodi.isdigit():
                open_single = int(jodi[0])
                close_single = int(jodi[1])
            else:
                is_valid = False
                
            parsed_records.append({
                "date": date_formatted,
                "weekday": day_name,
                "weekday_num": i, # 0 = Monday, 6 = Sunday
                "open_pana": open_pana,
                "jodi": jodi,
                "close_pana": close_pana,
                "open_single": open_single,
                "close_single": close_single,
                "is_valid": is_valid
            })
            
    parsed_records.sort(key=lambda x: x["date"])
    return parsed_records

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    markets = {
        "Mahadevi Morning": "mahadevi_morning_history.json",
        "Mahadevi": "mahadevi_history.json",
        "Mahadevi Night": "mahadevi_night_history.json"
    }
    
    for market_name, filename in markets.items():
        print(f"\n--- Scraping {market_name} ---")
        output_path = os.path.join(base_dir, filename)
        try:
            records = scrape_mahadevi_chart(market_name=market_name)
            print(f"Successfully scraped {len(records)} records ({len([r for r in records if r['is_valid']])} valid draws).")
            
            with open(output_path, "w") as f:
                json.dump(records, f, indent=4)
            print(f"Saved cache to {output_path}")
        except Exception as e:
            print(f"Scraper execution failed for {market_name}: {e}")

if __name__ == "__main__":
    main()
