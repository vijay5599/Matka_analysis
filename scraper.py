import json
import os
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

def get_url_from_api_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "matka_api.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("Mahadevi")
        except Exception as e:
            print(f"Error reading api config: {e}")
    return "https://tara567.com/mrecords/mahadevi-panel-chart"

def scrape_mahadevi_chart(url=None):
    if not url:
        url = get_url_from_api_config()
    
    print(f"Scraping Matka data from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        print(f"Failed to fetch live URL: {e}. Checking local cache source...")
        # Fallback to local cache file if it exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_cache_path = os.path.join(base_dir, "mahadevi_history.json")
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
    table = soup.find("table", class_="clsResultsTable")
    if not table:
        # Try to find any table if the class name changed
        table = soup.find("table")
        if not table:
            raise ValueError("No results table found in the HTML content.")
            
    rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")
    
    parsed_records = []
    
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 8:
            continue
            
        # Parse date cell
        date_cell = tds[0]
        
        # Extract year
        year_div = date_cell.find("div", class_="text-upright")
        year_str = year_div.text.strip() if year_div else ""
        if not year_str:
            # Fallback regex for 4 digit year
            year_match = re.search(r"\b(20\d{2})\b", date_cell.text)
            year_str = year_match.group(1) if year_match else str(datetime.now().year)
            
        # Extract starting date (usually top date)
        top_div = date_cell.find("div", class_="dateDivTop")
        start_date_str = ""
        if top_div:
            # text might look like "30-Dec \n to" or "13-Jan"
            start_date_str = top_div.text.replace("to", "").strip()
        else:
            # Fallback parsing
            text = date_cell.text.strip()
            # Match date patterns like DD-MMM (e.g. 30-Dec)
            match = re.search(r"(\d{1,2}-[A-Za-z]{3})", text)
            if match:
                start_date_str = match.group(1)
                
        if not start_date_str:
            continue
            
        try:
            # Parse start date (Monday)
            # Example format: "30-Dec-2024"
            clean_date_str = f"{start_date_str}-{year_str}"
            start_date = datetime.strptime(clean_date_str, "%d-%b-%Y")
        except Exception as e:
            print(f"Error parsing date string '{start_date_str}-{year_str}': {e}")
            continue
            
        # Parse the 7 days (Monday to Sunday)
        for i in range(1, 8):
            day_cell = tds[i]
            day_name = weekday_names[i-1]
            day_date = start_date + timedelta(days=i-1)
            date_formatted = day_date.strftime("%Y-%m-%d")
            
            panna_div = day_cell.find("div", class_="divPanna")
            if panna_div:
                divs = panna_div.find_all("div")
                if len(divs) >= 3:
                    open_pana = divs[0].text.strip()
                    jodi = divs[1].text.strip()
                    close_pana = divs[2].text.strip()
                else:
                    # Alternative structure
                    text_parts = [d.text.strip() for d in divs]
                    if len(text_parts) == 3:
                        open_pana, jodi, close_pana = text_parts
                    else:
                        open_pana, jodi, close_pana = "***", "**", "***"
            else:
                # Fallback if text is flat
                text = day_cell.text.strip()
                # Split by whitespace or newlines
                parts = [p.strip() for p in re.split(r'\s+', text) if p.strip()]
                if len(parts) >= 3:
                    open_pana, jodi, close_pana = parts[0], parts[1], parts[2]
                else:
                    open_pana, jodi, close_pana = "***", "**", "***"
            
            # Clean and validate values
            is_valid = True
            if open_pana == "***" or jodi == "**" or close_pana == "***":
                is_valid = False
                
            open_single = None
            close_single = None
            
            if is_valid:
                # Verify sums and extract single digits
                # Jodi digits
                if len(jodi) == 2 and jodi.isdigit():
                    open_single = int(jodi[0])
                    close_single = int(jodi[1])
                else:
                    is_valid = False
            
            parsed_records.append({
                "date": date_formatted,
                "weekday": day_name,
                "weekday_num": i - 1, # 0 = Monday, 6 = Sunday
                "open_pana": open_pana if open_pana != "***" else None,
                "jodi": jodi if jodi != "**" else None,
                "close_pana": close_pana if close_pana != "***" else None,
                "open_single": open_single,
                "close_single": close_single,
                "is_valid": is_valid
            })
            
    # Sort chronologically
    parsed_records.sort(key=lambda x: x["date"])
    return parsed_records

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "mahadevi_history.json")
    try:
        records = scrape_mahadevi_chart()
        print(f"Successfully scraped {len(records)} records ({len([r for r in records if r['is_valid']])} valid draws).")
        
        with open(output_path, "w") as f:
            json.dump(records, f, indent=4)
        print(f"Saved cache to {output_path}")
    except Exception as e:
        print(f"Scraper execution failed: {e}")

if __name__ == "__main__":
    main()
