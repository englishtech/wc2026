from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import time
import re
import pandas as pd
from config import (
    logger, ELORATINGS_URL_QUALIFIERS, ELORATINGS_URL_FINALS, 
    WC2026_QUALIFIERS_FILE, WC2026_FINALS_FILE,
    ELORATINGS_URL_PLAYOFFS, WC2026_PLAYOFFS_FILE,
    WC2026_PLAYOFFS_FOURTHS_FILE, WC2026_PLAYOFFS_FOURTHS_FILE
)

chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
service = Service(executable_path="D:/Python/chromedriver.exe")

chrome_options.add_argument("--headless")
chrome_options.add_argument('--headless=chrome')
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--log-level=2")  # WARNING
chrome_options.add_argument(
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
chrome_options.page_load_strategy = 'eager'


def parse_eloratings(url: str):
    start_time = time.time()
    print("[+] Запуск драйвера браузера...")
    with webdriver.Chrome(service=service, options=chrome_options) as browser:
        browser.get(url)
        time.sleep(10)
        rows = []
        for row in browser.find_elements(By.CLASS_NAME, "slick-row"):
            cells = row.find_elements(By.CLASS_NAME, "slick-cell")
            if len(cells) < 8: continue
            
            # Вспомогательная функция: берёт текст, заменяет спец-минус, делит по <br>
            def split_cell(cell, idx=0):
                txt = cell.text.replace("−", "-")
                parts = re.split(r"\n|<br\s*/?>", txt)
                return parts[idx].strip() if idx < len(parts) else None
            
            try:
                raw_date = f"{split_cell(cells[0], 0)} {split_cell(cells[0], 1)}"
                date = datetime.strptime(raw_date, "%b %d %Y").strftime("%Y-%m-%d") if raw_date.strip() else None
                teams = [a.text for a in cells[1].find_elements(By.TAG_NAME, "a")]
                scores = split_cell(cells[2], 0), split_cell(cells[2], 1)
                
                txt = cells[3].text.strip()
                parts = [p.strip() for p in txt.replace("\n", "|").split("|") if p.strip()]
                tournament = parts[0]
                location = parts[1].replace("in ", "").strip() if len(parts) > 1 else ""
                
                elo = split_cell(cells[5], 0), split_cell(cells[5], 1)
                rank = split_cell(cells[7], 0), split_cell(cells[7], 1)
                
                rows.append([date, teams[0], teams[1], scores[0], scores[1], tournament, location,
                            elo[0], elo[1], rank[0], rank[1]])
            except: continue

        df = pd.DataFrame(rows, columns=["date","home_team","away_team","home_score","away_score",
                                        "tournament", "location", "home_elo_rating","away_elo_rating","home_elo_rank","away_elo_rank"])
        df.to_csv(f"{WC2026_PLAYOFFS_FILE}.csv", index=False, encoding="utf-8")
        print(f"✅ {len(df)} матчей сохранено")
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n[+] Общее время выполнения: {total_time:.2f} секунд")

def parse_finals_eloratings(url: str):
    start_time = time.time()
    print("[+] Запуск драйвера браузера...")
    with webdriver.Chrome(service=service, options=chrome_options) as browser:
        browser.get(url)
        time.sleep(10)
        rows = []
        for row in browser.find_elements(By.CLASS_NAME, "slick-row"):
            cells = row.find_elements(By.CLASS_NAME, "slick-cell")
            if len(cells) < 8: continue
            
            # Вспомогательная функция: берёт текст, заменяет спец-минус, делит по <br>
            def split_cell(cell, idx=0):
                txt = cell.text.replace("−", "-")
                parts = re.split(r"\n|<br\s*/?>", txt)
                return parts[idx].strip() if idx < len(parts) else None
            
            try:
                raw_date = f"{split_cell(cells[0], 0)} {split_cell(cells[0], 1)}"
                date = datetime.strptime(f"{raw_date} 2026", "%a %b %d %Y").strftime("%Y-%m-%d") if raw_date.strip() else None
                
                teams = [a.text for a in cells[1].find_elements(By.TAG_NAME, "a")]
                # scores = split_cell(cells[2], 0), split_cell(cells[2], 1)
                
                txt = cells[2].text.strip()
                parts = [p.strip() for p in txt.replace("\n", "|").split("|") if p.strip()]
                tournament = parts[0]
                location = parts[1].replace("in ", "").strip() if len(parts) > 1 else ""
                
                elo = split_cell(cells[4], 0), split_cell(cells[4], 1)
                rank = split_cell(cells[3], 0), split_cell(cells[3], 1)
                
                rows.append([date, teams[0], teams[1], tournament, location,
                            elo[0], elo[1], rank[0], rank[1]])
                print(len(rows), date, teams)
            except: continue

        df = pd.DataFrame(rows, columns=["date","home_team","away_team",
                                        "tournament", "location", "home_elo_rating","away_elo_rating","home_elo_rank","away_elo_rank"])
        df.to_csv(f"{WC2026_PLAYOFFS_FOURTHS_FILE}.csv", index=False, encoding="utf-8")
        print(f"✅ {len(df)} матчей сохранено")
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n[+] Общее время выполнения: {total_time:.2f} секунд")

def clean_eloratings():
    df = pd.read_csv(f"{WC2026_PLAYOFFS_FOURTHS_FILE}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["date","home_team","away_team"])
    df = df[~df["tournament"].str.contains("Friendly", na=False)]
    df.to_csv(f"data/{WC2026_PLAYOFFS_FOURTHS_FILE}_clean.csv", index=False, encoding="utf-8")
    print(f"Готово: {len(df)} матчей")  # Выводим итог

def clean_finals_eloratings():
    df = pd.read_csv(f"{WC2026_PLAYOFFS_FOURTHS_FILE}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["date","home_team","away_team"])
    df = df[~df["tournament"].str.contains("Friendly", na=False)]
    df.to_csv(f"data/{WC2026_PLAYOFFS_FOURTHS_FILE}_clean.csv", index=False, encoding="utf-8")
    print(f"Готово: {len(df)} матчей")  # Выводим итог


parse_finals_eloratings(ELORATINGS_URL_PLAYOFFS)
clean_finals_eloratings()
