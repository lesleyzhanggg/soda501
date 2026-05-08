"""
HKSAR Government Press Release Scraper (Bilingual)
====================================================
Scrapes BOTH English and Traditional Chinese press releases.

English: https://www.info.gov.hk/gia/general/YYYYMM/DD.htm
Traditional Chinese: https://www.info.gov.hk/gia/general/YYYYMM/DDc.htm
Simplified Chinese: https://sc.isd.gov.hk/TuniS/www.info.gov.hk/gia/general/YYYYMM/DDc.htm?fontSize=1

Usage:
    python Code.py                    # Default: June 9, 2019 - Sept 30, 2020
    python Code.py 20190901 20191015  # Custom range

Output:
    hksar_bilingual.xlsx  — Two sheets: "english" and "chinese"
                            Each sheet has: date, title, url, priority, text, keyword flags
    statements_en/        — Individual English .txt files
    statements_zh/        — Individual Chinese .txt files
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import re
import sys
from tqdm import tqdm

BASE_URL = "https://www.info.gov.hk/gia/general"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

# ============================================================
# KEYWORDS (bilingual) — EXTENDED for full campaign period
# Covers: ELAB, Anti-Mask Law, PolyU siege, District Council,
#         COVID-19 + protest intersection, National Security Law
# ============================================================
EN_KEYWORDS = [
    # Government actors
    'chief executive', 'carrie lam', 'ce meets', 'ce speaks', 'ce attend',
    'secretary for security', 'john lee',
    'secretary for justice', 'teresa cheng',
    # Core ELAB terms
    'police', 'protest', 'bill', 'extradition', 'withdrawal', 'withdraw',
    'violence', 'violent', 'mask', 'emergency', 'ordinance',
    'public order', 'prohibition', 'riot', 'rioter',
    'condemn', 'vandal', 'disorder', 'radical',
    'dialogue', 'demand', 'rule of law',
    # Post-Nov 2019 terms
    'district council', 'election',
    'polytechnic', 'polyu', 'university', 'campus',
    'national security', 'security law', 'secession', 'subversion',
    'terrorism', 'terrorist',
    'foreign interference', 'external force',
    'black bloc', 'black-clad',
    'molotov', 'petrol bomb', 'arson',
    'arrest', 'prosecut',
    # COVID + protest intersection
    'gathering ban', 'group gathering', 'social distancing',
    'cap 599', 'prevention and control of disease',
    # Framing terms (key for text analysis)
    'mob', 'thug', 'lawless', 'criminal',
    'restore order', 'restore calm', 'stability',
]

ZH_KEYWORDS = [
    # Government actors
    '行政長官', '林鄭月娥', '特首',
    '保安局', '李家超',
    '律政司', '鄭若驊',
    # Core ELAB terms
    '警方', '警務處', '警隊', '警察',
    '逃犯條例', '修例', '撤回', '暫緩',
    '暴力', '暴徒', '止暴制亂', '激進',
    '蒙面', '禁蒙面', '緊急',
    '公眾秩序', '遊行', '集會', '示威',
    '譴責', '破壞', '法治',
    '對話', '訴求',
    # Post-Nov 2019 terms
    '區議會', '選舉',
    '理工大學', '理大', '大學', '校園',
    '國家安全', '國安法', '港區國安法', '分裂國家', '顛覆',
    '恐怖主義', '恐怖活動', '恐怖份子',
    '外國勢力', '外部勢力', '境外',
    '黑衣人', '黑衣',
    '汽油彈', '縱火',
    '拘捕', '檢控', '起訴',
    # COVID + protest intersection
    '限聚令', '聚集', '社交距離',
    '禁聚令', '防疫',
    '599章', '預防及控制疾病',
    # Framing terms
    '暴亂', '暴動', '刑事', '罪犯',
    '恢復秩序', '恢復平靜', '穩定',
    '攬炒', '攬抄',
]

EN_HIGH = [
    'chief executive', 'carrie lam',
    'withdraw', 'withdrawal',
    'mask', 'prohibition',
    'emergency regulation',
    'rule of law', 'dialogue',
    'stop violence', 'curb disorder',
    # New high-priority
    'national security', 'security law',
    'polytechnic', 'polyu',
    'district council election',
    'terrorism', 'terrorist',
    'gathering ban',
]

ZH_HIGH = [
    '行政長官', '林鄭月娥', '特首',
    '撤回', '暫緩',
    '蒙面', '禁蒙面',
    '緊急法', '緊急情況規例',
    '止暴制亂',
    '法治', '對話',
    # New high-priority
    '國家安全', '國安法', '港區國安法',
    '理工大學', '理大',
    '區議會選舉',
    '恐怖主義', '恐怖活動',
    '限聚令',
]


def get_page(url):
    """Fetch a page with retry."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = 'utf-8'
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            if attempt == 2:
                print(f"  Failed after 3 attempts: {url} ({e})")
            time.sleep(1)
    return None


def parse_daily_index(html, date_str, lang):
    """Parse daily index page for press release links."""
    soup = BeautifulSoup(html, 'html.parser')
    releases = []

    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]

    for link in soup.find_all('a', href=True):
        href = link['href']
        title = link.get_text(strip=True)

        if not title or len(title) < 3:
            continue

        # Match press release links (P2019XXXXXXX.htm or P2020XXXXXXX.htm)
        if re.search(r'P20\d{11}', href) or '/gia/general/' in href:
            # Normalize URL
            if href.startswith('P20') or href.startswith('./P20'):
                fname = href.replace('./', '')
                full_url = f"{BASE_URL}/{year}{month}/{day}/{fname}"
            elif href.startswith('/'):
                full_url = f"https://www.info.gov.hk{href}"
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = f"{BASE_URL}/{year}{month}/{day}/{href}"

            releases.append({
                'date': f"{year}-{month}-{day}",
                'title': title,
                'url': full_url,
                'lang': lang,
            })

    return releases


def classify_release(title, lang, text=""):
    """Classify by priority."""
    combined = (title + " " + text).lower()

    high_kws = EN_HIGH if lang == 'en' else ZH_HIGH
    all_kws = EN_KEYWORDS if lang == 'en' else ZH_KEYWORDS

    # For Chinese, don't lowercase (Chinese doesn't have case)
    if lang == 'zh':
        combined_raw = title + " " + text
        for kw in high_kws:
            if kw in combined_raw:
                return 'high'
        for kw in all_kws:
            if kw in combined_raw:
                return 'medium'
    else:
        for kw in high_kws:
            if kw.lower() in combined:
                return 'high'
        for kw in all_kws:
            if kw.lower() in combined:
                return 'medium'

    return 'low'


def fetch_full_text(url):
    """Fetch and extract full text from a press release page."""
    html = get_page(url)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Try known content containers
    content = None
    for selector in [
        {'id': 'pressrelease'},
        {'id': 'content_body'},
        {'class_': 'content'},
        {'id': 'bodyContent'},
    ]:
        content = soup.find('div', selector)
        if content:
            break

    if not content:
        content = soup.find('body')

    if content:
        for tag in content.find_all(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        text = content.get_text('\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    return soup.get_text('\n', strip=True)


def flag_keywords(text, lang):
    """Flag key concepts in text."""
    t = str(text)
    t_lower = t.lower()

    if lang == 'en':
        return {
            'mentions_withdrawal': bool(re.search(r'withdraw.*bill|formally withdraw|withdrawn', t_lower)),
            'mentions_violence': bool(re.search(r'violen', t_lower)),
            'mentions_rioter': bool(re.search(r'rioter|riot\b', t_lower)),
            'mentions_continue': bool(re.search(r'continue|persist|ongoing|linger', t_lower)),
            'mentions_stop_violence': bool(re.search(r'stop.*violen|end.*violen|curb.*disorder', t_lower)),
            'mentions_mask': bool(re.search(r'face cover|anti-mask|prohibition.*face', t_lower)),
            'mentions_emergency': bool(re.search(r'emergenc.*regulat', t_lower)),
            'mentions_dialogue': bool(re.search(r'dialogue|listen.*public|move forward', t_lower)),
            'mentions_condemn': bool(re.search(r'condemn', t_lower)),
            'mentions_rule_of_law': bool(re.search(r'rule of law', t_lower)),
            'mentions_vandal': bool(re.search(r'vandal|destructi|criminal damage', t_lower)),
            'mentions_foreign': bool(re.search(r'foreign.*force|foreign.*interfer|external.*force', t_lower)),
            # New flags
            'mentions_national_security': bool(re.search(r'national security|security law', t_lower)),
            'mentions_terrorism': bool(re.search(r'terroris', t_lower)),
            'mentions_polyu': bool(re.search(r'polytechnic|polyu', t_lower)),
            'mentions_election': bool(re.search(r'district council.*election|election', t_lower)),
            'mentions_gathering_ban': bool(re.search(r'gathering.*ban|group gathering|cap 599|social distanc', t_lower)),
            'mentions_arson': bool(re.search(r'arson|petrol bomb|molotov|firebomb', t_lower)),
            'mentions_arrest': bool(re.search(r'arrest|prosecut|charged with', t_lower)),
            'mentions_mob': bool(re.search(r'\bmob\b|thug|lawless|black.?clad', t_lower)),
            'mentions_restore_order': bool(re.search(r'restore.*order|restore.*calm|restor.*stability', t_lower)),
        }
    else:  # zh
        return {
            'mentions_withdrawal': bool(re.search(r'撤回|撤銷', t)),
            'mentions_suspension': bool(re.search(r'暫緩', t)),
            'mentions_violence': bool(re.search(r'暴力', t)),
            'mentions_rioter': bool(re.search(r'暴徒|暴亂', t)),
            'mentions_continue': bool(re.search(r'繼續|持續|仍然', t)),
            'mentions_stop_violence': bool(re.search(r'止暴制亂|制止暴力|平息', t)),
            'mentions_mask': bool(re.search(r'蒙面|禁蒙面|面罩', t)),
            'mentions_emergency': bool(re.search(r'緊急法|緊急情況規例|緊急規例', t)),
            'mentions_dialogue': bool(re.search(r'對話|聆聽|溝通', t)),
            'mentions_condemn': bool(re.search(r'譴責|嚴厲', t)),
            'mentions_rule_of_law': bool(re.search(r'法治', t)),
            'mentions_foreign': bool(re.search(r'外國勢力|外部勢力|境外', t)),
            'mentions_radical': bool(re.search(r'激進', t)),
            # New flags
            'mentions_national_security': bool(re.search(r'國家安全|國安法|港區國安法', t)),
            'mentions_terrorism': bool(re.search(r'恐怖主義|恐怖活動|恐怖份子', t)),
            'mentions_polyu': bool(re.search(r'理工大學|理大', t)),
            'mentions_election': bool(re.search(r'區議會.*選舉|選舉', t)),
            'mentions_gathering_ban': bool(re.search(r'限聚令|禁聚令|聚集.*禁|599章', t)),
            'mentions_arson': bool(re.search(r'縱火|汽油彈|火彈', t)),
            'mentions_arrest': bool(re.search(r'拘捕|檢控|起訴|落案', t)),
            'mentions_mob': bool(re.search(r'黑衣人|黑衣|暴民', t)),
            'mentions_restore_order': bool(re.search(r'恢復秩序|恢復平靜|回復.*穩定', t)),
            'mentions_laamchau': bool(re.search(r'攬炒|攬抄|玉石俱焚', t)),
        }


def generate_date_range(start_str, end_str):
    start = datetime.strptime(start_str, '%Y%m%d')
    end = datetime.strptime(end_str, '%Y%m%d')
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    return dates


def main():
    start_date = '20190609'
    end_date = '20200930'

    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]

    print(f"HKSAR Bilingual Press Release Scraper")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Languages: English + Traditional Chinese")
    print(f"{'='*60}")

    os.makedirs('statements_en', exist_ok=True)
    os.makedirs('statements_zh', exist_ok=True)

    dates = generate_date_range(start_date, end_date)
    all_releases = {'en': [], 'zh': []}

    # ============================================================
    # STEP 1: Fetch daily indices (both languages)
    # ============================================================
    print(f"\nStep 1: Fetching daily indices ({len(dates)} days × 2 languages)...")

    for date_str in tqdm(dates, desc="Fetching indices"):
        year, month, day = date_str[:4], date_str[4:6], date_str[6:8]

        # English index
        en_url = f"{BASE_URL}/{year}{month}/{day}.htm"
        en_html = get_page(en_url)
        if en_html:
            releases = parse_daily_index(en_html, date_str, 'en')
            for r in releases:
                r['priority'] = classify_release(r['title'], 'en')
            all_releases['en'].extend(releases)

        # Chinese index (add 'c' before .htm)
        zh_url = f"{BASE_URL}/{year}{month}/{day}c.htm"
        zh_html = get_page(zh_url)
        if zh_html:
            releases = parse_daily_index(zh_html, date_str, 'zh')
            for r in releases:
                r['priority'] = classify_release(r['title'], 'zh')
            all_releases['zh'].extend(releases)

        time.sleep(0.3)

    print(f"\nFound:")
    print(f"  English: {len(all_releases['en'])} releases ({sum(1 for r in all_releases['en'] if r['priority']=='high')} high, {sum(1 for r in all_releases['en'] if r['priority']=='medium')} medium)")
    print(f"  Chinese: {len(all_releases['zh'])} releases ({sum(1 for r in all_releases['zh'] if r['priority']=='high')} high, {sum(1 for r in all_releases['zh'] if r['priority']=='medium')} medium)")

    # ============================================================
    # STEP 2: Fetch full text of high + medium priority
    # ============================================================
    full_texts = {'en': [], 'zh': []}

    for lang in ['en', 'zh']:
        df_meta = pd.DataFrame(all_releases[lang])
        to_fetch = df_meta[df_meta['priority'].isin(['high', 'medium'])]

        lang_name = 'English' if lang == 'en' else 'Chinese'
        print(f"\nStep 2: Fetching {len(to_fetch)} {lang_name} full texts...")

        folder = f'statements_{lang}'

        for idx, row in tqdm(to_fetch.iterrows(), total=len(to_fetch), desc=f"{lang_name}"):
            text = fetch_full_text(row['url'])
            if text:
                # Re-classify with full text
                priority = classify_release(row['title'], lang, text)
                flags = flag_keywords(text, lang)

                entry = {
                    'date': row['date'],
                    'title': row['title'],
                    'url': row['url'],
                    'priority': priority,
                    'text': text,
                    'word_count': len(text) if lang == 'zh' else len(text.split()),
                    **flags,
                }
                full_texts[lang].append(entry)

                # Save individual file
                safe_title = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', row['title'])[:60]
                fname = f"{folder}/{row['date']}_{safe_title}.txt"
                try:
                    with open(fname, 'w', encoding='utf-8') as f:
                        f.write(f"Date: {row['date']}\n")
                        f.write(f"Title: {row['title']}\n")
                        f.write(f"URL: {row['url']}\n")
                        f.write(f"Language: {lang_name}\n")
                        f.write(f"{'='*60}\n\n")
                        f.write(text)
                except:
                    pass

            time.sleep(0.3)

    # ============================================================
    # STEP 3: Save to Excel (two sheets)
    # ============================================================
    print(f"\nStep 3: Saving to Excel...")

    with pd.ExcelWriter('hksar_bilingual.xlsx', engine='openpyxl') as writer:
        # English sheet
        df_en = pd.DataFrame(full_texts['en'])
        if len(df_en) > 0:
            df_en.to_excel(writer, sheet_name='english', index=False)
            print(f"  English sheet: {len(df_en)} statements")

        # Chinese sheet
        df_zh = pd.DataFrame(full_texts['zh'])
        if len(df_zh) > 0:
            df_zh.to_excel(writer, sheet_name='chinese', index=False)
            print(f"  Chinese sheet: {len(df_zh)} statements")

        # Metadata sheets
        pd.DataFrame(all_releases['en']).to_excel(writer, sheet_name='meta_english', index=False)
        pd.DataFrame(all_releases['zh']).to_excel(writer, sheet_name='meta_chinese', index=False)

    # Also save CSVs for convenience
    if len(df_en) > 0:
        df_en.to_csv('hksar_english_full.csv', index=False, encoding='utf-8-sig')
    if len(df_zh) > 0:
        df_zh.to_csv('hksar_chinese_full.csv', index=False, encoding='utf-8-sig')

    # ============================================================
    # STEP 4: Quick smoking gun scan
    # ============================================================
    print(f"\n{'='*60}")
    print("QUICK SMOKING GUN SCAN")
    print(f"{'='*60}")

    for lang, df in [('English', df_en), ('Chinese', df_zh)]:
        if len(df) == 0:
            continue
        print(f"\n  {lang}:")

        if 'mentions_withdrawal' in df.columns and 'mentions_violence' in df.columns:
            both = df[df['mentions_withdrawal'] & df['mentions_violence']]
            print(f"    Withdrawal + Violence: {len(both)} statements")
            for _, row in both.iterrows():
                print(f"      [{row['date']}] {row['title'][:80]}")

        if 'mentions_stop_violence' in df.columns:
            sv = df[df['mentions_stop_violence'] == True]
            print(f"    Stop-violence/止暴制亂: {len(sv)} statements")

        if 'mentions_rioter' in df.columns:
            ri = df[df['mentions_rioter'] == True]
            print(f"    Rioter/暴徒: {len(ri)} statements")
            if len(ri) > 0:
                first_date = ri['date'].min()
                print(f"      First appearance: {first_date}")

        # New scans
        if 'mentions_national_security' in df.columns:
            ns = df[df['mentions_national_security'] == True]
            print(f"    National Security/國安法: {len(ns)} statements")
            if len(ns) > 0:
                print(f"      First appearance: {ns['date'].min()}")

        if 'mentions_terrorism' in df.columns:
            ter = df[df['mentions_terrorism'] == True]
            print(f"    Terrorism/恐怖主義: {len(ter)} statements")
            if len(ter) > 0:
                print(f"      First appearance: {ter['date'].min()}")

        if 'mentions_gathering_ban' in df.columns:
            gb = df[df['mentions_gathering_ban'] == True]
            print(f"    Gathering ban/限聚令: {len(gb)} statements")

        if 'mentions_arson' in df.columns:
            ar = df[df['mentions_arson'] == True]
            print(f"    Arson/縱火: {len(ar)} statements")

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  hksar_bilingual.xlsx   — Main output (4 sheets)")
    print(f"  hksar_english_full.csv — English full texts")
    print(f"  hksar_chinese_full.csv — Chinese full texts")
    print(f"  statements_en/         — Individual English files")
    print(f"  statements_zh/         — Individual Chinese files")
    print(f"\nUpload hksar_bilingual.xlsx to Claude for analysis!")


if __name__ == '__main__':
    main()
