###############################################################################
# Web Scraping + Google Scholar Tutorial: Python (Penn State Faculty Example)
# Author: Jared Edgerton
# Date: (fill in manually or use datetime.date.today())
#
# This script demonstrates:
#   1) Web scraping a Wikipedia infobox table (warm-up example)
#   2) Web scraping Penn State faculty pages (text + targeted HTML scraping)
#   3) Pulling citation metrics from Google Scholar (via the scholarly package)
#   4) Simple plotting with matplotlib
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - You will see the same steps repeated for each professor so students can
#   follow the logic and edit one piece at a time.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# In a terminal:
#   pip install requests lxml cssselect pandas matplotlib scholarly
#
# Notes:
# - lxml + cssselect lets us use CSS selectors (like "table.infobox") and XPath.
# - Google Scholar scraping can be brittle (rate limits / CAPTCHAs). Consider
#   pre-running tonight and saving outputs if you want a guaranteed live demo.

import re
import requests
import pandas as pd
import matplotlib.pyplot as plt

from lxml import html
from scholarly import scholarly

# Polite headers (good practice for scraping)
HEADERS = {
    "User-Agent": "psu-webscrape-class-demo/1.0 (contact: you@example.com)"
}


# -----------------------------------------------------------------------------
# Part 1B: Hard-code three Penn State faculty (social sciences broadly)
# -----------------------------------------------------------------------------
# These are the three faculty members we will use throughout the script.
# (We will repeat the same scraping steps for each person.)

# -----------------------------------------------------------------------------
# Part 1B: Hard-code ten Penn State faculty (Political Science)
# -----------------------------------------------------------------------------

jwright_name = "Joe Wright"
jwright_dept = "Political Science (College of the Liberal Arts)"
jwright_url  = "https://polisci.la.psu.edu/people/jgw12/"

xcao_name = "Xun Cao"
xcao_dept = "Political Science (College of the Liberal Arts)"
xcao_url  = "https://polisci.la.psu.edu/people/xuc11/"

bdesmarais_name = "Bruce Desmarais"
bdesmarais_dept = "Political Science (College of the Liberal Arts)"
bdesmarais_url  = "https://polisci.la.psu.edu/people/bbd5087/"

jedgerton_name = "Jared Edgerton"
jedgerton_dept = "Political Science (College of the Liberal Arts)"
jedgerton_url  = "https://polisci.la.psu.edu/people/jared-edgerton/"

slinn_name = "Susanne Linn"
slinn_dept = "Political Science (College of the Liberal Arts)"
slinn_url  = "https://polisci.la.psu.edu/people/sld8/"

cloyle_name = "Cyanne Loyle"
cloyle_dept = "Political Science (College of the Liberal Arts)"
cloyle_url  = "https://polisci.la.psu.edu/people/cel5432/"

rmcmanus_name = "Roseanne McManus"
rmcmanus_dept = "Political Science (College of the Liberal Arts)"
rmcmanus_url  = "https://polisci.la.psu.edu/people/rum842/"

bmukherjee_name = "Bumba Mukherjee"
bmukherjee_dept = "Political Science (College of the Liberal Arts)"
bmukherjee_url  = "https://polisci.la.psu.edu/people/sxm73/"

dtavana_name = "Daniel Tavana"
dtavana_dept = "Political Science (College of the Liberal Arts)"
dtavana_url  = "https://polisci.la.psu.edu/people/daniel-tavana/"

vyadav_name = "Vineeta Yadav"
vyadav_dept = "Political Science (College of the Liberal Arts)"
vyadav_url  = "https://polisci.la.psu.edu/people/vuy2/"

# -----------------------------------------------------------------------------
# Step 1: Scrape Joe Wright (one complete example, step-by-step)
# -----------------------------------------------------------------------------
jwright_resp = requests.get(jwright_url, headers=HEADERS)
jwright_page = html.fromstring(jwright_resp.content)

heads = [h.text_content().strip() for h in jwright_page.cssselect("h1, h2, h3, h4, h5, h6")]
print(heads)

print(len(jwright_page.cssselect("main, article, header, nav, footer")))
print(len(jwright_page.cssselect("p")))
print(len(jwright_page.cssselect("a")))
print(len(jwright_page.cssselect("table")))

jwright_main_text = (jwright_page.cssselect("main")[0].text_content().strip() + " ")[:600]
print(jwright_main_text)

jwright_header_text = (jwright_page.cssselect("header")[0].text_content().strip() + " ")[:600]
print(jwright_header_text)

jwright_nav_text = (jwright_page.cssselect("nav")[0].text_content().strip() + " ")[:600]
print(jwright_nav_text)

# Extract "Areas of Interest" (HTML via XPath)
jwright_areas_nodes = jwright_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
jwright_areas = [n.text_content().strip() for n in jwright_areas_nodes]
print(jwright_areas)

jwright_interests = "; ".join(jwright_areas)
jwright_n_interest_items = len(jwright_areas)

jwright_row = {
    "name": jwright_name,
    "department": jwright_dept,
    "url": jwright_url,
    "scraped_interests": jwright_interests,
    "n_interest_items": jwright_n_interest_items,
}
# -----------------------------------------------------------------------------
# Step 2: Scrape Xun Cao
# -----------------------------------------------------------------------------
xcao_resp = requests.get(xcao_url, headers=HEADERS)
xcao_page = html.fromstring(xcao_resp.content)

xcao_areas_nodes = xcao_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
xcao_areas = [n.text_content().strip() for n in xcao_areas_nodes]

xcao_interests = "; ".join(xcao_areas)
xcao_n_interest_items = len(xcao_areas)

xcao_row = {
    "name": xcao_name,
    "department": xcao_dept,
    "url": xcao_url,
    "scraped_interests": xcao_interests,
    "n_interest_items": xcao_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 3: Scrape Bruce Desmarais
# -----------------------------------------------------------------------------
bdesmarais_resp = requests.get(bdesmarais_url, headers=HEADERS)
bdesmarais_page = html.fromstring(bdesmarais_resp.content)

bdesmarais_areas_nodes = bdesmarais_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
bdesmarais_areas = [n.text_content().strip() for n in bdesmarais_areas_nodes]

bdesmarais_interests = "; ".join(bdesmarais_areas)
bdesmarais_n_interest_items = len(bdesmarais_areas)

bdesmarais_row = {
    "name": bdesmarais_name,
    "department": bdesmarais_dept,
    "url": bdesmarais_url,
    "scraped_interests": bdesmarais_interests,
    "n_interest_items": bdesmarais_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 4: Scrape Jared Edgerton
# -----------------------------------------------------------------------------
jedgerton_resp = requests.get(jedgerton_url, headers=HEADERS)
jedgerton_page = html.fromstring(jedgerton_resp.content)

jedgerton_areas_nodes = jedgerton_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
jedgerton_areas = [n.text_content().strip() for n in jedgerton_areas_nodes]

jedgerton_interests = "; ".join(jedgerton_areas)
jedgerton_n_interest_items = len(jedgerton_areas)

jedgerton_row = {
    "name": jedgerton_name,
    "department": jedgerton_dept,
    "url": jedgerton_url,
    "scraped_interests": jedgerton_interests,
    "n_interest_items": jedgerton_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 5: Scrape Susanne Linn
# -----------------------------------------------------------------------------
slinn_resp = requests.get(slinn_url, headers=HEADERS)
slinn_page = html.fromstring(slinn_resp.content)

slinn_areas_nodes = slinn_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
slinn_areas = [n.text_content().strip() for n in slinn_areas_nodes]

slinn_interests = "; ".join(slinn_areas)
slinn_n_interest_items = len(slinn_areas)

slinn_row = {
    "name": slinn_name,
    "department": slinn_dept,
    "url": slinn_url,
    "scraped_interests": slinn_interests,
    "n_interest_items": slinn_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 6: Scrape Cyanne Loyle
# -----------------------------------------------------------------------------
cloyle_resp = requests.get(cloyle_url, headers=HEADERS)
cloyle_page = html.fromstring(cloyle_resp.content)

cloyle_areas_nodes = cloyle_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
cloyle_areas = [n.text_content().strip() for n in cloyle_areas_nodes]

cloyle_interests = "; ".join(cloyle_areas)
cloyle_n_interest_items = len(cloyle_areas)

cloyle_row = {
    "name": cloyle_name,
    "department": cloyle_dept,
    "url": cloyle_url,
    "scraped_interests": cloyle_interests,
    "n_interest_items": cloyle_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 7: Scrape Roseanne McManus
# -----------------------------------------------------------------------------
rmcmanus_resp = requests.get(rmcmanus_url, headers=HEADERS)
rmcmanus_page = html.fromstring(rmcmanus_resp.content)

rmcmanus_areas_nodes = rmcmanus_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
rmcmanus_areas = [n.text_content().strip() for n in rmcmanus_areas_nodes]

rmcmanus_interests = "; ".join(rmcmanus_areas)
rmcmanus_n_interest_items = len(rmcmanus_areas)

rmcmanus_row = {
    "name": rmcmanus_name,
    "department": rmcmanus_dept,
    "url": rmcmanus_url,
    "scraped_interests": rmcmanus_interests,
    "n_interest_items": rmcmanus_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 8: Scrape Bumba Mukherjee
# -----------------------------------------------------------------------------
bmukherjee_resp = requests.get(bmukherjee_url, headers=HEADERS)
bmukherjee_page = html.fromstring(bmukherjee_resp.content)

bmukherjee_areas_nodes = bmukherjee_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
bmukherjee_areas = [n.text_content().strip() for n in bmukherjee_areas_nodes]

bmukherjee_interests = "; ".join(bmukherjee_areas)
bmukherjee_n_interest_items = len(bmukherjee_areas)

bmukherjee_row = {
    "name": bmukherjee_name,
    "department": bmukherjee_dept,
    "url": bmukherjee_url,
    "scraped_interests": bmukherjee_interests,
    "n_interest_items": bmukherjee_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 9: Scrape Daniel Tavana
# -----------------------------------------------------------------------------
dtavana_resp = requests.get(dtavana_url, headers=HEADERS)
dtavana_page = html.fromstring(dtavana_resp.content)

dtavana_areas_nodes = dtavana_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
dtavana_areas = [n.text_content().strip() for n in dtavana_areas_nodes]

dtavana_interests = "; ".join(dtavana_areas)
dtavana_n_interest_items = len(dtavana_areas)

dtavana_row = {
    "name": dtavana_name,
    "department": dtavana_dept,
    "url": dtavana_url,
    "scraped_interests": dtavana_interests,
    "n_interest_items": dtavana_n_interest_items,
}

# -----------------------------------------------------------------------------
# Step 10: Scrape Vineeta Yadav
# -----------------------------------------------------------------------------
vyadav_resp = requests.get(vyadav_url, headers=HEADERS)
vyadav_page = html.fromstring(vyadav_resp.content)

vyadav_areas_nodes = vyadav_page.xpath("//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li")
vyadav_areas = [n.text_content().strip() for n in vyadav_areas_nodes]

vyadav_interests = "; ".join(vyadav_areas)
vyadav_n_interest_items = len(vyadav_areas)

vyadav_row = {
    "name": vyadav_name,
    "department": vyadav_dept,
    "url": vyadav_url,
    "scraped_interests": vyadav_interests,
    "n_interest_items": vyadav_n_interest_items,
}
# -----------------------------------------------------------------------------
# Step 5: Combine the scraped rows into one data frame and inspect
# -----------------------------------------------------------------------------
scraped_profiles = pd.DataFrame([
    jwright_row, xcao_row, bdesmarais_row, jedgerton_row, slinn_row,
    cloyle_row, rmcmanus_row, bmukherjee_row, dtavana_row, vyadav_row
])

print(scraped_profiles)
# -----------------------------------------------------------------------------
# Step 6: Quick plot (interest items captured per faculty member)
# -----------------------------------------------------------------------------
plot_df = scraped_profiles.sort_values("n_interest_items")

plt.figure()
plt.barh(plot_df["name"], plot_df["n_interest_items"])
plt.title("Interest Items Captured from PSU Profile Pages")
plt.xlabel("Number of interest items captured")
plt.ylabel("Faculty member")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Step 7: Word cloud plot (professors' interests)
# -----------------------------------------------------------------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_interest_text = " ".join(
    scraped_profiles["scraped_interests"]
    .fillna("")
    .astype(str)
    .str.replace(";", " ", regex=False)
    .tolist()
).strip()

plt.figure()
wc = WordCloud(width=900, height=500, background_color="white").generate(all_interest_text)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Faculty Research Interests (PSU Political Science)")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Part 2: Pulling Google Scholar Data (Citations Over Time)
# -----------------------------------------------------------------------------
# Goal:
# - For each professor, we will:
#   (1) Define the Google Scholar ID
#   (2) Pull a profile summary
#   (3) Pull publications (and view the first 5)
#   (4) Pull citation history by year
#   (5) Combine all citation histories into one table and plot them

# -----------------------------------------------------------------------------
# Step 1: Hard-code Google Scholar IDs
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Step 1: Hard-code Google Scholar IDs
# -----------------------------------------------------------------------------
jwright_scholar_id    = "DV5ECYgAAAAJ"
jedgerton_scholar_id = "LLcIlUkAAAAJ"
bdesmarais_scholar_id= "fRM8IN4AAAAJ"
cloyle_scholar_id    = "IMUIrJMAAAAJ"
xcao_scholar_id      = "w18ZmkEAAAAJ"
slinn_scholar_id     = "I7Jx1fAAAAAJ"
rmcmanus_scholar_id  = "3xe3Ck4AAAAJ"
bmukherjee_scholar_id= "6sS40fEAAAAJ"
dtavana_scholar_id   = "j2a1_doAAAAJ"
vyadav_scholar_id    = "vGjxl7YAAAAJ"

jwright_name    = "Joe Wright"
jedgerton_name  = "Jared Edgerton"
bdesmarais_name = "Bruce Desmarais"
cloyle_name     = "Cyanne Loyle"
xcao_name       = "Xun Cao"
slinn_name      = "Susanne Linn"
rmcmanus_name   = "Roseanne McManus"
bmukherjee_name = "Bumba Mukherjee"
dtavana_name    = "Daniel Tavana"
vyadav_name     = "Vineeta Yadav"

# -----------------------------------------------------------------------------
# Step 2: Pull Google Scholar profiles (sequentially)
# -----------------------------------------------------------------------------
jwright_profile    = scholarly.fill(scholarly.search_author_id(jwright_scholar_id))
jedgerton_profile  = scholarly.fill(scholarly.search_author_id(jedgerton_scholar_id))
bdesmarais_profile = scholarly.fill(scholarly.search_author_id(bdesmarais_scholar_id))
cloyle_profile     = scholarly.fill(scholarly.search_author_id(cloyle_scholar_id))
xcao_profile       = scholarly.fill(scholarly.search_author_id(xcao_scholar_id))
slinn_profile      = scholarly.fill(scholarly.search_author_id(slinn_scholar_id))
rmcmanus_profile   = scholarly.fill(scholarly.search_author_id(rmcmanus_scholar_id))
bmukherjee_profile = scholarly.fill(scholarly.search_author_id(bmukherjee_scholar_id))
dtavana_profile    = scholarly.fill(scholarly.search_author_id(dtavana_scholar_id))
vyadav_profile     = scholarly.fill(scholarly.search_author_id(vyadav_scholar_id))

print("\n------------------------------")
print("Google Scholar Profile Summaries")
print("------------------------------\n")

print(jwright_name);    print(jwright_profile)
print("\n"+jedgerton_name);  print(jedgerton_profile)
print("\n"+bdesmarais_name); print(bdesmarais_profile)
print("\n"+cloyle_name);     print(cloyle_profile)
print("\n"+xcao_name);       print(xcao_profile)
print("\n"+slinn_name);      print(slinn_profile)
print("\n"+rmcmanus_name);   print(rmcmanus_profile)
print("\n"+bmukherjee_name); print(bmukherjee_profile)
print("\n"+dtavana_name);    print(dtavana_profile)
print("\n"+vyadav_name);     print(vyadav_profile)

# -----------------------------------------------------------------------------
# Step 3: Pull Google Scholar publications (sequentially)
# -----------------------------------------------------------------------------
# Note: scholarly returns a list of publications inside each profile dict.
# We'll convert to a DataFrame and print the first 5 rows.
def pubs_df(profile):
    return pd.DataFrame([
        {
            "title": p.get("bib", {}).get("title"),
            "year": p.get("bib", {}).get("pub_year"),
            "citations": p.get("num_citations"),
        }
        for p in profile.get("publications", [])
    ])

jwright_pubs    = pubs_df(jwright_profile)
jedgerton_pubs  = pubs_df(jedgerton_profile)
bdesmarais_pubs = pubs_df(bdesmarais_profile)
cloyle_pubs     = pubs_df(cloyle_profile)
xcao_pubs       = pubs_df(xcao_profile)
slinn_pubs      = pubs_df(slinn_profile)
rmcmanus_pubs   = pubs_df(rmcmanus_profile)
bmukherjee_pubs = pubs_df(bmukherjee_profile)
dtavana_pubs    = pubs_df(dtavana_profile)
vyadav_pubs     = pubs_df(vyadav_profile)


print("\n------------------------------")
print("Recent Publications (first 5)")
print("------------------------------\n")

print(jwright_name);    print(jwright_pubs.head(5))
print("\n"+jedgerton_name);  print(jedgerton_pubs.head(5))
print("\n"+bdesmarais_name); print(bdesmarais_pubs.head(5))
print("\n"+cloyle_name);     print(cloyle_pubs.head(5))
print("\n"+xcao_name);       print(xcao_pubs.head(5))
print("\n"+slinn_name);      print(slinn_pubs.head(5))
print("\n"+rmcmanus_name);   print(rmcmanus_pubs.head(5))
print("\n"+bmukherjee_name); print(bmukherjee_pubs.head(5))
print("\n"+dtavana_name);    print(dtavana_pubs.head(5))
print("\n"+vyadav_name);     print(vyadav_pubs.head(5))


# -----------------------------------------------------------------------------
# Step 4: Pull citation history (citations by year) and combine
# -----------------------------------------------------------------------------
def cites_df(profile, name):
    d = profile.get("cites_per_year", {})
    df = pd.DataFrame({"year": list(d.keys()), "cites": list(d.values())})
    df["name"] = name
    return df

jwright_ct    = cites_df(jwright_profile, jwright_name)
jedgerton_ct  = cites_df(jedgerton_profile, jedgerton_name)
bdesmarais_ct = cites_df(bdesmarais_profile, bdesmarais_name)
cloyle_ct     = cites_df(cloyle_profile, cloyle_name)
xcao_ct       = cites_df(xcao_profile, xcao_name)
slinn_ct      = cites_df(slinn_profile, slinn_name)
rmcmanus_ct   = cites_df(rmcmanus_profile, rmcmanus_name)
bmukherjee_ct = cites_df(bmukherjee_profile, bmukherjee_name)
dtavana_ct    = cites_df(dtavana_profile, dtavana_name)
vyadav_ct     = cites_df(vyadav_profile, vyadav_name)

citation_df = pd.concat([
    jwright_ct, jedgerton_ct, bdesmarais_ct, cloyle_ct, xcao_ct,
    slinn_ct, rmcmanus_ct, bmukherjee_ct, dtavana_ct, vyadav_ct
], ignore_index=True).sort_values(["name","year"])

print(citation_df.head(10))

# -----------------------------------------------------------------------------
# Step 5: Plot citations over time for each professor
# -----------------------------------------------------------------------------
plt.figure(figsize=(8,6))

plt.plot(jwright_ct["year"], jwright_ct["cites"], marker="o", label=jwright_name)
plt.plot(jedgerton_ct["year"], jedgerton_ct["cites"], marker="o", label=jedgerton_name)
plt.plot(bdesmarais_ct["year"], bdesmarais_ct["cites"], marker="o", label=bdesmarais_name)
plt.plot(cloyle_ct["year"], cloyle_ct["cites"], marker="o", label=cloyle_name)
plt.plot(xcao_ct["year"], xcao_ct["cites"], marker="o", label=xcao_name)
plt.plot(slinn_ct["year"], slinn_ct["cites"], marker="o", label=slinn_name)
plt.plot(rmcmanus_ct["year"], rmcmanus_ct["cites"], marker="o", label=rmcmanus_name)
plt.plot(bmukherjee_ct["year"], bmukherjee_ct["cites"], marker="o", label=bmukherjee_name)
plt.plot(dtavana_ct["year"], dtavana_ct["cites"], marker="o", label=dtavana_name)
plt.plot(vyadav_ct["year"], vyadav_ct["cites"], marker="o", label=vyadav_name)

plt.title("Google Scholar Citation History (Recent Years)")
plt.xlabel("Year")
plt.ylabel("Citations")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------
# Step 6: Median citations per year for each professor
# -----------------------------------------------------------------------------
median_cites = citation_df.groupby("name", as_index=False)["cites"].median()
median_cites = median_cites.rename(columns={"cites": "median_cites"})

print(median_cites)