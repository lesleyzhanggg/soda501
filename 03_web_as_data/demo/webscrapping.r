###############################################################################
# Web Scraping + Google Scholar Tutorial: R (Penn State Faculty Example)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# This script demonstrates:
#   1) Web scraping a Wikipedia infobox table with rvest (warm-up example)
#   2) Web scraping Penn State faculty pages with rvest
#   3) Pulling citation metrics from Google Scholar with the scholar package
#   4) Simple plotting with ggplot2
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

# install.packages(c("rvest", "dplyr", "ggplot2", "scholar", "stringr", "tibble"))
library(rvest)
library(dplyr)
library(ggplot2)
library(scholar)
library(stringr)
library(tibble)

# -----------------------------------------------------------------------------
# Part 1: Web Scraping (Wikipedia Warm-up + Penn State Faculty Pages)
# -----------------------------------------------------------------------------
# We will do web scraping in two stages:
#
# A) Wikipedia warm-up (table scraping)
# - Read a Wikipedia page
# - Extract the "infobox" table
# - Clean it into a Key/Value table
#
# B) Penn State faculty pages (text + targeted HTML scraping)
# - Read each faculty member’s PSU profile page
# - Pull the page text and extract:
#     * A title line (regex)
#     * A PSU email address (regex)
# - Pull structured items like:
#     * "Areas of Interest" (HTML nodes)
#     * "Research Interests" (HTML nodes)
#
# Note:
# - Department websites do not always have the same structure.
# - Some pages may have "Areas of Interest" while others have "Research Interests".

# -----------------------------------------------------------------------------
# Part 1A: Wikipedia Warm-up (Scraping an Infobox Table)
# -----------------------------------------------------------------------------
# In Wikipedia, many biography pages include an "infobox" on the right side.
# That infobox is typically stored as an HTML table with class "infobox".

# URL of the Wikipedia page
# URL of the Wikipedia page
url <- "https://en.wikipedia.org/wiki/Donald_Trump"

# Read the HTML content
page <- read_html(url)

###############################################################################
# 2) Identify the HTML “thing” you want
#
# Wikipedia biographies commonly have an "infobox": a table on the right side
# that summarizes key facts (Born, Education, Occupation, etc.).
#
# In the HTML, this is usually a <table> element with class="infobox".
#
# CSS selector reminder:
#   - "table.infobox" means: a <table> tag with class "infobox"
#   - "." indicates a class; "#" indicates an id
###############################################################################

# Extract the infobox table (usually the first one on the page)
infobox_nodes <- page %>% html_elements("table.infobox")

# (Optional safety check for teaching: if none found, you’d stop or change strategy)
# length(infobox_nodes)

infobox_node <- infobox_nodes[[1]]

###############################################################################
# 3) Convert the HTML element to a data structure
#
# html_table() converts an HTML <table> into an R data frame-like object.
# fill = TRUE helps because infobox tables often have merged cells / uneven rows.
###############################################################################
infobox_raw <- infobox_node %>%
  html_table(fill = TRUE)

###############################################################################
# 4) Standardize column names (so cleaning code is consistent)
#
# Infobox tables usually come out as 2 columns:
#   left column  = "label" (e.g., "Born")
#   right column = "value" (e.g., "January 1, 19xx ...")
#
# But sometimes rows span columns, and the parsed table can look messy.
###############################################################################
colnames(infobox_raw) <- paste0("X", seq_len(ncol(infobox_raw)))

###############################################################################
# 5) Clean into "key-value" (tidy) format
#
# Goal: one row per field, with:
#   field = label text
#   value = value text
#
# Notes:
# - filter() removes rows that aren’t label/value
# - str_squish() collapses repeated whitespace + trims ends
###############################################################################
infobox_kv <- infobox_raw %>%
  filter(!is.na(X1), !is.na(X2), X1 != "", X2 != "") %>%
  transmute(
    field = str_squish(as.character(X1)),
    value = str_squish(as.character(X2))
  )

View(infobox_kv)

# -----------------------------------------------------------------------------
# Part 1B: Hard-code three Penn State faculty (social sciences broadly)
# -----------------------------------------------------------------------------
# These are the three faculty members we will use throughout the script.
# (We will repeat the same scraping steps for each person.)

matt_name <- "Matt Golder"
matt_dept <- "Political Science (College of the Liberal Arts)"
matt_url  <- "https://polisci.la.psu.edu/people/mrg19/"

sona_name <- "Sona N. Golder"
sona_dept <- "Political Science (College of the Liberal Arts)"
sona_url  <- "https://polisci.la.psu.edu/people/sng11/"

derek_name <- "Derek Kreager"
derek_dept <- "Sociology & Criminology (College of the Liberal Arts)"
derek_url  <- "https://sociology.la.psu.edu/people/derek-kreager/"

# -----------------------------------------------------------------------------
# Step 1: Scrape Matt Golder (one complete example, step-by-step)
# -----------------------------------------------------------------------------
# 1) Read the PSU profile page
matt_page <- read_html(matt_url)

# HTML headings run from <h1> through <h6> (six levels total), and they indicate document structure:
#   h1 = page title (top-level; usually one per page)
#   h2 = major section
#   h3 = subsection
#   h4 = sub-subsection
#   h5 = very fine-grained subsection (rare in practice)
#   h6 = smallest heading level (very rare)
# In practice, most modern pages rarely use h5/h6; you usually see h1–h3 (sometimes h4).
# For scraping, the *heading text* ("Research", "Publications", etc.) often matters more than the level,
# so it’s common to search across multiple heading levels when mapping the page structure.

heads <- matt_page %>%
  html_elements("h1, h2, h3, h4") %>%
  html_text(trim = TRUE)

heads

matt_page %>% html_elements("main, article, header, nav, footer") %>% length()
matt_page %>% html_elements("p") %>% length()
matt_page %>% html_elements("a") %>% length()
matt_page %>% html_elements("table") %>% length()

matt_page %>%
  html_element("main") %>%
  html_text(trim = TRUE)

matt_page %>%
  html_element("header") %>%
  html_text(trim = TRUE)

matt_page %>%
  html_element("nav") %>%
  html_text(trim = TRUE)

# 2) Pull the full page text (useful for regex extraction)
matt_text <- matt_page %>%
  html_element("body") %>%
  html_text(trim = TRUE)

# 3) Extract a job title line (regex)
# This pattern tries to capture a chunk of text containing "Professor ...".
matt_title <- str_extract(
  matt_text,
  "(Distinguished|Liberal Arts|Roy C\\.|Arnold S\\.|James P\\.)?\\s*(Associate\\s+)?Professor[^\\n\\r]{0,120}"
)
print(matt_title)
matt_title <- trimws(matt_title)

# 4) Extract a PSU email address (regex)
matt_email <- str_extract(matt_text, "[A-Za-z0-9._%+-]+@psu\\.edu")

# 5) Extract "Areas of Interest" (HTML)
matt_areas <- matt_page %>%
  html_elements(xpath = "//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li") %>%
  html_text(trim = TRUE)

# 6) Extract "Bio" (HTML)
# (Some sites use h2, others use h3 — we grab both without deciding which is “right”.)
matt_bio <- matt_page %>%
  html_elements(xpath = paste0(
    "//h2[normalize-space()='Professional Bio']/following-sibling::*[1]",
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]"
  )) %>%
  html_text(trim = TRUE)

# 7) Combine whatever we found into one string (semicolon-separated)
matt_interests <- paste(c(matt_areas), collapse = "; ")

# 8) Count how many interest items we captured
matt_n_interest_items <- length(c(matt_areas))

# 9) Store results in a single row (tibble)
matt_row <- tibble(
  name = matt_name,
  department = matt_dept,
  url = matt_url,
  scraped_title = matt_title,
  scraped_email = matt_email,
  scraped_interests = matt_interests,
  n_interest_items = matt_n_interest_items,
  bio = matt_bio
)

# -----------------------------------------------------------------------------
# Step 2: Scrape Sona N. Golder (repeat the same workflow)
# -----------------------------------------------------------------------------
sona_page <- read_html(sona_url)

sona_text <- sona_page %>%
  html_element("body") %>%
  html_text(trim = TRUE)

sona_title <- str_extract(
  sona_text,
  "(Distinguished|Liberal Arts|Roy C\\.|Arnold S\\.|James P\\.)?\\s*(Associate\\s+)?Professor[^\\n\\r]{0,120}"
)

sona_title <- trimws(sona_title)

sona_email <- str_extract(sona_text, "[A-Za-z0-9._%+-]+@psu\\.edu")

sona_areas <- sona_page %>%
  html_elements(xpath = "//h2[normalize-space()='Areas of Interest']/following-sibling::ul[1]/li") %>%
  html_text(trim = TRUE)

sona_bio <- sona_page %>%
  html_elements(xpath = paste0(
    "//h2[normalize-space()='Professional Bio']/following-sibling::*[1]",
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]"
  )) %>%
  html_text(trim = TRUE)

sona_interests <- paste(c(sona_areas), collapse = "; ")
sona_n_interest_items <- length(c(sona_areas))

sona_row <- tibble(
  name = sona_name,
  department = sona_dept,
  url = sona_url,
  scraped_title = sona_title,
  scraped_email = sona_email,
  scraped_interests = sona_interests,
  n_interest_items = sona_n_interest_items,
  bio = sona_bio
)

# -----------------------------------------------------------------------------
# Step 3: Scrape Derek Kreager (repeat the same workflow)
# -----------------------------------------------------------------------------
derek_page <- read_html(derek_url)

derek_text <- derek_page %>%
  html_element("body") %>%
  html_text(trim = TRUE)

derek_title <- str_extract(
  derek_text,
  "(Distinguished|Liberal Arts|Roy C\\.|Arnold S\\.|James P\\.)?\\s*(Associate\\s+)?Professor[^\\n\\r]{0,120}"
)

derek_title <- trimws(derek_title)

derek_email <- str_extract(derek_text, "[A-Za-z0-9._%+-]+@psu\\.edu")

derek_areas <- derek_page %>%
  html_elements(xpath = "//h2[normalize-space()='Research Interests']/following-sibling::ul[1]/li") %>%
  html_text(trim = TRUE)

derek_bio <- derek_page %>%
  html_elements(xpath = paste0(
    "//h2[normalize-space()='Professional Bio']/following-sibling::*[1]",
    " | //h3[normalize-space()='Research Interests']/following-sibling::*[1]"
  )) %>%
  html_text(trim = TRUE)

derek_interests <- paste(c(derek_areas), collapse = "; ")
derek_n_interest_items <- length(derek_areas)

derek_row <- tibble(
  name = derek_name,
  department = derek_dept,
  url = derek_url,
  scraped_title = derek_title,
  scraped_email = derek_email,
  scraped_interests = derek_interests,
  n_interest_items = derek_n_interest_items,
  bio = derek_bio[2]
)

# -----------------------------------------------------------------------------
# Step 5: Combine the scraped rows into one data frame and inspect
# -----------------------------------------------------------------------------
scraped_profiles <- bind_rows(matt_row, sona_row, derek_row)

# Print the scraped data table
print(scraped_profiles)

# -----------------------------------------------------------------------------
# Step 6: Quick plot (interest items captured per faculty member)
# -----------------------------------------------------------------------------
ggplot(scraped_profiles, aes(x = reorder(name, n_interest_items), y = n_interest_items)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Interest Items Captured from PSU Profile Pages",
    x = "Faculty member",
    y = "Number of interest items captured"
  )

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
matt_scholar_id   <- "yPbxmSwAAAAJ"
sona_scholar_id   <- "Cuz1fTcAAAAJ"
derek_scholar_id  <- "9c6_ChYAAAAJ"

# -----------------------------------------------------------------------------
# Step 2: Pull Google Scholar profiles (sequentially)
# -----------------------------------------------------------------------------
matt_profile <- get_profile(matt_scholar_id)
sona_profile <- get_profile(sona_scholar_id)
derek_profile <- get_profile(derek_scholar_id)

cat("\n------------------------------\n")
cat("Google Scholar Profile Summaries\n")
cat("------------------------------\n")

cat("\n", matt_name, "\n", sep = "")
print(matt_profile)

cat("\n", sona_name, "\n", sep = "")
print(sona_profile)

cat("\n", derek_name, "\n", sep = "")
print(derek_profile)

# -----------------------------------------------------------------------------
# Step 3: Pull Google Scholar publications (sequentially)
# -----------------------------------------------------------------------------
matt_pubs <- get_publications(matt_scholar_id)
sona_pubs <- get_publications(sona_scholar_id)
derek_pubs <- get_publications(derek_scholar_id)

cat("\n------------------------------\n")
cat("Recent Publications (first 5)\n")
cat("------------------------------\n")

cat("\n", matt_name, "\n", sep = "")
print(head(matt_pubs, 5))

cat("\n", sona_name, "\n", sep = "")
print(head(sona_pubs, 5))

cat("\n", derek_name, "\n", sep = "")
print(head(derek_pubs, 5))

# -----------------------------------------------------------------------------
# Step 4: Pull citation history (citations by year) and combine
# -----------------------------------------------------------------------------
matt_ct <- get_citation_history(matt_scholar_id) %>% mutate(name = matt_name)
sona_ct <- get_citation_history(sona_scholar_id) %>% mutate(name = sona_name)
derek_ct <- get_citation_history(derek_scholar_id) %>% mutate(name = derek_name)

citation_df <- bind_rows(matt_ct, sona_ct, derek_ct)

# Print the combined citation data
print(head(citation_df, 10))

# -----------------------------------------------------------------------------
# Step 5: Plot citations over time for each professor
# -----------------------------------------------------------------------------
ggplot(citation_df, aes(x = year, y = cites, color = name)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Google Scholar Citation History (Recent Years)",
    x = "Year",
    y = "Citations",
    color = "Faculty"
  )

# -----------------------------------------------------------------------------
# Step 6: Median citations per year for each professor
# -----------------------------------------------------------------------------
median_cites <- citation_df %>%
  group_by(name) %>%
  summarize(median_cites = median(cites, na.rm = TRUE), .groups = "drop")

print(median_cites)
