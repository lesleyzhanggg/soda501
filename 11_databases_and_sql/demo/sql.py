###############################################################################
# SQL Tutorial in Python: SQLite + Campaign Finance (Simulated Data)
# Author: Jared Edgerton
# Date: date.today()
#
# This script demonstrates:
#   1) Creating a local SQLite database (tables + indexes)
#   2) Inserting simulated campaign finance data
#   3) Writing and running SQL queries from Python
#   4) Visualizing query outputs with matplotlib
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are repeated explicitly so students can follow and modify each piece.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# If you do not have these installed, run (in Terminal / Anaconda Prompt):
#   pip install pandas numpy matplotlib
#
# NOTE:
# - sqlite3 is part of Python's standard library (no install needed).

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date, timedelta
import os

os.makedirs("figures", exist_ok=True)

# -----------------------------------------------------------------------------
# Part 1: Create and Populate a Local SQLite Database
# -----------------------------------------------------------------------------
# SQLite is a lightweight database that lives in a single file (e.g., .db).
# We will:
#   1) Connect to a SQLite file
#   2) Drop existing tables
#   3) Create tables
#   4) Insert simulated data
#   5) Add indexes to speed up common queries

# -----------------------------------------------------------------------------
# Step 1: Connect to a database file
# -----------------------------------------------------------------------------
# If the file does not exist, SQLite creates it automatically.
con = sqlite3.connect("campaign_finance.db")
cur = con.cursor()

# -----------------------------------------------------------------------------
# Step 2: Drop tables (so the script can be rerun from scratch)
# -----------------------------------------------------------------------------
# SQL keyword notes:
# - DROP TABLE removes a table
# - IF EXISTS prevents errors if the table does not exist

cur.execute("DROP TABLE IF EXISTS contributions;")
cur.execute("DROP TABLE IF EXISTS contributors;")
cur.execute("DROP TABLE IF EXISTS candidates;")
con.commit()

# -----------------------------------------------------------------------------
# Step 3: Create tables
# -----------------------------------------------------------------------------
# SQL keyword notes:
# - CREATE TABLE creates a new table
# - PRIMARY KEY uniquely identifies each row
# - FOREIGN KEY enforces relationships between tables (relational structure)

cur.execute("""
  CREATE TABLE candidates (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    party TEXT,
    office TEXT,
    winner INTEGER  -- 1 = winner, 0 = not winner (SQLite stores booleans as integers)
  );
""")

cur.execute("""
  CREATE TABLE contributors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    occupation TEXT,
    employer TEXT,
    state TEXT
  );
""")

cur.execute("""
  CREATE TABLE contributions (
    id INTEGER PRIMARY KEY,
    contributor_id INTEGER,
    candidate_id INTEGER,
    amount REAL,
    date TEXT,
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
  );
""")
con.commit()

# -----------------------------------------------------------------------------
# Step 4: Generate simulated data
# -----------------------------------------------------------------------------
# Teaching idea:
# - We are building a "toy" campaign finance database that is large enough
#   to make SQL meaningful, but still simple enough to understand.

np.random.seed(123)

# ---- Candidates table (100 candidates) ----
candidate_ids = np.arange(1, 101)

candidate_names = np.array([f"Candidate {i}" for i in candidate_ids])

candidate_parties = np.random.choice(
    ["Democrat", "Republican", "Independent"],
    size=100,
    replace=True,
    p=[0.45, 0.45, 0.10]
)

candidate_offices = np.random.choice(
    ["Senate", "House", "Governor", "State Senate", "State House"],
    size=100,
    replace=True
)

candidate_winner = np.random.choice(
    [1, 0],
    size=100,
    replace=True,
    p=[0.5, 0.5]
)

candidates = pd.DataFrame({
    "id": candidate_ids,
    "name": candidate_names,
    "party": candidate_parties,
    "office": candidate_offices,
    "winner": candidate_winner
})

# ---- Contributors table (100,000 contributors) ----
contributor_ids = np.arange(1, 100001)

contributor_names = np.array([f"Contributor {i}" for i in contributor_ids])

contributor_occupations = np.random.choice(
    ["Engineer", "Teacher", "Doctor", "Lawyer", "Business Owner"],
    size=100000,
    replace=True
)

contributor_employers = np.array([f"Company {i}" for i in np.random.randint(1, 5001, size=100000)])

state_abb = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

contributor_states = np.random.choice(state_abb, size=100000, replace=True)

contributors = pd.DataFrame({
    "id": contributor_ids,
    "name": contributor_names,
    "occupation": contributor_occupations,
    "employer": contributor_employers,
    "state": contributor_states
})

# ---- Contributions table (1,000,000 contributions) ----
# - amount is log-normal to mimic a skewed donation distribution
# - date is sampled uniformly across 2024
contribution_ids = np.arange(1, 1000001)

contribution_contributor_ids = np.random.randint(1, 100001, size=1000000)
contribution_candidate_ids = np.random.randint(1, 101, size=1000000)

contribution_amounts = np.round(
    np.random.lognormal(mean=np.log(1000), sigma=1, size=1000000),
    2
)

start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)
n_days = (end_date - start_date).days + 1

random_day_offsets = np.random.randint(0, n_days, size=1000000)
contribution_dates = np.array([(start_date + timedelta(days=int(d))).isoformat() for d in random_day_offsets])

contributions = pd.DataFrame({
    "id": contribution_ids,
    "contributor_id": contribution_contributor_ids,
    "candidate_id": contribution_candidate_ids,
    "amount": contribution_amounts,
    "date": contribution_dates
})

# -----------------------------------------------------------------------------
# Step 5: Insert data into the database
# -----------------------------------------------------------------------------
# pandas.DataFrame.to_sql() writes a DataFrame into a database table.
# chunksize helps avoid huge single inserts.

candidates.to_sql("candidates", con, if_exists="append", index=False, chunksize=5000)
contributors.to_sql("contributors", con, if_exists="append", index=False, chunksize=5000)
contributions.to_sql("contributions", con, if_exists="append", index=False, chunksize=5000)
con.commit()

# -----------------------------------------------------------------------------
# Step 6: Create indexes (performance optimization)
# -----------------------------------------------------------------------------
# Indexes speed up queries that filter/join on these columns.
# SQL keyword notes:
# - CREATE INDEX builds an index structure for faster lookups
# - IF NOT EXISTS prevents errors if an index already exists

cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_contributor_id ON contributions (contributor_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_candidate_id   ON contributions (candidate_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_amount         ON contributions (amount);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_date           ON contributions (date);")
con.commit()

# -----------------------------------------------------------------------------
# Step 7: Quick sanity checks (counts + small samples)
# -----------------------------------------------------------------------------
print("\n------------------------------")
print("Sanity checks: table sizes")
print("------------------------------")

print(pd.read_sql_query("SELECT COUNT(*) AS n_candidates FROM candidates;", con))
print(pd.read_sql_query("SELECT COUNT(*) AS n_contributors FROM contributors;", con))
print(pd.read_sql_query("SELECT COUNT(*) AS n_contributions FROM contributions;", con))

print("\n------------------------------")
print("Sanity checks: sample rows")
print("------------------------------")

print(pd.read_sql_query("SELECT * FROM candidates LIMIT 5;", con))

print(pd.read_sql_query("""
  SELECT
    co.id,
    co.candidate_id,
    ca.name AS candidate_name,
    co.amount,
    co.date
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  LIMIT 5;
""", con))

# -----------------------------------------------------------------------------
# Question 4: Schema inspection (PRAGMA table_info)
# -----------------------------------------------------------------------------
print("\n------------------------------")
print("Schema: candidates")
print("------------------------------")
print(pd.read_sql_query("PRAGMA table_info(candidates);", con))

print("\n------------------------------")
print("Schema: contributors")
print("------------------------------")
print(pd.read_sql_query("PRAGMA table_info(contributors);", con))

print("\n------------------------------")
print("Schema: contributions")
print("------------------------------")
print(pd.read_sql_query("PRAGMA table_info(contributions);", con))

# -----------------------------------------------------------------------------
# Part 2: SQL Queries for Analysis
# -----------------------------------------------------------------------------
# In each example below, we:
#   (1) Write a SQL query as a string
#   (2) Run it with pandas.read_sql_query(query, con)
#   (3) Print or plot the result
#
# SQL syntax checklist you’ll see repeatedly:
# - SELECT: which columns to return
# - FROM: which table to start from
# - JOIN ... ON: how to combine tables
# - WHERE: filter rows
# - GROUP BY: aggregate by groups
# - ORDER BY: sort results
# - LIMIT: keep only the top rows

# -----------------------------------------------------------------------------
# Query 1: Top 10 occupations by total contribution amount
# -----------------------------------------------------------------------------
# Concepts:
# - SUM() aggregates amounts by occupation
# - COUNT(*) counts rows per group
# - GROUP BY creates groups
# - ORDER BY sorts totals descending

query_1 = """
  SELECT
    c.occupation,
    SUM(co.amount) AS total_amount,
    COUNT(*) AS num_contributions
  FROM contributors c
  JOIN contributions co
    ON c.id = co.contributor_id
  GROUP BY c.occupation
  ORDER BY total_amount DESC
  LIMIT 10;
"""
top_occupations = pd.read_sql_query(query_1, con)

print("\n------------------------------")
print("Top 10 occupations by total contribution amount")
print("------------------------------")
print(top_occupations)

# -----------------------------------------------------------------------------
# Query 2: Percentage of contributions by party for donations > $1000
# -----------------------------------------------------------------------------
# Concepts:
# - WHERE filters to only contributions > 1000
# - subquery (SELECT SUM(...)) computes the total for the denominator
# - grouping by party creates one row per party

query_2 = """
  SELECT
    ca.party,
    SUM(co.amount) AS total_amount,
    SUM(co.amount) * 100.0 / (
      SELECT SUM(amount)
      FROM contributions
      WHERE amount > 1000
    ) AS percentage
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  WHERE co.amount > 1000
  GROUP BY ca.party;
"""
party_pct = pd.read_sql_query(query_2, con)

print("\n------------------------------")
print("Percent of total contributions by party (amount > $1000)")
print("------------------------------")
print(party_pct)

# -----------------------------------------------------------------------------
# Query 3: Candidates receiving contributions from the most distinct states
# -----------------------------------------------------------------------------
# Concepts:
# - COUNT(DISTINCT state) counts unique states
# - join chain: candidates -> contributions -> contributors
# - ORDER BY sorts by (num_states, then contribution_count)

query_3 = """
  SELECT
    ca.id AS candidate_id,
    ca.name,
    ca.party,
    COUNT(DISTINCT c.state) AS num_states,
    COUNT(co.id) AS contribution_count
  FROM candidates ca
  JOIN contributions co
    ON ca.id = co.candidate_id
  JOIN contributors c
    ON co.contributor_id = c.id
  GROUP BY ca.id, ca.name, ca.party
  ORDER BY num_states DESC, contribution_count DESC
  LIMIT 5;
"""
candidates_most_states = pd.read_sql_query(query_3, con)

print("\n------------------------------")
print("Candidates with contributions from the most distinct states")
print("------------------------------")
print(candidates_most_states)

# -----------------------------------------------------------------------------
# Query 4: Contributors donating to multiple parties (cross-party donors)
# -----------------------------------------------------------------------------
# Concepts:
# - HAVING filters groups after aggregation
# - GROUP_CONCAT summarizes unique parties into one string (SQLite feature)
# - CASE WHEN creates party-specific sums

query_4 = """
  SELECT
    c.name,
    GROUP_CONCAT(DISTINCT ca.party) AS parties,
    SUM(CASE WHEN ca.party = 'Democrat' THEN co.amount ELSE 0 END) AS dem_amount,
    SUM(CASE WHEN ca.party = 'Republican' THEN co.amount ELSE 0 END) AS rep_amount
  FROM contributors c
  JOIN contributions co
    ON c.id = co.contributor_id
  JOIN candidates ca
    ON co.candidate_id = ca.id
  GROUP BY c.id
  HAVING COUNT(DISTINCT ca.party) > 1
  LIMIT 20;
"""
cross_party = pd.read_sql_query(query_4, con)

print("\n------------------------------")
print("Cross-party contributors (sample)")
print("------------------------------")
print(cross_party)

# -----------------------------------------------------------------------------
# Query 5: Moving average of contribution amounts for top 3 candidates
# -----------------------------------------------------------------------------
# Concepts:
# - WITH ... AS creates CTEs (Common Table Expressions)
# - window functions use OVER (PARTITION BY ... ORDER BY ...)
# - ROWS BETWEEN 29 PRECEDING AND CURRENT ROW defines a rolling window
# - This query:
#   (1) builds cumulative totals per candidate
#   (2) computes 30-day moving average of contribution amounts by candidate
#   (3) keeps only the top 3 candidates by cumulative amount

query_5 = """
  WITH ranked_contributions AS (
    SELECT
      ca.id,
      ca.name,
      co.date,
      co.amount,
      SUM(co.amount) OVER (
        PARTITION BY ca.id
        ORDER BY co.date
      ) AS cumulative_amount
    FROM candidates ca
    JOIN contributions co
      ON ca.id = co.candidate_id
  ),
  windowed_avg AS (
    SELECT
      id,
      name,
      date,
      amount,
      AVG(amount) OVER (
        PARTITION BY id
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
      ) AS moving_avg
    FROM ranked_contributions
  )
  SELECT
    id,
    name,
    date,
    amount,
    moving_avg
  FROM windowed_avg
  WHERE id IN (
    SELECT id
    FROM ranked_contributions
    GROUP BY id
    ORDER BY MAX(cumulative_amount) DESC
    LIMIT 3
  )
  ORDER BY id, date
  LIMIT 100;
"""
moving_avg = pd.read_sql_query(query_5, con)

print("\n------------------------------")
print("Moving average contributions (top 3 candidates, sample rows)")
print("------------------------------")
print(moving_avg)

# -----------------------------------------------------------------------------
# Query 6: Visualize total contributions by party
# -----------------------------------------------------------------------------
# Concepts:
# - This query aggregates contribution totals by party (SUM + GROUP BY)
# - Then we plot with matplotlib

query_6 = """
  SELECT
    ca.party,
    SUM(co.amount) AS total_amount
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  GROUP BY ca.party;
"""
party_totals = pd.read_sql_query(query_6, con)

print("\n------------------------------")
print("Total contributions by party")
print("------------------------------")
print(party_totals)

# Plot: total amount by party
plt.figure()
plt.bar(party_totals["party"], party_totals["total_amount"])
plt.title("Total Contributions by Party")
plt.xlabel("Party")
plt.ylabel("Total Amount ($)")
plt.tight_layout()
plt.savefig("figures/contributions_by_party.png", dpi=150)
plt.show()

print("\nSaved plot: contributions_by_party.png")

# =============================================================================
# PROBLEM SET ANSWERS: Questions 4, 5, 6
# =============================================================================

# -----------------------------------------------------------------------------
# Question 5: Join + aggregation (required query)
# -----------------------------------------------------------------------------
# Join contributions to candidates, filter amount > 1000,
# output: party, total_amount, num_contributions

query_q5 = """
  SELECT
    ca.party,
    SUM(co.amount) AS total_amount,
    COUNT(*) AS num_contributions
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  WHERE co.amount > 1000
  GROUP BY ca.party
  ORDER BY total_amount DESC;
"""
q5_result = pd.read_sql_query(query_q5, con)

print("\n------------------------------")
print("Q5: Total contributions by party (amount > $1000)")
print("------------------------------")
print(q5_result)

# Q5 bar plot
plt.figure()
plt.bar(q5_result["party"], q5_result["total_amount"],
        color=["#534AB7", "#1D9E75", "#D85A30"])
plt.title("Total contributions by party (amount > $1000)")
plt.xlabel("Party")
plt.ylabel("Total Amount ($)")
plt.tight_layout()
plt.savefig("figures/q5_contributions_by_party_gt1000.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# Question 6: Indexes + EXPLAIN QUERY PLAN
# -----------------------------------------------------------------------------

# 6A: Check which indexes exist on contributions
print("\n------------------------------")
print("Q6: Existing indexes on the database")
print("------------------------------")
indexes = pd.read_sql_query("""
  SELECT type, name, tbl_name, sql
  FROM sqlite_master
  WHERE type = 'index';
""", con)
print(indexes)

# 6B: Run EXPLAIN QUERY PLAN on a query that filters by candidate_id and amount
print("\n------------------------------")
print("Q6: EXPLAIN QUERY PLAN (filter by candidate_id and amount > 1000)")
print("------------------------------")
explain_result = pd.read_sql_query("""
  EXPLAIN QUERY PLAN
  SELECT candidate_id, SUM(amount) AS total
  FROM contributions
  WHERE candidate_id = 1 AND amount > 1000
  GROUP BY candidate_id;
""", con)
print(explain_result)

# 6C: EXPLAIN QUERY PLAN on the Q5 join query for comparison
print("\n------------------------------")
print("Q6: EXPLAIN QUERY PLAN (Q5 join query)")
print("------------------------------")
explain_join = pd.read_sql_query("""
  EXPLAIN QUERY PLAN
  SELECT
    ca.party,
    SUM(co.amount) AS total_amount,
    COUNT(*) AS num_contributions
  FROM contributions co
  JOIN candidates ca
    ON co.candidate_id = ca.id
  WHERE co.amount > 1000
  GROUP BY ca.party;
""", con)
print(explain_join)

# -----------------------------------------------------------------------------
# Close the database connection
# -----------------------------------------------------------------------------
con.close()


