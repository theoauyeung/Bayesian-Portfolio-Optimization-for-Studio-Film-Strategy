import pandas as pd
import ast
import os

DATA_DIR = "/Users/theoauyeung/Documents/Projects/425 Final Project/data/raw"
OUT_DIR  = "/Users/theoauyeung/Documents/Projects/425 Final Project/data/cleaned"

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading files...")
meta = pd.read_csv(os.path.join(DATA_DIR, "movies_metadata.csv"), low_memory=False)
kw   = pd.read_csv(os.path.join(DATA_DIR, "keywords.csv"))
print(f"  metadata: {meta.shape}  |  keywords: {kw.shape}")

# ── 2. Basic filters ──────────────────────────────────────────────────────────
meta = meta[meta["status"] == "Released"].copy()
meta = meta[meta["original_language"] == "en"].copy()       # English-language only
meta["release_date"] = pd.to_datetime(meta["release_date"], errors="coerce")
meta = meta[meta["release_date"].notna()].copy()
meta = meta[meta["release_date"].dt.year >= 1980].copy()    # modern era only
meta = meta[meta["runtime"] >= 60].copy()                   # feature films only
meta = meta[meta["vote_count"] >= 50].copy()                # enough ratings for quality signal
print(f"  after basic filters: {meta.shape}")

# ── 3. Budget / Revenue / ROI ─────────────────────────────────────────────────
meta["budget"]  = pd.to_numeric(meta["budget"],  errors="coerce")
meta["revenue"] = pd.to_numeric(meta["revenue"], errors="coerce")
meta = meta[(meta["budget"] > 0) & (meta["revenue"] > 0)].copy()

# Inflation-adjust to 2017 USD using a simple CPI approximation
# CPI index anchored to 2017; multipliers per decade are rough but standard
cpi = {1980: 2.95, 1981: 2.74, 1982: 2.58, 1983: 2.50, 1984: 2.40,
       1985: 2.32, 1986: 2.27, 1987: 2.19, 1988: 2.10, 1989: 2.00,
       1990: 1.90, 1991: 1.82, 1992: 1.77, 1993: 1.72, 1994: 1.68,
       1995: 1.63, 1996: 1.58, 1997: 1.55, 1998: 1.52, 1999: 1.49,
       2000: 1.44, 2001: 1.40, 2002: 1.37, 2003: 1.34, 2004: 1.30,
       2005: 1.26, 2006: 1.22, 2007: 1.18, 2008: 1.13, 2009: 1.13,
       2010: 1.11, 2011: 1.07, 2012: 1.04, 2013: 1.02, 2014: 1.01,
       2015: 1.01, 2016: 1.00, 2017: 1.00}

meta["year"] = meta["release_date"].dt.year
meta["cpi_mult"] = meta["year"].map(cpi).fillna(1.0)
meta["budget_adj"]  = meta["budget"]  * meta["cpi_mult"]
meta["revenue_adj"] = meta["revenue"] * meta["cpi_mult"]
meta["roi"] = (meta["revenue_adj"] - meta["budget_adj"]) / meta["budget_adj"]

# Winsorise ROI at 1st / 99th percentile to limit outlier pull
lo, hi = meta["roi"].quantile(0.01), meta["roi"].quantile(0.99)
meta["roi"] = meta["roi"].clip(lo, hi)
print(f"  after budget/revenue filter: {meta.shape}")
print(f"  ROI range after winsorising: [{lo:.2f}, {hi:.2f}]")

# ── 4. Parse JSON fields ───────────────────────────────────────────────────────
def parse_json_names(val, n=1):
    """Return list of 'name' values from a JSON-like string field."""
    try:
        items = ast.literal_eval(val)
        names = [d["name"] for d in items if isinstance(d, dict) and "name" in d]
        return names[:n] if n else names
    except Exception:
        return []

meta["genres_list"]    = meta["genres"].apply(lambda x: parse_json_names(x, n=0))
meta["genre_primary"]  = meta["genres"].apply(lambda x: parse_json_names(x, n=1))
meta["genre_primary"]  = meta["genre_primary"].apply(lambda x: x[0] if x else "Unknown")

meta["studios_list"]   = meta["production_companies"].apply(lambda x: parse_json_names(x, n=0))
meta["studio_primary"] = meta["production_companies"].apply(lambda x: parse_json_names(x, n=1))
meta["studio_primary"] = meta["studio_primary"].apply(lambda x: x[0] if x else "Unknown")

# ── 5. Studio tier mapping ────────────────────────────────────────────────────
# Map to major studio groups; everything else = Independent
STUDIO_MAP = {
    # Disney family
    "Walt Disney Pictures": "Disney",
    "Walt Disney Productions": "Disney",
    "Pixar Animation Studios": "Disney",
    "Marvel Studios": "Disney",
    "Lucasfilm": "Disney",
    "Touchstone Pictures": "Disney",
    "Hollywood Pictures": "Disney",
    "Miramax Films": "Disney",           # Disney-owned 1993–2010
    "Miramax": "Disney",
    "Buena Vista Pictures": "Disney",
    # Universal family
    "Universal Pictures": "Universal",
    "Universal Studios": "Universal",
    "Illumination Entertainment": "Universal",
    "DreamWorks Animation": "Universal",
    "Amblin Entertainment": "Universal",
    "Working Title Films": "Universal",
    # Warner Bros. family
    "Warner Bros.": "Warner Bros.",
    "New Line Cinema": "Warner Bros.",
    "Castle Rock Entertainment": "Warner Bros.",
    "Fine Line Features": "Warner Bros.",
    "Village Roadshow Pictures": "Warner Bros.",
    "DC Comics": "Warner Bros.",
    # Sony / Columbia
    "Columbia Pictures": "Sony",
    "Columbia Pictures Corporation": "Sony",
    "Sony Pictures": "Sony",
    "Sony Pictures Classics": "Sony",
    "TriStar Pictures": "Sony",
    "Screen Gems": "Sony",
    # Paramount
    "Paramount Pictures": "Paramount",
    "DreamWorks Pictures": "Paramount",
    "DreamWorks SKG": "Paramount",
    "Skydance Media": "Paramount",
    "Spyglass Entertainment": "Paramount",
    # Fox
    "20th Century Fox": "Fox",
    "Twentieth Century Fox Film Corporation": "Fox",
    "Fox Searchlight Pictures": "Fox",
    "Fox 2000 Pictures": "Fox",
    "Regency Enterprises": "Fox",         # long-term Fox co-production partner
    # Lionsgate
    "Lionsgate": "Lionsgate",
    "Lions Gate Films": "Lionsgate",
    "Summit Entertainment": "Lionsgate",
    "Dimension Films": "Lionsgate",       # Weinstein/Dimension folded into Lionsgate era
    # MGM
    "Metro-Goldwyn-Mayer (MGM)": "MGM",
    "Metro-Goldwyn-Mayer": "MGM",
    "United Artists": "MGM",
    "Orion Pictures": "MGM",
    "Eon Productions": "MGM",             # James Bond — distributed by MGM/UA
    # A24
    "A24": "A24",
}

meta["studio_group"] = meta["studio_primary"].map(STUDIO_MAP).fillna("Independent")

# ── 5b. Studio label — subsidiary identity for meaningful sub-studios ─────────
# Keep specific name for studios with 20+ films; collapse the rest into
# "{studio_group} - Other" so the variable stays model-ready (~30 levels)
# Normalize variant names to a canonical label before frequency thresholding
STUDIO_CANONICAL = {
    "Columbia Pictures Corporation": "Columbia Pictures",
    "Lions Gate Films": "Lionsgate",
    "Metro-Goldwyn-Mayer": "Metro-Goldwyn-Mayer (MGM)",
    "Twentieth Century Fox Film Corporation": "20th Century Fox",
}
meta["studio_primary"] = meta["studio_primary"].replace(STUDIO_CANONICAL)

MIN_FILMS = 20
studio_counts = meta["studio_primary"].value_counts()
frequent_studios = set(studio_counts[studio_counts >= MIN_FILMS].index)

def studio_label(row):
    if row["studio_primary"] in frequent_studios:
        return row["studio_primary"]
    return f"{row['studio_group']} - Other"

meta["studio_label"] = meta.apply(studio_label, axis=1)

# ── 6. Budget tier ────────────────────────────────────────────────────────────
def budget_tier(b):
    if b < 10_000_000:
        return "Micro"
    elif b < 40_000_000:
        return "Indie"
    elif b < 100_000_000:
        return "Mid"
    elif b < 200_000_000:
        return "Tentpole"
    else:
        return "Mega"

meta["budget_tier"] = meta["budget_adj"].apply(budget_tier)

# ── 7. Release season & decade ────────────────────────────────────────────────
def release_season(month):
    if month in [1, 2]:
        return "Dump"           # Jan–Feb
    elif month in [3, 4]:
        return "Spring"
    elif month in [5, 6, 7, 8]:
        return "Summer"
    elif month in [9]:
        return "Fall"
    else:                        # Oct–Dec
        return "Awards"

meta["month"]   = meta["release_date"].dt.month
meta["season"]  = meta["month"].apply(release_season)
meta["decade"]  = (meta["year"] // 10 * 10).astype(str) + "s"

# ── 8. IP type classification ─────────────────────────────────────────────────
# Join keywords
kw["id"] = kw["id"].astype(str)
meta["id"] = meta["id"].astype(str)
meta = meta.merge(kw, on="id", how="left")
meta["kw_names"] = meta["keywords"].apply(lambda x: parse_json_names(x, n=0))
meta["kw_str"]   = meta["kw_names"].apply(lambda x: " | ".join(x).lower())

FRANCHISE_KW  = {"sequel", "prequel", "second part", "third part"}
ADAPTATION_KW = {"based on novel", "based on comic book", "based on true story",
                 "based on play", "based on short story", "based on children's book",
                 "based on tv series"}
REMAKE_KW     = {"remake", "reboot", "reimagining"}

def ip_type(row):
    kws = set(row["kw_str"].split(" | "))
    # Sequel keywords + belongs_to_collection both map to Franchise
    if kws & FRANCHISE_KW:
        return "Franchise"
    if pd.notna(row["belongs_to_collection"]) and row["belongs_to_collection"] != "":
        return "Franchise"
    if kws & REMAKE_KW:
        return "Remake"
    if kws & ADAPTATION_KW:
        return "Adaptation"
    return "Original"

meta["ip_type"] = meta.apply(ip_type, axis=1)

# ── 9. Final column selection & save ─────────────────────────────────────────
KEEP = [
    "id", "imdb_id", "title", "year", "decade", "month", "season",
    "budget_adj", "revenue_adj", "roi",
    "budget_tier", "genre_primary", "genres_list",
    "ip_type", "studio_label", "studio_group",
    "runtime", "vote_average", "vote_count",
    "original_language", "popularity",
]

clean = meta[KEEP].reset_index(drop=True)
out_path = os.path.join(OUT_DIR, "movies_clean.csv")
clean.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Final shape: {clean.shape}")

# ── 10. Quick summary ─────────────────────────────────────────────────────────
print("\n── Genre distribution ──")
print(clean["genre_primary"].value_counts().head(12))

print("\n── Budget tier distribution ──")
print(clean["budget_tier"].value_counts())

print("\n── IP type distribution ──")
print(clean["ip_type"].value_counts())

print("\n── Release season distribution ──")
print(clean["season"].value_counts())

print("\n── Studio group distribution ──")
print(clean["studio_group"].value_counts())

print("\n── Decade distribution ──")
print(clean["decade"].value_counts().sort_index())

print("\n── ROI summary ──")
print(clean["roi"].describe().round(2))
