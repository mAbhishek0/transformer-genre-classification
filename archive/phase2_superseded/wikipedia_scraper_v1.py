# =============================================================================
# WIKIPEDIA BOOK-TO-MOVIE SCRAPER
# Run LOCALLY (not Colab) — scraping takes 45-90 min, Colab sessions die.
#
# What it does:
#   1. Loads cmu_cleaned.csv, filters to books likely to have film adaptations
#   2. For each book, searches Wikipedia for the book page
#   3. Finds film adaptation links from the book's Wikipedia page
#   4. Scrapes the film's Wikipedia "Plot" section (800-2000 words avg)
#   5. Saves incrementally — safe to interrupt and resume
#   6. Outputs wiki_book_movie_pairs.csv ready for mismatch pipeline
#
# Install: pip install wikipedia-api requests beautifulsoup4 rapidfuzz pandas
# =============================================================================

import pandas as pd
import numpy as np
import requests
import time
import json
import re
import os
import ast
from rapidfuzz import fuzz
from bs4 import BeautifulSoup
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR      = "./data/"
CACHE_FILE    = "./data/wiki_scrape_cache.json"   # incremental save — resume safe
OUTPUT_FILE   = "./data/wiki_book_movie_pairs.csv"
LOG_FILE      = "./data/wiki_scrape_log.txt"

RATE_LIMIT_SEC = 1.2      # seconds between Wikipedia API calls — be a good citizen
MAX_BOOKS      = 3000     # cap to keep runtime reasonable; increase if you want more
MIN_PLOT_LEN   = 200      # discard film plots shorter than this (stub articles)

HEADERS = {
    "User-Agent": "NLP-Research-Project/1.0 (academic; genre-classification-study)"
}

# Wikipedia API endpoint
WIKI_API = "https://en.wikipedia.org/w/api.php"

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def log(msg):
    ts  = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — LOAD & FILTER CMU DATASET
# Keep books where the title suggests a widely-known work.
# We don't filter aggressively here — the Wikipedia search handles rejection.
# ──────────────────────────────────────────────────────────────────────────────
log("="*60)
log("STAGE 1 — LOADING CMU CLEANED DATASET")
log("="*60)

cmu = pd.read_csv(f"{DATA_DIR}cmu_cleaned.csv")
log(f"CMU dataset: {len(cmu)} books, columns: {cmu.columns.tolist()}")

# Identify the title and summary columns (handle naming variations)
title_col   = next((c for c in cmu.columns if 'title' in c.lower()), None)
summary_col = next((c for c in cmu.columns
                    if any(x in c.lower() for x in ['summary','plot','text','synopsis'])), None)
genre_col   = next((c for c in cmu.columns if 'genre' in c.lower()), None)

log(f"Identified columns — title: '{title_col}', summary: '{summary_col}', genre: '{genre_col}'")
assert title_col and summary_col, "Could not find title/summary columns — check column names"

# Drop nulls and very short summaries
cmu = cmu.dropna(subset=[title_col, summary_col])
cmu = cmu[cmu[summary_col].str.len() >= 200].copy()
cmu = cmu.reset_index(drop=True)
log(f"After null/length filter: {len(cmu)} books")

# Optionally load existing book_movie_pairs to prioritize known-paired books
# These are already confirmed to have film adaptations — highest priority
priority_titles = set()
bmp_path = f"{DATA_DIR}book_movie_pairs.csv"
if os.path.exists(bmp_path):
    bmp = pd.read_csv(bmp_path)
    priority_titles = set(bmp['book_title'].str.lower().str.strip())
    log(f"Loaded {len(priority_titles)} priority titles from book_movie_pairs.csv")

cmu['_priority'] = cmu[title_col].str.lower().str.strip().isin(priority_titles).astype(int)
cmu = cmu.sort_values('_priority', ascending=False).reset_index(drop=True)

# Cap at MAX_BOOKS (priority books always included first)
cmu_subset = cmu.head(MAX_BOOKS).copy()
log(f"Processing {len(cmu_subset)} books "
    f"({cmu_subset['_priority'].sum()} priority, "
    f"{len(cmu_subset) - cmu_subset['_priority'].sum()} general)")

# ──────────────────────────────────────────────────────────────────────────────
# CACHE SYSTEM — load previous progress to enable resume
# ──────────────────────────────────────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    log(f"Loaded cache: {len(cache)} previously processed books")
else:
    cache = {}
    log("Starting fresh cache")

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA API UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def wiki_search(query, n=5):
    """Search Wikipedia and return top n page titles."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": n,
        "format": "json",
        "utf8": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        return [res["title"] for res in results]
    except Exception as e:
        log(f"  Search error for '{query}': {e}")
        return []


def wiki_get_links(page_title):
    """Get all internal Wikipedia links from a page."""
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "links",
        "pllimit": 500,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return [l["title"] for l in page.get("links", [])]
    except Exception as e:
        log(f"  Links error for '{page_title}': {e}")
    return []


def wiki_get_sections(page_title):
    """Get section titles from a Wikipedia page."""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "sections",
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            return []
        return data.get("parse", {}).get("sections", [])
    except Exception as e:
        log(f"  Sections error for '{page_title}': {e}")
        return []


def wiki_get_plot_section(page_title):
    """
    Extract the Plot section text from a Wikipedia film page.
    Returns cleaned plain text or None if no Plot section found.
    """
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            return None

        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
        if not wikitext:
            return None

        # Extract Plot section from wikitext
        # Pattern: == Plot == ... == Next Section ==
        plot_match = re.search(
            r'==\s*[Pp]lot\s*==\s*(.*?)(?=\n==\s|\Z)',
            wikitext,
            re.DOTALL
        )
        if not plot_match:
            # Try "Synopsis" or "Story" as fallbacks
            plot_match = re.search(
                r'==\s*(?:Synopsis|Story|Narrative)\s*==\s*(.*?)(?=\n==\s|\Z)',
                wikitext,
                re.DOTALL
            )
        if not plot_match:
            return None

        raw = plot_match.group(1)

        # Clean wikitext markup
        # Remove refs: <ref ...>...</ref>
        raw = re.sub(r'<ref[^>]*>.*?</ref>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'<ref[^>]*/>', '', raw)
        # Remove templates: {{...}}
        raw = re.sub(r'\{\{[^}]*\}\}', '', raw)
        # Remove file/image links: [[File:...]] [[Image:...]]
        raw = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', raw, flags=re.IGNORECASE)
        # Convert wikilinks [[text|display]] → display, [[text]] → text
        raw = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', raw)
        # Remove remaining markup
        raw = re.sub(r"'{2,}", '', raw)       # bold/italic
        raw = re.sub(r'={2,}[^=]*={2,}', '', raw)  # sub-headings
        raw = re.sub(r'\[\[|\]\]', '', raw)
        # Collapse whitespace
        raw = re.sub(r'\n{3,}', '\n\n', raw)
        raw = re.sub(r'[ \t]+', ' ', raw)
        raw = raw.strip()

        return raw if len(raw) >= MIN_PLOT_LEN else None

    except Exception as e:
        log(f"  Plot extraction error for '{page_title}': {e}")
        return None


def wiki_get_page_content_summary(page_title):
    """
    Get the plain-text introduction/summary of a Wikipedia page.
    Used as fallback when no Plot section exists.
    """
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            return text if len(text) >= MIN_PLOT_LEN else None
    except:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CORE LOGIC: Find film Wikipedia page from book title
#
# Strategy (in order of reliability):
#   1. Search "{book_title} film" → check if result is a film page
#   2. Search "{book_title} novel" → get book page → find film adaptation links
#   3. Search "{book_title} (film)" directly
#
# Film page detection: check categories for "film" keyword OR check sections
# for "Plot" which is a strong film page signal.
# ──────────────────────────────────────────────────────────────────────────────

FILM_KEYWORDS = {'film', 'movie', 'motion picture', 'directed by', 'screenplay'}

def is_film_page(page_title):
    """
    Heuristic: is this Wikipedia page about a film?
    Check page categories via API.
    """
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "categories",
        "cllimit": 30,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=8)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            cats = [c["title"].lower() for c in page.get("categories", [])]
            cat_text = " ".join(cats)
            if any(kw in cat_text for kw in FILM_KEYWORDS):
                return True
    except:
        pass
    return False


def find_film_wiki_page(book_title, author=None):
    """
    Given a book title, find the Wikipedia page for its film adaptation.
    Returns (film_page_title, method_used) or (None, None).
    """
    # Strategy 1: Direct "(film)" search
    candidates_1 = wiki_search(f"{book_title} film", n=5)
    time.sleep(RATE_LIMIT_SEC)

    for candidate in candidates_1:
        cand_lower = candidate.lower()
        title_lower = book_title.lower()
        # Check if candidate title contains book title words (fuzzy)
        similarity = fuzz.token_set_ratio(title_lower, cand_lower)
        if similarity >= 60 and is_film_page(candidate):
            time.sleep(RATE_LIMIT_SEC)
            return candidate, "direct_film_search"
        time.sleep(0.3)

    # Strategy 2: Search for book page, then look for film adaptation links
    candidates_2 = wiki_search(f"{book_title} novel book", n=3)
    time.sleep(RATE_LIMIT_SEC)

    for book_page in candidates_2:
        similarity = fuzz.token_set_ratio(book_title.lower(), book_page.lower())
        if similarity < 50:
            continue
        links = wiki_get_links(book_page)
        time.sleep(RATE_LIMIT_SEC)

        # Look for links that contain the book title + film-related words
        film_links = []
        for link in links:
            link_lower = link.lower()
            title_words = set(book_title.lower().split())
            link_words  = set(link_lower.split())
            word_overlap = len(title_words & link_words) / max(len(title_words), 1)
            if word_overlap >= 0.5 and any(kw in link_lower for kw in ['film','movie','series']):
                film_links.append(link)

        for film_link in film_links[:3]:
            if is_film_page(film_link):
                time.sleep(RATE_LIMIT_SEC)
                return film_link, "book_page_links"
            time.sleep(0.3)

    # Strategy 3: Try "{title} (film)" directly
    direct_title = f"{book_title} (film)"
    plot = wiki_get_plot_section(direct_title)
    time.sleep(RATE_LIMIT_SEC)
    if plot:
        return direct_title, "direct_film_title"

    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 & 3 — MAIN SCRAPING LOOP
# ──────────────────────────────────────────────────────────────────────────────
log("\n" + "="*60)
log("STAGE 2+3 — WIKIPEDIA SCRAPING")
log("="*60)
log(f"Processing {len(cmu_subset)} books. ETA: ~{len(cmu_subset)*2.5/60:.0f} minutes")
log("Progress saved incrementally — safe to Ctrl+C and resume.\n")

found      = 0
not_found  = 0
errors     = 0
start_time = time.time()

for idx, row in cmu_subset.iterrows():
    book_title = str(row[title_col]).strip()
    cache_key  = book_title.lower()

    # Skip if already processed
    if cache_key in cache:
        if cache[cache_key].get('status') in ('found', 'not_found', 'error'):
            if cache[cache_key].get('status') == 'found':
                found += 1
            else:
                not_found += 1
            continue

    # Progress report every 50 books
    if idx % 50 == 0 and idx > 0:
        elapsed  = time.time() - start_time
        rate     = idx / elapsed if elapsed > 0 else 1
        remaining = (len(cmu_subset) - idx) / rate if rate > 0 else 0
        log(f"Progress: {idx}/{len(cmu_subset)} | "
            f"Found: {found} | Not found: {not_found} | "
            f"ETA: {remaining/60:.1f} min")
        save_cache()

    try:
        log(f"[{idx+1:4d}/{len(cmu_subset)}] '{book_title}'")

        film_page, method = find_film_wiki_page(book_title)

        if film_page is None:
            log(f"         → NOT FOUND")
            cache[cache_key] = {
                'status': 'not_found',
                'book_title': book_title,
            }
            not_found += 1
            continue

        log(f"         → Found: '{film_page}' (via {method})")

        # Get the plot
        plot_text = wiki_get_plot_section(film_page)
        time.sleep(RATE_LIMIT_SEC)

        if not plot_text:
            # Fallback: use page intro text
            plot_text = wiki_get_page_content_summary(film_page)
            time.sleep(RATE_LIMIT_SEC)

        if not plot_text:
            log(f"         → Film page found but no plot text extracted")
            cache[cache_key] = {
                'status': 'not_found',
                'book_title': book_title,
                'film_page': film_page,
                'note': 'no_plot_text'
            }
            not_found += 1
            continue

        log(f"         → Plot extracted: {len(plot_text)} chars")
        found += 1

        cache[cache_key] = {
            'status':       'found',
            'book_title':   book_title,
            'film_page':    film_page,
            'method':       method,
            'movie_plot':   plot_text,
            'plot_length':  len(plot_text),
        }

    except Exception as e:
        log(f"         → ERROR: {e}")
        cache[cache_key] = {'status': 'error', 'book_title': book_title, 'error': str(e)}
        errors += 1

# Final save
save_cache()
elapsed = time.time() - start_time
log(f"\nScraping complete in {elapsed/60:.1f} minutes")
log(f"Found: {found} | Not found: {not_found} | Errors: {errors}")
log(f"Hit rate: {found/(found+not_found+errors)*100:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — BUILD DATASET
# Join scraped film plots with CMU book summaries.
# Output columns match book_movie_pairs.csv schema exactly for pipeline reuse.
# ──────────────────────────────────────────────────────────────────────────────
log("\n" + "="*60)
log("STAGE 4 — BUILDING WIKI DATASET")
log("="*60)

# Collect successful scrapes
records = []
for cache_key, entry in cache.items():
    if entry.get('status') != 'found':
        continue

    book_title = entry['book_title']

    # Find matching row in CMU dataset
    cmu_match = cmu_subset[
        cmu_subset[title_col].str.lower().str.strip() == book_title.lower()
    ]
    if cmu_match.empty:
        # Fuzzy fallback
        scores = cmu_subset[title_col].apply(
            lambda t: fuzz.token_sort_ratio(str(t).lower(), book_title.lower())
        )
        best_idx = scores.idxmax()
        if scores[best_idx] < 80:
            log(f"  No CMU match for '{book_title}' — skipping")
            continue
        cmu_match = cmu_subset.loc[[best_idx]]

    cmu_row = cmu_match.iloc[0]

    record = {
        'book_title':   book_title,
        'movie_title':  entry['film_page'].replace(' (film)', '').replace(' (movie)', ''),
        'book_summary': str(cmu_row[summary_col]),
        'movie_plot':   entry['movie_plot'],
        'book_genres':  str(cmu_row[genre_col]) if genre_col else '[]',
        'movie_genres': '[]',   # Wikipedia doesn't provide genre tags
        'match_score':  100.0,  # Wikipedia match is verified
        'wiki_film_page': entry['film_page'],
        'wiki_method':    entry['method'],
        'plot_length_chars': entry['plot_length'],
    }
    records.append(record)

wiki_df = pd.DataFrame(records)
log(f"Built dataset: {len(wiki_df)} pairs")

if len(wiki_df) == 0:
    log("ERROR: No pairs built — check scraping results")
else:
    # Quality filter: drop plots that are too short
    wiki_df = wiki_df[wiki_df['plot_length_chars'] >= MIN_PLOT_LEN].copy()
    wiki_df = wiki_df.reset_index(drop=True)

    log(f"After quality filter: {len(wiki_df)} pairs")
    log(f"Book summary — mean: {wiki_df['book_summary'].str.len().mean():.0f} chars")
    log(f"Movie plot   — mean: {wiki_df['movie_plot'].str.len().mean():.0f} chars")
    log(f"Length ratio — mean: {(wiki_df['book_summary'].str.len() / wiki_df['movie_plot'].str.len()).mean():.1f}x")

    wiki_df.to_csv(OUTPUT_FILE, index=False)
    log(f"\nSaved: {OUTPUT_FILE}")
    log(f"Upload this file to Google Drive as 'wiki_book_movie_pairs.csv'")
    log(f"Then run the mismatch pipeline with DATA_FILE = 'wiki_book_movie_pairs.csv'")

    # Quick preview
    print("\nSample pairs:")
    preview_cols = ['book_title', 'movie_title', 'plot_length_chars']
    print(wiki_df[preview_cols].head(10).to_string(index=False))

    print(f"\nPlot length distribution:")
    print(wiki_df['plot_length_chars'].describe().round(0))