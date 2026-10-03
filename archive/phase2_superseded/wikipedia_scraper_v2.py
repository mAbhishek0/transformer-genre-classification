# =============================================================================
# ROBUST WIKIPEDIA BOOK-TO-MOVIE SCRAPER — V2
# Run LOCALLY. Takes 60-120 min for 3000 books.
#
# Strategy priority chain (stops at first confirmed match):
#   [A] Wikidata P144 "based on" — structured, authoritative, handles
#       title changes (Blade Runner ← Do Androids Dream of Electric Sheep)
#   [B] Book Wikipedia page → Adaptations section → explicit film links
#       → verify film infobox "based_on" contains book/author
#   [C] Author's Wikipedia category "Films based on X novels"
#   [D] Fuzzy title search — ONLY accepted if infobox verification passes
#
# Every accepted match requires infobox or Wikidata confirmation.
# No match is accepted on title similarity alone.
#
# pip install requests beautifulsoup4 rapidfuzz pandas
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
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR       = "./data/"
CACHE_FILE     = "./data/wiki_scrape_v2_cache.json"
OUTPUT_FILE    = "./data/wiki_book_movie_pairs_v2.csv"
LOG_FILE       = "./data/wiki_scrape_v2_log.txt"

RATE_LIMIT     = 1.0       # seconds between API calls
MAX_BOOKS      = 3000
MIN_PLOT_LEN   = 300       # chars — stricter than V1

WIKI_API       = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API   = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL= "https://query.wikidata.org/sparql"

HEADERS = {
    "User-Agent": "NLP-Academic-Research/2.0 (book-movie-adaptation-study)"
}

# Section names that indicate film adaptations on book pages
ADAPTATION_SECTION_NAMES = {
    'adaptations', 'film adaptation', 'film adaptations',
    'film', 'films', 'movie', 'movies',
    'in other media', 'media adaptations', 'theatrical adaptations',
    'television and film', 'film and television', 'legacy',
    'cultural impact', 'screen adaptations'
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────────────────────────
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA API HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def api_get(params, base=WIKI_API, retries=3):
    """Robust GET with retry on transient errors."""
    params["format"] = "json"
    params["utf8"]   = 1
    for attempt in range(retries):
        try:
            r = requests.get(base, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:   # rate limited
                time.sleep(5 * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(2)
    return None


def wiki_search(query, n=5):
    data = api_get({"action": "query", "list": "search",
                    "srsearch": query, "srlimit": n})
    if not data:
        return []
    return [r["title"] for r in data.get("query", {}).get("search", [])]


def wiki_get_wikitext(page_title):
    """Full wikitext of a page."""
    data = api_get({"action": "parse", "page": page_title, "prop": "wikitext"})
    if not data or "error" in data:
        return ""
    return data.get("parse", {}).get("wikitext", {}).get("*", "")


def wiki_get_sections(page_title):
    """List of section objects: {index, line (title), level}"""
    data = api_get({"action": "parse", "page": page_title, "prop": "sections"})
    if not data or "error" in data:
        return []
    return data.get("parse", {}).get("sections", [])


def wiki_get_section_wikitext(page_title, section_index):
    """Wikitext of one section by index."""
    data = api_get({"action": "parse", "page": page_title,
                    "prop": "wikitext", "section": section_index})
    if not data or "error" in data:
        return ""
    return data.get("parse", {}).get("wikitext", {}).get("*", "")


def wiki_get_page_id(page_title):
    """Get Wikipedia numeric page ID — used to query Wikidata."""
    data = api_get({"action": "query", "titles": page_title, "prop": "info"})
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid != "-1":
            return pid
    return None


def wiki_get_wikidata_id(page_title):
    """Get Wikidata QID for a Wikipedia page title."""
    data = api_get({"action": "query", "titles": page_title,
                    "prop": "pageprops", "ppprop": "wikibase_item"})
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("pageprops", {}).get("wikibase_item")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# INFOBOX VERIFICATION
# Parses raw wikitext to extract the "based_on" infobox field.
# Returns True if book_title or author_name appears in "based_on" value.
# This is the CONFIRMATION step — every match must pass this or Wikidata.
# ──────────────────────────────────────────────────────────────────────────────

def extract_infobox_field(wikitext, field_name):
    """
    Extract a field value from a wikitext infobox.
    Handles multi-line values and nested templates.
    """
    # Match: | field_name = value (until next | at same nesting level)
    pattern = rf'\|\s*{re.escape(field_name)}\s*=\s*(.*?)(?=\n\s*\||\n\s*\}}|\Z)'
    match = re.search(pattern, wikitext, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    raw = match.group(1).strip()
    # Clean wikitext markup
    raw = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', raw)
    raw = re.sub(r'\{\{[^}]*\}\}', ' ', raw)
    raw = re.sub(r"'{2,}", '', raw)
    raw = re.sub(r'<[^>]+>', '', raw)
    return raw.strip().lower()


def verify_infobox_based_on(film_page_title, book_title, author_name=None):
    """
    Fetch the film page wikitext and check if the "based_on" infobox
    field references the book title or author.

    Returns: (bool confirmed, str based_on_value)
    """
    wikitext = wiki_get_wikitext(film_page_title)
    time.sleep(RATE_LIMIT)
    if not wikitext:
        return False, ""

    based_on = extract_infobox_field(wikitext, "based_on")
    if not based_on:
        # Some infoboxes use "based on" with space
        based_on = extract_infobox_field(wikitext, "based on")
    if not based_on:
        based_on = extract_infobox_field(wikitext, "screenplay")

    book_words   = set(book_title.lower().split())
    based_words  = set(based_on.split())
    word_overlap = len(book_words & based_words) / max(len(book_words), 1)

    # Confirmed if: >50% of book title words appear in based_on field
    confirmed = word_overlap >= 0.5
    # Also confirmed if author name appears
    if not confirmed and author_name:
        author_lower = author_name.lower()
        confirmed = any(part in based_on for part in author_lower.split()
                        if len(part) > 3)

    return confirmed, based_on


# ──────────────────────────────────────────────────────────────────────────────
# PLOT EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def clean_wikitext(raw):
    """Convert raw wikitext to clean plain text."""
    raw = re.sub(r'<ref[^>]*>.*?</ref>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<ref[^>]*/>', '', raw)
    raw = re.sub(r'\{\{[^}]*\}\}', '', raw)
    raw = re.sub(r'\[\[(?:File|Image|Category):[^\]]*\]\]', '',
                 raw, flags=re.IGNORECASE)
    raw = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', raw)
    raw = re.sub(r"'{2,}", '', raw)
    raw = re.sub(r'={2,}[^=\n]*={2,}', '', raw)
    raw = re.sub(r'\[\[|\]\]|\[|\]', '', raw)
    raw = re.sub(r'<[^>]+>', '', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    raw = re.sub(r'[ \t]{2,}', ' ', raw)
    return raw.strip()


def extract_plot(film_page_title):
    """
    Extract Plot section from a film Wikipedia page.
    Returns clean text or None.
    """
    sections = wiki_get_sections(film_page_title)
    time.sleep(RATE_LIMIT)

    plot_section = None
    for sec in sections:
        sec_name = sec.get("line", "").lower().strip()
        if sec_name in ("plot", "synopsis", "story", "narrative",
                        "plot summary", "film synopsis"):
            plot_section = sec
            break

    if plot_section:
        raw = wiki_get_section_wikitext(film_page_title,
                                        plot_section["index"])
        time.sleep(RATE_LIMIT)
        text = clean_wikitext(raw)
        if len(text) >= MIN_PLOT_LEN:
            return text

    # Fallback: search in full wikitext with regex
    wikitext = wiki_get_wikitext(film_page_title)
    time.sleep(RATE_LIMIT)
    plot_match = re.search(
        r'==\s*(?:Plot|Synopsis|Story)\s*==\s*(.*?)(?=\n==\s|\Z)',
        wikitext, re.DOTALL | re.IGNORECASE
    )
    if plot_match:
        text = clean_wikitext(plot_match.group(1))
        if len(text) >= MIN_PLOT_LEN:
            return text

    return None


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY A — WIKIDATA P144 "based on"
#
# P144 = "based on" property in Wikidata
# Query: find all films (Q11424) where P144 = book's Wikidata entity
# This handles title changes completely and is authoritative.
# Limitation: Wikidata coverage is ~70-80% for major films.
# ──────────────────────────────────────────────────────────────────────────────

def strategy_a_wikidata(book_title, author_name=None):
    """
    Query Wikidata for films based on this book.
    Returns list of (film_label, film_wikipedia_title) tuples.
    """
    # First get book's Wikidata QID via Wikipedia
    # Search for book page
    book_pages = wiki_search(f"{book_title} novel", n=3)
    time.sleep(RATE_LIMIT)

    book_qid = None
    for bp in book_pages:
        sim = fuzz.token_sort_ratio(book_title.lower(), bp.lower())
        if sim >= 65:
            qid = wiki_get_wikidata_id(bp)
            time.sleep(RATE_LIMIT)
            if qid:
                book_qid = qid
                break

    if not book_qid:
        return []

    # SPARQL query: films where P144 (based on) = book QID
    sparql = f"""
    SELECT ?film ?filmLabel ?article WHERE {{
      ?film wdt:P144 wd:{book_qid} .
      ?film wdt:P31 wd:Q11424 .
      OPTIONAL {{
        ?article schema:about ?film ;
                 schema:isPartOf <https://en.wikipedia.org/> .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 10
    """
    try:
        r = requests.get(
            WIKIDATA_SPARQL,
            params={"query": sparql, "format": "json"},
            headers={**HEADERS, "Accept": "application/sparql-results+json"},
            timeout=20
        )
        time.sleep(RATE_LIMIT)
        if r.status_code != 200:
            return []
        results = r.json().get("results", {}).get("bindings", [])
        films = []
        for res in results:
            label   = res.get("filmLabel", {}).get("value", "")
            article = res.get("article", {}).get("value", "")
            if article:
                # Extract Wikipedia page title from URL
                wiki_title = article.replace("https://en.wikipedia.org/wiki/", "").replace("_", " ")
                films.append((label, wiki_title))
            elif label:
                films.append((label, label))
        return films
    except Exception as e:
        log(f"    Wikidata SPARQL error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY B — Book page → Adaptations section → verified film links
#
# Most reliable Wikipedia-native method.
# Finds the book's Wikipedia page, locates adaptation sections,
# extracts [[WikiLink]] film page references, then verifies each
# film page's infobox "based_on" field.
# ──────────────────────────────────────────────────────────────────────────────

def strategy_b_book_page(book_title, author_name=None):
    """
    Returns list of (film_page_title, based_on_value) that are verified.
    """
    # Find book's Wikipedia page
    candidates = wiki_search(f"{book_title} novel", n=5)
    time.sleep(RATE_LIMIT)

    book_page = None
    for c in candidates:
        sim = fuzz.token_sort_ratio(book_title.lower(), c.lower())
        if sim >= 70:
            book_page = c
            break

    if not book_page:
        return []

    # Get all sections of the book page
    sections = wiki_get_sections(book_page)
    time.sleep(RATE_LIMIT)

    # Find adaptation-related sections
    adaptation_sections = []
    for sec in sections:
        sec_name = sec.get("line", "").lower().strip()
        if any(kw in sec_name for kw in ADAPTATION_SECTION_NAMES):
            adaptation_sections.append(sec)

    if not adaptation_sections:
        # Try full wikitext for "film" or "adaptat" mentions
        wikitext = wiki_get_wikitext(book_page)
        time.sleep(RATE_LIMIT)
        # Extract any [[X (film)]] or [[X (movie)]] links
        film_links = re.findall(
            r'\[\[([^\]|]+(?:\(film\)|\(movie\)|\(film,\s*\d+\)))[^\]]*\]\]',
            wikitext, re.IGNORECASE
        )
        return _verify_film_links(film_links, book_title, author_name)

    # Extract [[links]] from adaptation sections
    all_film_links = []
    for sec in adaptation_sections:
        sec_text = wiki_get_section_wikitext(book_page, sec["index"])
        time.sleep(RATE_LIMIT)

        # All [[WikiLinks]] in this section
        links = re.findall(r'\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]', sec_text)

        for link in links:
            link = link.strip()
            # Skip obvious non-film links
            if any(skip in link.lower() for skip in
                   ['category:', 'file:', 'image:', 'wikipedia:',
                    'help:', 'template:', 'talk:']):
                continue
            all_film_links.append(link)

    return _verify_film_links(all_film_links, book_title, author_name)


def _verify_film_links(links, book_title, author_name):
    """
    For each candidate link, verify it's a film page about this book
    via infobox "based_on" confirmation.
    Returns list of (film_page_title, based_on_value).
    """
    verified = []
    seen = set()
    for link in links:
        if link in seen:
            continue
        seen.add(link)

        confirmed, based_on = verify_infobox_based_on(link, book_title, author_name)
        time.sleep(RATE_LIMIT)

        if confirmed:
            verified.append((link, based_on))
            log(f"      ✓ Verified via infobox: '{link}' — based_on: '{based_on[:60]}'")
        else:
            # Try with "(film)" suffix if not already there
            if "(film)" not in link.lower():
                film_variant = f"{link} (film)"
                confirmed2, based_on2 = verify_infobox_based_on(
                    film_variant, book_title, author_name)
                time.sleep(RATE_LIMIT)
                if confirmed2:
                    verified.append((film_variant, based_on2))
                    log(f"      ✓ Verified via infobox: '{film_variant}'")

    return verified


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY C — Author's Wikipedia category
#
# Wikipedia has categories like "Films based on works by Stephen King"
# Query the category, then cross-reference with book title.
# ──────────────────────────────────────────────────────────────────────────────

def strategy_c_author_category(book_title, author_name):
    """
    Returns list of (film_page_title, based_on_value) from author category.
    Only runs if author_name is provided.
    """
    if not author_name:
        return []

    # Normalize author name: "Stephen Edwin King" → "Stephen King"
    # Take first and last word only
    parts  = author_name.strip().split()
    if len(parts) >= 2:
        short_name = f"{parts[0]} {parts[-1]}"
    else:
        short_name = author_name

    category = f"Films based on works by {short_name}"

    # Get members of this category
    data = api_get({
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 100,
        "cmtype":  "page",
    })
    time.sleep(RATE_LIMIT)

    if not data:
        return []

    members = [m["title"] for m in
               data.get("query", {}).get("categorymembers", [])]

    # Cross-reference: does any member title fuzzy-match the book title?
    verified = []
    for member in members:
        sim = fuzz.token_set_ratio(book_title.lower(), member.lower())
        if sim >= 55:   # lower threshold — author category already constrains it
            confirmed, based_on = verify_infobox_based_on(
                member, book_title, author_name)
            time.sleep(RATE_LIMIT)
            if confirmed:
                verified.append((member, based_on))
                log(f"      ✓ Author category match: '{member}'")

    return verified


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY D — Fuzzy title search with MANDATORY infobox verification
#
# Last resort. Searches "{book_title} film" and "{book_title} (film)".
# ONLY accepted if infobox "based_on" confirms the book.
# Rejected if title match only, no infobox confirmation.
# ──────────────────────────────────────────────────────────────────────────────

def strategy_d_fuzzy_verified(book_title, author_name=None):
    """
    Returns list of (film_page_title, based_on_value).
    Every result has passed infobox verification.
    """
    queries = [
        f"{book_title} film",
        f"{book_title} (film)",
        f"{book_title} movie",
    ]
    verified = []
    seen = set()

    for query in queries:
        results = wiki_search(query, n=5)
        time.sleep(RATE_LIMIT)
        for result in results:
            if result in seen:
                continue
            seen.add(result)
            sim = fuzz.token_set_ratio(book_title.lower(), result.lower())
            if sim < 55:
                continue
            confirmed, based_on = verify_infobox_based_on(
                result, book_title, author_name)
            time.sleep(RATE_LIMIT)
            if confirmed:
                verified.append((result, based_on))
                log(f"      ✓ Fuzzy+infobox verified: '{result}'")
                return verified   # take first confirmed match, don't over-search

    return verified


# ──────────────────────────────────────────────────────────────────────────────
# MAIN FINDER — runs all strategies in priority order
# ──────────────────────────────────────────────────────────────────────────────

def find_film_verified(book_title, author_name=None):
    """
    Master function. Tries strategies A→B→C→D.
    Returns (film_page_title, strategy_used, based_on_value) or (None, None, None).
    """
    log(f"  [A] Wikidata P144 query for '{book_title}'")
    results_a = strategy_a_wikidata(book_title, author_name)
    if results_a:
        film_label, film_wiki = results_a[0]
        # Verify plot exists
        plot = extract_plot(film_wiki)
        if plot:
            return film_wiki, "wikidata_P144", f"wikidata confirmed: {film_label}"
        # Try film_label as page title if film_wiki didn't work
        if film_label != film_wiki:
            plot = extract_plot(film_label)
            if plot:
                return film_label, "wikidata_P144", f"wikidata confirmed: {film_label}"

    log(f"  [B] Book page → Adaptations section for '{book_title}'")
    results_b = strategy_b_book_page(book_title, author_name)
    if results_b:
        film_page, based_on = results_b[0]
        plot = extract_plot(film_page)
        if plot:
            return film_page, "book_page_adaptations", based_on

    log(f"  [C] Author category for '{book_title}' (author: {author_name})")
    results_c = strategy_c_author_category(book_title, author_name)
    if results_c:
        film_page, based_on = results_c[0]
        plot = extract_plot(film_page)
        if plot:
            return film_page, "author_category", based_on

    log(f"  [D] Fuzzy+infobox verified search for '{book_title}'")
    results_d = strategy_d_fuzzy_verified(book_title, author_name)
    if results_d:
        film_page, based_on = results_d[0]
        plot = extract_plot(film_page)
        if plot:
            return film_page, "fuzzy_verified", based_on

    return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
log("="*65)
log("LOADING CMU DATASET")
log("="*65)

cmu = pd.read_csv(f"{DATA_DIR}cmu_cleaned.csv")

# Identify columns
title_col   = next((c for c in cmu.columns if 'title'  in c.lower()), None)
summary_col = next((c for c in cmu.columns
                    if any(x in c.lower() for x in
                           ['summary','plot','text','synopsis'])), None)
genre_col   = next((c for c in cmu.columns if 'genre' in c.lower()), None)
author_col  = next((c for c in cmu.columns
                    if any(x in c.lower() for x in
                           ['author','writer','creator'])), None)

log(f"Columns — title:'{title_col}' summary:'{summary_col}' "
    f"genre:'{genre_col}' author:'{author_col}'")

cmu = cmu.dropna(subset=[title_col, summary_col])
cmu = cmu[cmu[summary_col].str.len() >= 200].reset_index(drop=True)

# Prioritize books from existing confirmed pairs
priority_titles = set()
if os.path.exists(f"{DATA_DIR}book_movie_pairs.csv"):
    bmp = pd.read_csv(f"{DATA_DIR}book_movie_pairs.csv")
    priority_titles = set(bmp['book_title'].str.lower().str.strip())
    log(f"Priority titles from existing pairs: {len(priority_titles)}")

cmu['_priority'] = cmu[title_col].str.lower().str.strip().isin(priority_titles).astype(int)
cmu = cmu.sort_values('_priority', ascending=False).reset_index(drop=True)
cmu_subset = cmu.head(MAX_BOOKS)
log(f"Processing {len(cmu_subset)} books "
    f"({cmu_subset['_priority'].sum()} priority)")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN SCRAPING LOOP
# ──────────────────────────────────────────────────────────────────────────────
cache = load_cache()
log(f"Cache: {len(cache)} previously processed\n")

counts = {"found": 0, "not_found": 0, "error": 0}
start  = time.time()

for idx, row in cmu_subset.iterrows():
    book_title  = str(row[title_col]).strip()
    author_name = str(row[author_col]).strip() if author_col and pd.notna(row.get(author_col)) else None
    cache_key   = book_title.lower()

    if cache_key in cache:
        status = cache[cache_key].get("status", "")
        if status == "found":
            counts["found"] += 1
        else:
            counts["not_found"] += 1
        continue

    if idx % 50 == 0 and idx > 0:
        elapsed   = time.time() - start
        rate      = idx / elapsed if elapsed > 0 else 1
        eta_min   = (len(cmu_subset) - idx) / rate / 60 if rate > 0 else 0
        log(f"\nProgress {idx}/{len(cmu_subset)} | "
            f"Found:{counts['found']} NotFound:{counts['not_found']} "
            f"Errors:{counts['error']} | ETA:{eta_min:.0f}min")
        save_cache(cache)

    log(f"\n[{idx+1:4d}/{len(cmu_subset)}] '{book_title}'"
        + (f" by {author_name}" if author_name else ""))

    try:
        film_page, strategy, based_on = find_film_verified(book_title, author_name)

        if film_page:
            plot = extract_plot(film_page)
            if plot and len(plot) >= MIN_PLOT_LEN:
                log(f"  ✅ FOUND via [{strategy}]: '{film_page}' | plot={len(plot)} chars")
                cache[cache_key] = {
                    "status":       "found",
                    "book_title":   book_title,
                    "author":       author_name,
                    "film_page":    film_page,
                    "strategy":     strategy,
                    "based_on":     based_on,
                    "movie_plot":   plot,
                    "plot_length":  len(plot),
                }
                counts["found"] += 1
            else:
                log(f"  ⚠️  Film found but plot too short/missing")
                cache[cache_key] = {"status": "no_plot", "book_title": book_title,
                                    "film_page": film_page}
                counts["not_found"] += 1
        else:
            log(f"  ✗ NOT FOUND (all strategies exhausted)")
            cache[cache_key] = {"status": "not_found", "book_title": book_title}
            counts["not_found"] += 1

    except Exception as e:
        log(f"  ❌ ERROR: {e}")
        cache[cache_key] = {"status": "error", "book_title": book_title, "error": str(e)}
        counts["error"] += 1

save_cache(cache)
total_time = time.time() - start
log(f"\n{'='*65}")
log(f"Scraping done in {total_time/60:.1f} min | "
    f"Found:{counts['found']} NotFound:{counts['not_found']} "
    f"Errors:{counts['error']}")
log(f"Hit rate: {counts['found']/(sum(counts.values()) or 1)*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DATASET
# ──────────────────────────────────────────────────────────────────────────────
log(f"\n{'='*65}")
log("BUILDING FINAL DATASET")
log(f"{'='*65}")

records = []
for cache_key, entry in cache.items():
    if entry.get("status") != "found":
        continue

    book_title = entry["book_title"]
    match = cmu_subset[cmu_subset[title_col].str.lower().str.strip() == book_title.lower()]
    if match.empty:
        scores = cmu_subset[title_col].apply(
            lambda t: fuzz.token_sort_ratio(str(t).lower(), book_title.lower()))
        best = scores.idxmax()
        if scores[best] < 80:
            continue
        match = cmu_subset.loc[[best]]

    cmu_row = match.iloc[0]
    records.append({
        "book_title":          book_title,
        "movie_title":         entry["film_page"].replace(" (film)","").replace(" (movie)",""),
        "book_summary":        str(cmu_row[summary_col]),
        "movie_plot":          entry["movie_plot"],
        "book_genres":         str(cmu_row[genre_col]) if genre_col else "[]",
        "movie_genres":        "[]",
        "match_score":         100.0,
        "verification_method": entry["strategy"],
        "based_on_field":      entry.get("based_on",""),
        "wiki_film_page":      entry["film_page"],
        "plot_length_chars":   entry["plot_length"],
    })

wiki_df = pd.DataFrame(records)
log(f"Total pairs: {len(wiki_df)}")

if len(wiki_df) > 0:
    # Strategy breakdown
    log("\nVerification method breakdown:")
    for method, count in wiki_df["verification_method"].value_counts().items():
        pct = count/len(wiki_df)*100
        log(f"  {method:<30} {count:4d} ({pct:.1f}%)")

    log(f"\nBook summary — mean: {wiki_df['book_summary'].str.len().mean():.0f} chars")
    log(f"Movie plot   — mean: {wiki_df['movie_plot'].str.len().mean():.0f} chars")
    ratio = (wiki_df['book_summary'].str.len() / wiki_df['movie_plot'].str.len()).mean()
    log(f"Length ratio — mean: {ratio:.1f}x  (was 10x with TMDB, target <3x)")

    wiki_df.to_csv(OUTPUT_FILE, index=False)
    log(f"\nSaved: {OUTPUT_FILE}")
    log("Upload to Drive as 'wiki_book_movie_pairs_v2.csv' then run mismatch pipeline.")

    print("\nSample (sorted by plot length):")
    print(wiki_df[["book_title","movie_title","verification_method","plot_length_chars"]]
          .sort_values("plot_length_chars", ascending=False).head(15).to_string(index=False))