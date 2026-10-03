# =============================================================================
# MULTI-PASS WIKIPEDIA BOOK-TO-MOVIE SCRAPER — V3
# Run LOCALLY. Each pass is independent and resumable.
#
# PASS 1 — Wikidata P144 "based on"          [fastest, highest confidence]
# PASS 2 — Book page → Adaptations section   [medium speed, high confidence]
# PASS 3 — Author Wikipedia category         [medium speed, high confidence]
# PASS 4 — Fuzzy title + infobox verified    [slowest, lowest confidence]
#
# After each pass: summary printed → you decide to stop or continue.
# Run with: python wikipedia_scraper_v3.py --pass 1
#           python wikipedia_scraper_v3.py --pass 2
#           python wikipedia_scraper_v3.py --pass 3
#           python wikipedia_scraper_v3.py --pass 4
#           python wikipedia_scraper_v3.py --build   (assemble final CSV)
#
# pip install requests rapidfuzz pandas
# =============================================================================

import argparse
import pandas as pd
import numpy as np
import requests
import time
import json
import re
import os
from rapidfuzz import fuzz
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR    = "./data/"
STATE_FILE  = "./data/scrape_state.json"   # master state across all passes
OUTPUT_FILE = "./data/wiki_book_movie_pairs_v3.csv"
LOG_FILE    = "./data/scrape_v3_log.txt"

RATE_LIMIT  = 1.0      # seconds between API calls
MAX_BOOKS   = 3000     # cap total books processed
MIN_PLOT    = 300      # minimum plot length in chars to accept

WIKI_API        = "https://en.wikipedia.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "NLP-Academic-Research/3.0 (book-movie-adaptation-study)"}

ADAPTATION_KEYWORDS = {
    'adaptations', 'film adaptation', 'film adaptations', 'film', 'films',
    'movie', 'movies', 'in other media', 'media adaptations',
    'theatrical adaptations', 'television and film', 'film and television',
    'screen adaptations', 'legacy', 'cultural impact'
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def log(msg, also_print=True):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# Each book has one record in state:
# {
#   "book_title":   str,
#   "author":       str | null,
#   "book_summary": str,
#   "book_genres":  str,
#   "status":       "pending" | "found" | "exhausted",
#   "pass_tried":   int,          # highest pass number attempted
#   "film_page":    str | null,   # Wikipedia page title of film
#   "strategy":     str | null,   # which pass found it
#   "based_on":     str | null,   # infobox confirmation text
#   "movie_plot":   str | null,   # extracted plot text
#   "plot_length":  int | null,
# }
# ──────────────────────────────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def state_key(book_title):
    return book_title.lower().strip()

def get_pending(state, pass_number):
    """Books not yet found and not tried at this pass level."""
    return {
        k: v for k, v in state.items()
        if v["status"] == "pending" and v.get("pass_tried", 0) < pass_number
    }

def print_state_summary(state, label="Current State"):
    found     = [v for v in state.values() if v["status"] == "found"]
    pending   = [v for v in state.values() if v["status"] == "pending"]
    exhausted = [v for v in state.values() if v["status"] == "exhausted"]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total books:   {len(state)}")
    print(f"  ✅ Found:      {len(found)}")
    print(f"  ⏳ Pending:    {len(pending)}")
    print(f"  ✗  Exhausted:  {len(exhausted)}")

    if found:
        by_strategy = {}
        for v in found:
            s = v.get("strategy", "unknown")
            by_strategy[s] = by_strategy.get(s, 0) + 1
        print(f"\n  Matches by strategy:")
        for s, c in sorted(by_strategy.items(), key=lambda x: -x[1]):
            print(f"    {s:<35} {c:4d} ({c/len(found)*100:.1f}%)")

        plots = [v["plot_length"] for v in found if v.get("plot_length")]
        if plots:
            print(f"\n  Plot length — mean: {np.mean(plots):.0f} | "
                  f"median: {np.median(plots):.0f} | "
                  f"min: {min(plots):.0f} | max: {max(plots):.0f}")

    print(f"{'='*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA / WIKIDATA API HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def api_get(params, base=WIKI_API, retries=3):
    params["format"] = "json"
    params["utf8"]   = 1
    for attempt in range(retries):
        try:
            r = requests.get(base, params=params,
                             headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(2)
    return None

def wiki_search(query, n=5):
    data = api_get({"action":"query","list":"search",
                    "srsearch":query,"srlimit":n})
    if not data:
        return []
    return [r["title"] for r in data.get("query",{}).get("search",[])]

def wiki_get_wikitext(page_title):
    data = api_get({"action":"parse","page":page_title,"prop":"wikitext"})
    if not data or "error" in data:
        return ""
    return data.get("parse",{}).get("wikitext",{}).get("*","")

def wiki_get_sections(page_title):
    data = api_get({"action":"parse","page":page_title,"prop":"sections"})
    if not data or "error" in data:
        return []
    return data.get("parse",{}).get("sections",[])

def wiki_get_section_wikitext(page_title, section_index):
    data = api_get({"action":"parse","page":page_title,
                    "prop":"wikitext","section":section_index})
    if not data or "error" in data:
        return ""
    return data.get("parse",{}).get("wikitext",{}).get("*","")

def wiki_get_wikidata_id(page_title):
    data = api_get({"action":"query","titles":page_title,
                    "prop":"pageprops","ppprop":"wikibase_item"})
    if not data:
        return None
    for page in data.get("query",{}).get("pages",{}).values():
        return page.get("pageprops",{}).get("wikibase_item")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# INFOBOX VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def extract_infobox_field(wikitext, field):
    pattern = rf'\|\s*{re.escape(field)}\s*=\s*(.*?)(?=\n\s*\||\n\s*\}}|\Z)'
    m = re.search(pattern, wikitext, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    raw = m.group(1).strip()
    raw = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', raw)
    raw = re.sub(r'\{\{[^}]*\}\}', ' ', raw)
    raw = re.sub(r"'{2,}|<[^>]+>", '', raw)
    return raw.strip().lower()

def verify_infobox(film_page, book_title, author=None):
    """
    Returns (confirmed: bool, based_on_text: str).
    Fetches film page wikitext and checks "based_on" field.
    """
    wikitext = wiki_get_wikitext(film_page)
    time.sleep(RATE_LIMIT)
    if not wikitext:
        return False, ""

    based_on = (extract_infobox_field(wikitext, "based_on") or
                extract_infobox_field(wikitext, "based on") or
                extract_infobox_field(wikitext, "screenplay"))

    book_words   = set(book_title.lower().split())
    overlap      = len(book_words & set(based_on.split())) / max(len(book_words), 1)
    confirmed    = overlap >= 0.5

    if not confirmed and author:
        author_parts = [p for p in author.lower().split() if len(p) > 3]
        confirmed = any(p in based_on for p in author_parts)

    return confirmed, based_on

# ──────────────────────────────────────────────────────────────────────────────
# PLOT EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def clean_wikitext(raw):
    raw = re.sub(r'<ref[^>]*>.*?</ref>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<ref[^>]*/>', '', raw)
    raw = re.sub(r'\{\{[^}]*\}\}', '', raw)
    raw = re.sub(r'\[\[(?:File|Image|Category):[^\]]*\]\]', '',
                 raw, flags=re.IGNORECASE)
    raw = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', raw)
    raw = re.sub(r"'{2,}|={2,}[^=\n]*={2,}|<[^>]+>|\[\[|\]\]", '', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    raw = re.sub(r'[ \t]{2,}', ' ', raw)
    return raw.strip()

def extract_plot(film_page):
    """Extract Plot section. Returns clean text or None."""
    sections = wiki_get_sections(film_page)
    time.sleep(RATE_LIMIT)

    PLOT_NAMES = {"plot","synopsis","story","narrative",
                  "plot summary","film synopsis"}
    plot_sec = next(
        (s for s in sections if s.get("line","").lower().strip() in PLOT_NAMES),
        None
    )

    if plot_sec:
        raw  = wiki_get_section_wikitext(film_page, plot_sec["index"])
        time.sleep(RATE_LIMIT)
        text = clean_wikitext(raw)
        if len(text) >= MIN_PLOT:
            return text

    # Regex fallback on full wikitext
    wikitext = wiki_get_wikitext(film_page)
    time.sleep(RATE_LIMIT)
    m = re.search(
        r'==\s*(?:Plot|Synopsis|Story)\s*==\s*(.*?)(?=\n==|\Z)',
        wikitext, re.DOTALL | re.IGNORECASE
    )
    if m:
        text = clean_wikitext(m.group(1))
        if len(text) >= MIN_PLOT:
            return text
    return None

# ──────────────────────────────────────────────────────────────────────────────
# PASS 1 — WIKIDATA P144
# For each pending book: get its Wikidata QID → SPARQL for films where P144=QID
# No fuzzy matching. If Wikidata says "film X is based on book Y", it's correct.
# API calls per book: ~3-4 (search book page + get QID + SPARQL)
# ──────────────────────────────────────────────────────────────────────────────

def run_pass1(state):
    pending = get_pending(state, pass_number=1)
    log(f"\nPASS 1 — Wikidata P144 | {len(pending)} books pending")
    log("Estimated time: {:.0f}-{:.0f} min".format(
        len(pending)*4/60, len(pending)*6/60))

    found_this_pass = 0
    for i, (key, entry) in enumerate(pending.items()):
        book_title  = entry["book_title"]
        author      = entry.get("author")

        if i % 25 == 0 and i > 0:
            log(f"  Progress: {i}/{len(pending)} | found this pass: {found_this_pass}")
            save_state(state)

        log(f"  [{i+1:4d}/{len(pending)}] '{book_title}'", also_print=True)

        # Step 1: find book's Wikipedia page to get Wikidata QID
        book_pages = wiki_search(f"{book_title} novel", n=3)
        time.sleep(RATE_LIMIT)

        book_qid = None
        for bp in book_pages:
            if fuzz.token_sort_ratio(book_title.lower(), bp.lower()) >= 65:
                qid = wiki_get_wikidata_id(bp)
                time.sleep(RATE_LIMIT)
                if qid:
                    book_qid = qid
                    break

        state[key]["pass_tried"] = 1

        if not book_qid:
            log(f"         No Wikidata QID found", also_print=False)
            continue

        # Step 2: SPARQL — films where P144 = book QID
        sparql = f"""
        SELECT ?film ?filmLabel ?article WHERE {{
          ?film wdt:P144 wd:{book_qid} .
          ?film wdt:P31 wd:Q11424 .
          OPTIONAL {{
            ?article schema:about ?film ;
                     schema:isPartOf <https://en.wikipedia.org/> .
          }}
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
          }}
        }} LIMIT 5
        """
        try:
            r = requests.get(
                WIKIDATA_SPARQL,
                params={"query": sparql, "format": "json"},
                headers={**HEADERS, "Accept":"application/sparql-results+json"},
                timeout=20
            )
            time.sleep(RATE_LIMIT)
            if r.status_code != 200:
                continue
            bindings = r.json().get("results",{}).get("bindings",[])
        except Exception as e:
            log(f"         SPARQL error: {e}", also_print=False)
            continue

        if not bindings:
            log(f"         No P144 film found in Wikidata", also_print=False)
            continue

        # Step 3: for each result, get Wikipedia page and extract plot
        for binding in bindings:
            film_label   = binding.get("filmLabel",{}).get("value","")
            article_url  = binding.get("article",{}).get("value","")
            film_page    = (article_url
                            .replace("https://en.wikipedia.org/wiki/","")
                            .replace("_"," ")
                            if article_url else film_label)

            log(f"         Wikidata hit: '{film_page}'", also_print=True)

            plot = extract_plot(film_page)
            if plot:
                log(f"         ✅ Plot extracted: {len(plot)} chars")
                state[key].update({
                    "status":      "found",
                    "film_page":   film_page,
                    "strategy":    "pass1_wikidata_P144",
                    "based_on":    f"wikidata QID {book_qid}",
                    "movie_plot":  plot,
                    "plot_length": len(plot),
                })
                found_this_pass += 1
                break   # take first film with a valid plot
            else:
                log(f"         Plot too short/missing — trying next Wikidata result")

    save_state(state)
    log(f"\nPass 1 complete. Found: {found_this_pass} new pairs.")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# PASS 2 — BOOK PAGE → ADAPTATIONS SECTION
# Finds book's Wikipedia page, locates adaptation sections,
# extracts film links, verifies each via infobox "based_on".
# API calls per book: ~5-8
# ──────────────────────────────────────────────────────────────────────────────

def run_pass2(state):
    pending = get_pending(state, pass_number=2)
    log(f"\nPASS 2 — Book page adaptations | {len(pending)} books pending")

    found_this_pass = 0
    for i, (key, entry) in enumerate(pending.items()):
        book_title = entry["book_title"]
        author     = entry.get("author")

        if i % 25 == 0 and i > 0:
            log(f"  Progress: {i}/{len(pending)} | found: {found_this_pass}")
            save_state(state)

        log(f"  [{i+1:4d}/{len(pending)}] '{book_title}'", also_print=True)
        state[key]["pass_tried"] = 2

        # Find book's Wikipedia page
        candidates = wiki_search(f"{book_title} novel", n=5)
        time.sleep(RATE_LIMIT)

        book_page = next(
            (c for c in candidates
             if fuzz.token_sort_ratio(book_title.lower(), c.lower()) >= 70),
            None
        )
        if not book_page:
            log(f"         No book Wikipedia page found", also_print=False)
            continue

        log(f"         Book page: '{book_page}'", also_print=False)

        # Get sections — look for adaptation sections
        sections = wiki_get_sections(book_page)
        time.sleep(RATE_LIMIT)

        adapt_sections = [
            s for s in sections
            if any(kw in s.get("line","").lower()
                   for kw in ADAPTATION_KEYWORDS)
        ]

        # Collect candidate film links
        candidate_links = []

        if adapt_sections:
            for sec in adapt_sections:
                sec_text = wiki_get_section_wikitext(book_page, sec["index"])
                time.sleep(RATE_LIMIT)
                # Extract all [[WikiLinks]]
                links = re.findall(r'\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]', sec_text)
                candidate_links.extend(
                    l.strip() for l in links
                    if not any(skip in l.lower() for skip in
                               ['category:','file:','image:','template:','wikipedia:'])
                )
        else:
            # No dedicated section — search full wikitext for (film) links
            wikitext = wiki_get_wikitext(book_page)
            time.sleep(RATE_LIMIT)
            film_links = re.findall(
                r'\[\[([^\]|]+\((?:film|movie)\)[^\]]*)\]\]',
                wikitext, re.IGNORECASE
            )
            candidate_links.extend(l.strip() for l in film_links)

        if not candidate_links:
            log(f"         No film links found in book page", also_print=False)
            continue

        log(f"         Candidates: {candidate_links[:5]}", also_print=False)

        # Verify each candidate
        found = False
        for link in dict.fromkeys(candidate_links):   # deduplicate preserving order
            confirmed, based_on = verify_infobox(link, book_title, author)
            time.sleep(RATE_LIMIT)

            if not confirmed and "(film)" not in link.lower():
                confirmed, based_on = verify_infobox(
                    f"{link} (film)", book_title, author)
                if confirmed:
                    link = f"{link} (film)"
                time.sleep(RATE_LIMIT)

            if confirmed:
                plot = extract_plot(link)
                if plot:
                    log(f"         ✅ Verified+plot: '{link}' ({len(plot)} chars)")
                    state[key].update({
                        "status":      "found",
                        "film_page":   link,
                        "strategy":    "pass2_book_adaptations",
                        "based_on":    based_on,
                        "movie_plot":  plot,
                        "plot_length": len(plot),
                    })
                    found_this_pass += 1
                    found = True
                    break

        if not found:
            log(f"         No verified film with plot found", also_print=False)

    save_state(state)
    log(f"\nPass 2 complete. Found: {found_this_pass} new pairs.")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# PASS 3 — AUTHOR WIKIPEDIA CATEGORY
# Queries "Films based on works by {Author}" category,
# cross-references with book title, verifies via infobox.
# Requires author_name. Skips books with no author info.
# ──────────────────────────────────────────────────────────────────────────────

def run_pass3(state):
    pending = get_pending(state, pass_number=3)
    # Only books where we have an author name
    pending = {k: v for k, v in pending.items() if v.get("author")}
    log(f"\nPASS 3 — Author category | {len(pending)} books pending (with author)")

    found_this_pass = 0
    author_category_cache = {}   # avoid re-fetching same author's category

    for i, (key, entry) in enumerate(pending.items()):
        book_title = entry["book_title"]
        author     = entry["author"]

        if i % 25 == 0 and i > 0:
            log(f"  Progress: {i}/{len(pending)} | found: {found_this_pass}")
            save_state(state)

        log(f"  [{i+1:4d}/{len(pending)}] '{book_title}' by {author}",
            also_print=True)
        state[key]["pass_tried"] = 3

        # Normalize author: "Stephen Edwin King" → "Stephen King"
        parts = author.strip().split()
        short_author = f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else author
        category = f"Films based on works by {short_author}"

        # Fetch category members (cached per author)
        if category not in author_category_cache:
            data = api_get({
                "action":  "query",
                "list":    "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": 200,
                "cmtype":  "page",
            })
            time.sleep(RATE_LIMIT)
            members = [m["title"] for m in
                       data.get("query",{}).get("categorymembers",[])
                       ] if data else []
            author_category_cache[category] = members
            log(f"         Category '{category}': {len(members)} films",
                also_print=False)
        else:
            members = author_category_cache[category]

        if not members:
            log(f"         Category empty or not found", also_print=False)
            continue

        # Cross-reference: find members that fuzzy-match book title
        matches = sorted(
            [(m, fuzz.token_set_ratio(book_title.lower(), m.lower()))
             for m in members],
            key=lambda x: -x[1]
        )
        top_matches = [(m, s) for m, s in matches if s >= 50]

        if not top_matches:
            log(f"         No title match in author category", also_print=False)
            continue

        log(f"         Top matches: {top_matches[:3]}", also_print=False)

        for film_page, sim in top_matches[:3]:
            confirmed, based_on = verify_infobox(film_page, book_title, author)
            time.sleep(RATE_LIMIT)

            if confirmed:
                plot = extract_plot(film_page)
                if plot:
                    log(f"         ✅ '{film_page}' (sim={sim}, {len(plot)} chars)")
                    state[key].update({
                        "status":      "found",
                        "film_page":   film_page,
                        "strategy":    "pass3_author_category",
                        "based_on":    based_on,
                        "movie_plot":  plot,
                        "plot_length": len(plot),
                    })
                    found_this_pass += 1
                    break

    save_state(state)
    log(f"\nPass 3 complete. Found: {found_this_pass} new pairs.")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# PASS 4 — FUZZY TITLE + MANDATORY INFOBOX VERIFICATION
# Last resort. Every result must pass infobox check.
# ──────────────────────────────────────────────────────────────────────────────

def run_pass4(state):
    pending = get_pending(state, pass_number=4)
    log(f"\nPASS 4 — Fuzzy+infobox | {len(pending)} books pending")

    found_this_pass = 0
    for i, (key, entry) in enumerate(pending.items()):
        book_title = entry["book_title"]
        author     = entry.get("author")

        if i % 25 == 0 and i > 0:
            log(f"  Progress: {i}/{len(pending)} | found: {found_this_pass}")
            save_state(state)

        log(f"  [{i+1:4d}/{len(pending)}] '{book_title}'", also_print=True)
        state[key]["pass_tried"] = 4

        for query in [f"{book_title} film", f"{book_title} (film)"]:
            results = wiki_search(query, n=5)
            time.sleep(RATE_LIMIT)

            for result in results:
                if fuzz.token_set_ratio(book_title.lower(), result.lower()) < 55:
                    continue
                confirmed, based_on = verify_infobox(result, book_title, author)
                time.sleep(RATE_LIMIT)
                if confirmed:
                    plot = extract_plot(result)
                    if plot:
                        log(f"         ✅ '{result}' ({len(plot)} chars)")
                        state[key].update({
                            "status":      "found",
                            "film_page":   result,
                            "strategy":    "pass4_fuzzy_verified",
                            "based_on":    based_on,
                            "movie_plot":  plot,
                            "plot_length": len(plot),
                        })
                        found_this_pass += 1
                        break
            else:
                continue
            break

        # Mark as exhausted after pass 4
        if state[key]["status"] == "pending":
            state[key]["status"] = "exhausted"

    save_state(state)
    log(f"\nPass 4 complete. Found: {found_this_pass} new pairs.")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# BUILD FINAL DATASET
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(state):
    log(f"\n{'='*60}")
    log("BUILDING FINAL DATASET")
    log(f"{'='*60}")

    records = [
        {
            "book_title":          v["book_title"],
            "movie_title":         v["film_page"].replace(" (film)","").replace(" (movie)",""),
            "book_summary":        v["book_summary"],
            "movie_plot":          v["movie_plot"],
            "book_genres":         v.get("book_genres","[]"),
            "movie_genres":        "[]",
            "match_score":         100.0,
            "verification_method": v["strategy"],
            "based_on_field":      v.get("based_on",""),
            "wiki_film_page":      v["film_page"],
            "plot_length_chars":   v["plot_length"],
        }
        for v in state.values()
        if v["status"] == "found" and v.get("movie_plot")
    ]

    df = pd.DataFrame(records)
    if df.empty:
        log("No found pairs yet — run at least Pass 1 first.")
        return

    df = df[df["plot_length_chars"] >= MIN_PLOT].reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    log(f"Saved {len(df)} pairs to {OUTPUT_FILE}")
    log(f"Book summary  — mean: {df['book_summary'].str.len().mean():.0f} chars")
    log(f"Movie plot    — mean: {df['movie_plot'].str.len().mean():.0f} chars")
    ratio = (df['book_summary'].str.len() / df['movie_plot'].str.len()).mean()
    log(f"Length ratio  — {ratio:.1f}x  (target: <3x; TMDB baseline: 10x)")

    log("\nVerification breakdown:")
    for method, cnt in df["verification_method"].value_counts().items():
        log(f"  {method:<35} {cnt:4d} ({cnt/len(df)*100:.1f}%)")

    print("\nSample pairs (highest plot length):")
    print(df[["book_title","movie_title","verification_method","plot_length_chars"]]
          .sort_values("plot_length_chars", ascending=False)
          .head(10).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# INITIALISE STATE from CMU dataset (run once before Pass 1)
# ──────────────────────────────────────────────────────────────────────────────

def initialise_state():
    cmu = pd.read_csv(f"{DATA_DIR}cmu_cleaned.csv")

    title_col   = next((c for c in cmu.columns if 'title'   in c.lower()), None)
    summary_col = next((c for c in cmu.columns
                        if any(x in c.lower() for x in
                               ['summary','plot','text','synopsis'])), None)
    genre_col   = next((c for c in cmu.columns if 'genre'   in c.lower()), None)
    author_col  = next((c for c in cmu.columns
                        if any(x in c.lower() for x in
                               ['author','writer','creator'])), None)

    cmu = cmu.dropna(subset=[title_col, summary_col])
    cmu = cmu[cmu[summary_col].str.len() >= 200].reset_index(drop=True)

    # Priority sort: books with confirmed film pairs first
    priority = set()
    bmp_path = f"{DATA_DIR}book_movie_pairs.csv"
    if os.path.exists(bmp_path):
        bmp = pd.read_csv(bmp_path)
        priority = set(bmp["book_title"].str.lower().str.strip())

    cmu["_pri"] = cmu[title_col].str.lower().str.strip().isin(priority).astype(int)
    cmu = cmu.sort_values("_pri", ascending=False).head(MAX_BOOKS).reset_index(drop=True)

    state = {}
    for _, row in cmu.iterrows():
        book_title = str(row[title_col]).strip()
        key = state_key(book_title)
        state[key] = {
            "book_title":   book_title,
            "author":       str(row[author_col]).strip() if author_col and pd.notna(row.get(author_col)) else None,
            "book_summary": str(row[summary_col]),
            "book_genres":  str(row[genre_col]) if genre_col and pd.notna(row.get(genre_col)) else "[]",
            "status":       "pending",
            "pass_tried":   0,
            "film_page":    None,
            "strategy":     None,
            "based_on":     None,
            "movie_plot":   None,
            "plot_length":  None,
            "_pri":         int(row["_pri"]),
        }

    save_state(state)
    n_priority = sum(1 for v in state.values() if v.get("_pri", 0))
    log(f"Initialised state: {len(state)} books ({n_priority} priority)")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass",  type=int, choices=[1,2,3,4],
                        dest="run_pass", help="Which pass to run")
    parser.add_argument("--init",  action="store_true",
                        help="Initialise state from CMU dataset (run first)")
    parser.add_argument("--build", action="store_true",
                        help="Build final CSV from current state")
    parser.add_argument("--status",action="store_true",
                        help="Print current state summary and exit")
    args = parser.parse_args()

    if args.init:
        state = initialise_state()
        print_state_summary(state, "After initialisation")

    elif args.status:
        state = load_state()
        print_state_summary(state, "Current scraping state")

    elif args.build:
        state = load_state()
        build_dataset(state)

    elif args.run_pass:
        state = load_state()
        if not state:
            print("State is empty. Run --init first.")
            exit(1)

        print_state_summary(state, f"Before Pass {args.run_pass}")

        if args.run_pass == 1:
            state = run_pass1(state)
        elif args.run_pass == 2:
            state = run_pass2(state)
        elif args.run_pass == 3:
            state = run_pass3(state)
        elif args.run_pass == 4:
            state = run_pass4(state)

        print_state_summary(state, f"After Pass {args.run_pass}")

        found = sum(1 for v in state.values() if v["status"] == "found")
        pending = sum(1 for v in state.values() if v["status"] == "pending")
        print(f"\n{'='*60}")
        print(f"  STOP/CONTINUE DECISION")
        print(f"{'='*60}")
        print(f"  Pairs found so far: {found}")
        print(f"  Books still pending: {pending}")
        print(f"  Recommendation:")
        if found >= 1000:
            print(f"  ✅ {found} pairs is sufficient for robust mismatch analysis.")
            print(f"     You can stop here and run --build, OR continue for more.")
        elif found >= 500:
            print(f"  ⚠️  {found} pairs is workable but more would improve reliability.")
            print(f"     Consider running the next pass.")
        else:
            print(f"  ❌ {found} pairs may be too few. Run the next pass.")
        if args.run_pass < 4:
            print(f"\n  Next: python {os.path.basename(__file__)} --pass {args.run_pass+1}")
        print(f"  Build: python {os.path.basename(__file__)} --build")
        print(f"{'='*60}")
    else:
        parser.print_help()