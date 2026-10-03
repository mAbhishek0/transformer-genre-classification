# =============================================================================
# MULTI-PASS WIKIPEDIA BOOK-TO-MOVIE SCRAPER — V4
#
# KEY IMPROVEMENTS OVER V3:
#   • Batch QID lookup    — 50 pages per Wikipedia API call (was 1)
#   • Batch SPARQL        — 50 QIDs per query via VALUES clause (was 1)
#   • Batch wikitext      — 20 film pages per API call for plot extraction
#   • AlertTracker        — every silent failure is named, counted, and logged
#                           with the remediation action being taken
#   • SPARQL via POST     — to query-main.wikidata.org (less loaded endpoint)
#   • Exponential backoff — for SPARQL 429 / 503 with Retry-After support
#   • SPARQL failure flag — distinct from "no results"; retryable via --retry-sparql
#   • Improved book search— fallback queries, subtitle stripping, disambiguation
#                           detection, non-novel suffix detection
#   • Pass 1 pipeline     — 4-phase batch flow; Phase A search results saved to
#                           state so interrupted runs never re-search
#
# ESTIMATED TIME (3000 books):
#   V3: ~4–5 hours   V4: ~55–80 minutes
#
# USAGE:
#   python wikipedia_scraper_v4.py --init
#   python wikipedia_scraper_v4.py --pass 1
#   python wikipedia_scraper_v4.py --pass 2
#   python wikipedia_scraper_v4.py --pass 3
#   python wikipedia_scraper_v4.py --pass 4
#   python wikipedia_scraper_v4.py --retry-sparql   # retry SPARQL failures from Pass 1
#   python wikipedia_scraper_v4.py --build
#   python wikipedia_scraper_v4.py --status
#
# pip install requests rapidfuzz pandas numpy
# =============================================================================


# MULTI-PASS WIKIPEDIA BOOK-TO-MOVIE SCRAPER — V4
#
# KEY IMPROVEMENTS OVER V3:
#   • Batch QID lookup    — 50 pages per Wikipedia API call (was 1)
#   • Batch SPARQL        — 50 QIDs per query via VALUES clause (was 1)
#   • Batch wikitext      — 20 film pages per API call for plot extraction
#   • AlertTracker        — every silent failure is named, counted, and logged
#                           with the remediation action being taken
#   • SPARQL via POST     — to query-main.wikidata.org (less loaded endpoint)
#   • Exponential backoff — for SPARQL 429 / 503 with Retry-After support
#   • SPARQL failure flag — distinct from "no results"; retryable via --retry-sparql
#   • Improved book search— fallback queries, subtitle stripping, disambiguation
#                           detection, non-novel suffix detection
#   • Pass 1 pipeline     — 4-phase batch flow; Phase A search results saved to
#                           state so interrupted runs never re-search
# USAGE:
#   python wikipedia_scraper_v4.py --init
#   python wikipedia_scraper_v4.py --pass 1
#   python wikipedia_scraper_v4.py --pass 2
#   python wikipedia_scraper_v4.py --pass 3
#   python wikipedia_scraper_v4.py --pass 4
#   python wikipedia_scraper_v4.py --retry-sparql   # retry SPARQL failures from Pass 1
#   python wikipedia_scraper_v4.py --build
#   python wikipedia_scraper_v4.py --status
#
# pip install requests rapidfuzz pandas numpy

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR    = "./data/"
STATE_FILE  = "./data/scrape_state.json"
OUTPUT_FILE = "./data/wiki_book_movie_pairs_v4.csv"
LOG_FILE    = "./data/scrape_v4_log.txt"

RATE_LIMIT        = 1.0    # seconds between Wikipedia API calls
SPARQL_RATE_LIMIT = 2.5    # seconds between SPARQL calls (stricter endpoint)

MAX_BOOKS  = 3000
MIN_PLOT   = 300

SPARQL_BATCH_SIZE   = 50   # QIDs per SPARQL VALUES query
QID_BATCH_SIZE      = 50   # page titles per Wikipedia pageprops call
WIKITEXT_BATCH_SIZE = 20   # pages per batch wikitext call

BOOK_PAGE_FUZZY_THRESHOLD = 65
FILM_PAGE_FUZZY_THRESHOLD = 55

WIKI_API        = "https://en.wikipedia.org/w/api.php"
WIKIDATA_SPARQL = "https://query-main.wikidata.org/sparql"   # main subgraph, less load

HEADERS = {
    "User-Agent": "NLP-Academic-Research/4.0 (book-movie-adaptation-study; "
                  "contact: research@example.com)"
}

ADAPTATION_KEYWORDS = {
    'adaptations', 'film adaptation', 'film adaptations', 'film', 'films',
    'movie', 'movies', 'in other media', 'media adaptations',
    'theatrical adaptations', 'television and film', 'film and television',
    'screen adaptations', 'legacy', 'cultural impact'
}

# ──────────────────────────────────────────────────────────────────────────────
# ALERT TRACKER
# Centralises every silent failure: names it, counts it, logs the remediation
# action being taken so nothing disappears into the void.
# ──────────────────────────────────────────────────────────────────────────────

class AlertTracker:
    # alert_type -> (severity, description)
    TYPES = {
        'SEARCH_EMPTY'       : ('WARN',  'Wikipedia search returned 0 results'),
        'FUZZY_MISS'         : ('WARN',  'Best candidate below fuzzy threshold'),
        'DISAMBIGUATION'     : ('WARN',  'Matched page is a disambiguation page'),
        'WRONG_ENTITY'       : ('ERROR', 'QID entity is not a literary work'),
        'QID_NONE'           : ('WARN',  'Page exists but has no Wikidata QID'),
        'QID_MISSING_PAGE'   : ('WARN',  'Wikipedia page does not exist'),
        'SPARQL_429'         : ('ERROR', 'SPARQL rate-limited (HTTP 429)'),
        'SPARQL_503'         : ('ERROR', 'SPARQL service unavailable (HTTP 503)'),
        'SPARQL_TIMEOUT'     : ('ERROR', 'SPARQL query timed out'),
        'SPARQL_EMPTY'       : ('INFO',  'SPARQL found no P144 film for QID'),
        'SPARQL_BATCH_FAIL'  : ('ERROR', 'Entire SPARQL batch failed after retries'),
        'PLOT_TOO_SHORT'     : ('WARN',  f'Plot below MIN_PLOT={MIN_PLOT} chars'),
        'PLOT_MISSING'       : ('WARN',  'No plot section found on film page'),
        'WIKI_API_FAIL'      : ('ERROR', 'Wikipedia API call failed after retries'),
        'BATCH_QID_PARTIAL'  : ('WARN',  'Some pages in QID batch returned no data'),
        'SLOW_BOOK'          : ('WARN',  'Single book taking unusually long'),
        'THROUGHPUT_DROP'    : ('ERROR', 'Rolling avg time/book spiked — likely throttled'),
    }

    def __init__(self):
        self.counts      = defaultdict(int)
        self.recent_times = []       # rolling window for throughput alerts
        self._pass_start  = time.time()

    def record(self, alert_type, detail="", book_title="", remediation=""):
        """Log a named alert with optional book context and what we're doing about it."""
        self.counts[alert_type] += 1
        sev, desc = self.TYPES.get(alert_type, ('WARN', alert_type))
        parts = [f"[{sev}][{alert_type}]"]
        if book_title:
            parts.append(f"'{book_title}'")
        if detail:
            parts.append(f"→ {detail}")
        if remediation:
            parts.append(f"| ACTION: {remediation}")
        log("  " + " ".join(parts), also_print=(sev in ('ERROR', 'WARN')))

    def tick(self, elapsed_seconds, book_title):
        """Call after each book. Alerts if throughput has degraded."""
        self.recent_times.append(elapsed_seconds)
        if len(self.recent_times) > 10:
            self.recent_times.pop(0)
        if elapsed_seconds > 30:
            self.record('SLOW_BOOK', f"{elapsed_seconds:.1f}s",
                        book_title=book_title,
                        remediation="Continuing — check network or SPARQL backoff logs")
        if len(self.recent_times) == 10:
            avg = sum(self.recent_times) / 10
            if avg > 20:
                self.record('THROUGHPUT_DROP',
                            f"10-book avg = {avg:.1f}s/book",
                            remediation="Possible rate limit — SPARQL backoff active")

    def summary(self, label="Alert summary"):
        lines = [f"\n{'─'*55}", f"  {label}"]
        total = sum(self.counts.values())
        if total == 0:
            lines.append("  No alerts recorded ")
        else:
            for atype, cnt in sorted(self.counts.items(), key=lambda x: -x[1]):
                sev, desc = self.TYPES.get(atype, ('?', atype))
                lines.append(f"  [{sev:5s}] {atype:<25} {cnt:4d}  — {desc}")
        elapsed = (time.time() - self._pass_start) / 60
        lines.append(f"\n  Pass wall-time: {elapsed:.1f} min")
        lines.append(f"{'─'*55}")
        msg = "\n".join(lines)
        log(msg)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def log(msg, also_print=True):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
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
    return {
        k: v for k, v in state.items()
        if v["status"] == "pending" and v.get("pass_tried", 0) < pass_number
    }

def get_sparql_failures(state):
    return {
        k: v for k, v in state.items()
        if v.get("sparql_failed") and v["status"] == "pending"
    }

def print_state_summary(state, label="Current State"):
    found     = [v for v in state.values() if v["status"] == "found"]
    pending   = [v for v in state.values() if v["status"] == "pending"]
    exhausted = [v for v in state.values() if v["status"] == "exhausted"]
    sparql_fail = [v for v in state.values() if v.get("sparql_failed")]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total books:      {len(state)}")
    print(f"   Found:         {len(found)}")
    print(f"   Pending:       {len(pending)}")
    print(f"    Exhausted:     {len(exhausted)}")
    print(f"    SPARQL failed: {len(sparql_fail)}  (run --retry-sparql)")

    if found:
        by_strategy = defaultdict(int)
        for v in found:
            by_strategy[v.get("strategy", "unknown")] += 1
        print(f"\n  Matches by strategy:")
        for s, c in sorted(by_strategy.items(), key=lambda x: -x[1]):
            print(f"    {s:<38} {c:4d} ({c/len(found)*100:.1f}%)")
        plots = [v["plot_length"] for v in found if v.get("plot_length")]
        if plots:
            print(f"\n  Plot length — mean:{np.mean(plots):.0f} "
                  f"median:{np.median(plots):.0f} "
                  f"min:{min(plots)} max:{max(plots)}")
    print(f"{'='*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# CORE API HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def api_get(params, retries=3):
    params["format"] = "json"
    params["utf8"]   = 1
    for attempt in range(retries):
        try:
            r = requests.get(WIKI_API, params=params,
                             headers=HEADERS, timeout=15)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10 * (attempt + 1)))
                log(f"  [WARN][WIKI_429] sleeping {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            time.sleep(3 * (attempt + 1))
        except requests.exceptions.HTTPError:
            return None
        except Exception as e:
            log(f"  [WARN][WIKI_API] unexpected error: {e}", also_print=False)
            time.sleep(2)
    return None

_last_sparql_time = 0.0

def sparql_post(query, alert, retries=5):
    """
    POST to query-main.wikidata.org with exponential backoff.
    Returns list of bindings, or None if the call itself failed (retryable).
    Returns [] for genuine no-results (not retryable).
    """
    global _last_sparql_time

    gap = time.time() - _last_sparql_time
    if gap < SPARQL_RATE_LIMIT:
        time.sleep(SPARQL_RATE_LIMIT - gap)

    for attempt in range(retries):
        try:
            r = requests.post(
                WIKIDATA_SPARQL,
                data={"query": query, "format": "json"},
                headers={**HEADERS, "Accept": "application/sparql-results+json"},
                timeout=30
            )
            _last_sparql_time = time.time()

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 15 * (2 ** attempt)))
                alert.record('SPARQL_429',
                             f"sleeping {wait}s (attempt {attempt+1}/{retries})",
                             remediation=f"Backing off {wait}s then retrying")
                time.sleep(wait)
                continue

            if r.status_code == 503:
                wait = 20 * (2 ** attempt)
                alert.record('SPARQL_503',
                             f"sleeping {wait}s (attempt {attempt+1}/{retries})",
                             remediation=f"Service unavailable, waiting {wait}s")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json().get("results", {}).get("bindings", [])

        except requests.exceptions.Timeout:
            wait = 10 * (attempt + 1)
            alert.record('SPARQL_TIMEOUT',
                         f"attempt {attempt+1}/{retries}",
                         remediation=f"Retrying after {wait}s")
            time.sleep(wait)
        except Exception as e:
            log(f"  [ERROR][SPARQL] unexpected: {e}", also_print=True)
            time.sleep(5 * (attempt + 1))

    alert.record('SPARQL_BATCH_FAIL',
                 f"after {retries} attempts",
                 remediation="Marking books as sparql_failed; run --retry-sparql later")
    return None

def wiki_search(query, n=7):
    data = api_get({"action": "query", "list": "search",
                    "srsearch": query, "srlimit": n})
    if not data:
        return []
    return [r["title"] for r in data.get("query", {}).get("search", [])]

# ──────────────────────────────────────────────────────────────────────────────
# BATCH API HELPERS  ← core speedup vs V3
# ──────────────────────────────────────────────────────────────────────────────

def batch_wiki_qid_lookup(page_titles, alert):
    """
    Fetch Wikidata QIDs for up to QID_BATCH_SIZE pages per API call.
    Also detects disambiguation pages in the same call.
    Returns: {page_title: qid_or_None}
    A value of None means: page exists but has no QID.
    Missing keys mean: page doesn't exist on Wikipedia.
    """
    results = {}
    unique = list(dict.fromkeys(t for t in page_titles if t))

    for i in range(0, len(unique), QID_BATCH_SIZE):
        batch = unique[i:i + QID_BATCH_SIZE]
        data = api_get({
            "action":      "query",
            "titles":      "|".join(batch),
            "prop":        "pageprops|categories",
            "ppprop":      "wikibase_item",
            "cllimit":     "5",
            "clcategories":"Category:Disambiguation pages",
        })
        if not data:
            alert.record('WIKI_API_FAIL',
                         f"QID batch {i//QID_BATCH_SIZE+1} failed",
                         remediation="Pages in this batch will be skipped (no QID)")
            continue

        for page in data.get("query", {}).get("pages", {}).values():
            title    = page.get("title", "")
            page_id  = page.get("pageid", -1)

            if page_id < 0 or "missing" in page:
                alert.record('QID_MISSING_PAGE', f"'{title}'",
                             remediation="Skipping — will try alternate search query")
                continue

            cats = [c.get("title","") for c in page.get("categories", [])]
            if any("Disambiguation" in c for c in cats):
                alert.record('DISAMBIGUATION', f"'{title}'",
                             remediation="Skipping this candidate, trying next search result")
                continue

            qid = page.get("pageprops", {}).get("wikibase_item")
            if not qid:
                alert.record('QID_NONE', f"'{title}'",
                             remediation="Page exists but has no Wikidata item — skipping")
            results[title] = qid   # None if no QID but page exists

        time.sleep(RATE_LIMIT)

    return results


def batch_sparql_p144(key_to_qid, alert):
    """
    Run P144 SPARQL queries in batches of SPARQL_BATCH_SIZE.
    Returns: {book_key: [film_page, ...]}
      - None value   → SPARQL call itself failed (mark sparql_failed, retryable)
      - []  value    → no P144 film found (genuine miss, move to Pass 2)
      - [..] value   → film page(s) found
    """
    all_results = {}
    items       = list(key_to_qid.items())
    total_batches = (len(items) + SPARQL_BATCH_SIZE - 1) // SPARQL_BATCH_SIZE

    for b_idx in range(0, len(items), SPARQL_BATCH_SIZE):
        batch       = items[b_idx:b_idx + SPARQL_BATCH_SIZE]
        key_qid     = dict(batch)
        qid_key     = {qid: key for key, qid in key_qid.items()}
        values_str  = " ".join(f"wd:{qid}" for qid in qid_key)
        batch_num   = b_idx // SPARQL_BATCH_SIZE + 1

        log(f"   SPARQL batch {batch_num}/{total_batches} "
            f"({len(batch)} QIDs)...", also_print=True)

        sparql = f"""
        SELECT ?book ?film ?filmLabel ?article WHERE {{
          VALUES ?book {{ {values_str} }}
          ?film wdt:P144 ?book ;
                wdt:P31  wd:Q11424 .
          OPTIONAL {{
            ?article schema:about ?film ;
                     schema:isPartOf <https://en.wikipedia.org/> .
          }}
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
          }}
        }} LIMIT {len(batch) * 5}
        """

        bindings = sparql_post(sparql, alert)

        if bindings is None:
            # Hard failure — mark all books in this batch as sparql_failed
            for key in key_qid:
                all_results[key] = None
            log(f"   Batch {batch_num} FAILED — {len(batch)} books marked "
                f"sparql_failed", also_print=True)
            continue

        # Group results by book
        batch_results = defaultdict(list)
        for b in bindings:
            book_qid    = b.get("book",      {}).get("value", "").split("/")[-1]
            film_label  = b.get("filmLabel", {}).get("value", "")
            article_url = b.get("article",   {}).get("value", "")
            film_page   = (
                article_url
                .replace("https://en.wikipedia.org/wiki/", "")
                .replace("_", " ")
                if article_url else film_label
            )
            key = qid_key.get(book_qid)
            if key and film_page:
                batch_results[key].append(film_page)

        hits = sum(1 for v in batch_results.values() if v)
        log(f"  Batch {batch_num} done: {hits}/{len(batch)} books have P144 films",
            also_print=True)

        for key in key_qid:
            all_results[key] = batch_results.get(key, [])
            if not batch_results.get(key):
                alert.record('SPARQL_EMPTY',
                             detail=f"QID {key_qid[key]}",
                             remediation="No P144 film in Wikidata — will try Pass 2")

    return all_results


def batch_wiki_wikitext(page_titles, alert):
    """
    Fetch raw wikitext for multiple pages in one API call.
    Returns: {normalised_title: wikitext}
    """
    results = {}
    unique  = list(dict.fromkeys(t for t in page_titles if t))

    for i in range(0, len(unique), WIKITEXT_BATCH_SIZE):
        batch = unique[i:i + WIKITEXT_BATCH_SIZE]
        data  = api_get({
            "action":  "query",
            "titles":  "|".join(batch),
            "prop":    "revisions",
            "rvprop":  "content",
            "rvslots": "main",
        })
        if not data:
            alert.record('WIKI_API_FAIL',
                         f"wikitext batch {i//WIKITEXT_BATCH_SIZE+1} failed",
                         remediation="Films in this batch will have no plot extracted")
            continue

        # Wikipedia normalises titles; map normalised → original
        norm_map = {}
        for n in data.get("query", {}).get("normalized", []):
            norm_map[n["to"]] = n["from"]

        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            if "missing" in page:
                continue
            revs = page.get("revisions", [])
            if not revs:
                continue
            wt = (revs[0].get("slots", {}).get("main", {}).get("*")
                  or revs[0].get("*", ""))
            if wt:
                results[title] = wt
                # also store under original name if normalised
                orig = norm_map.get(title)
                if orig:
                    results[orig] = wt

        time.sleep(RATE_LIMIT)

    return results

# ──────────────────────────────────────────────────────────────────────────────
# BOOK SEARCH HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_search_suffix(book_title, genres_str=""):
    g = genres_str.lower()
    t = book_title.lower()
    if any(x in g for x in ["non-fiction", "essay", "journalism", "biography"]):
        return "book"
    if any(x in g for x in ["play", "drama", "theatre", "theater"]):
        return "play"
    if any(x in g for x in ["poem", "poetry", "epic", "verse"]):
        return "poem"
    if any(x in t for x in ["collected", "complete works", "poems", "plays"]):
        return "book"
    return "novel"

def strip_wiki_qualifier(title):
    """Remove trailing '(novel)', '(1984 novel)', '(book)' etc. before fuzzy match."""
    return re.sub(
        r'\s*\([^)]*(?:novel|book|play|poem|story|tale|film|movie|series)[^)]*\)\s*$',
        '', title, flags=re.IGNORECASE
    ).strip()

def search_book_wikipedia(book_title, genres_str, author, alert):
    """
    Try multiple query variants to find the book's Wikipedia page.
    Returns list of candidate page titles (already filtered by fuzzy score).
    """
    suffix      = get_search_suffix(book_title, genres_str)
    # Strip subtitle after ; or : for cleaner search
    clean_title = re.sub(r'\s*[;:,]\s.*$', '', book_title).strip()

    queries = list(dict.fromkeys([               # ordered, deduplicated
        f"{book_title} {suffix}",
        f"{clean_title} {suffix}",
        f"{clean_title}",
        f"{book_title} book",
        f"{book_title} {author}" if author else None,
    ]))
    queries = [q for q in queries if q]

    all_candidates = []
    query_used     = None

    for q in queries:
        results = wiki_search(q, n=7)
        time.sleep(RATE_LIMIT)
        if not results:
            continue
        candidates = [
            r for r in results
            if fuzz.token_sort_ratio(
                book_title.lower(),
                strip_wiki_qualifier(r).lower()
            ) >= BOOK_PAGE_FUZZY_THRESHOLD
        ]
        if candidates:
            all_candidates = candidates
            query_used     = q
            break

    if not all_candidates:
        # Keep top-3 raw results for QID lookup even if fuzzy threshold not met
        # (pass-through — we'll recheck after QID lookup)
        fallback = wiki_search(f"{clean_title}", n=5)
        time.sleep(RATE_LIMIT)
        if fallback:
            all_candidates = fallback[:3]
            alert.record('FUZZY_MISS',
                         f"'{book_title}' — best below threshold, using raw top-3",
                         remediation="Will attempt QID lookup on raw results")
        else:
            alert.record('SEARCH_EMPTY',
                         f"'{book_title}'",
                         remediation="No Wikipedia search results — book skipped in Pass 1")

    return all_candidates, query_used

# ──────────────────────────────────────────────────────────────────────────────
# INFOBOX VERIFICATION  (unchanged from V3, used in Passes 2–4)
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

def verify_infobox(film_page, book_title, author, wikitext=None):
    """
    Returns (confirmed: bool, based_on_text: str).
    Accepts pre-fetched wikitext to avoid redundant API calls.
    """
    if wikitext is None:
        wikitext = _single_wikitext(film_page)
    if not wikitext:
        return False, ""
    based_on = (extract_infobox_field(wikitext, "based_on") or
                extract_infobox_field(wikitext, "based on") or
                extract_infobox_field(wikitext, "screenplay"))
    book_words = set(book_title.lower().split())
    overlap    = len(book_words & set(based_on.split())) / max(len(book_words), 1)
    confirmed  = overlap >= 0.5
    if not confirmed and author:
        author_parts = [p for p in author.lower().split() if len(p) > 3]
        confirmed    = any(p in based_on for p in author_parts)
    return confirmed, based_on

def _single_wikitext(page_title):
    data = api_get({"action": "parse", "page": page_title, "prop": "wikitext"})
    time.sleep(RATE_LIMIT)
    if not data or "error" in data:
        return ""
    return data.get("parse", {}).get("wikitext", {}).get("*", "")

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

PLOT_HEADER_RE = re.compile(
    r'==\s*(?:Plot|Synopsis|Story|Narrative|Plot\s+summary|Film\s+synopsis)\s*=='
    r'\s*(.*?)(?=\n==|\Z)',
    re.DOTALL | re.IGNORECASE
)

def extract_plot_from_wikitext(wikitext, alert=None, film_page=""):
    """Extract plot from pre-fetched wikitext. No API call needed."""
    if not wikitext:
        if alert:
            alert.record('PLOT_MISSING', f"'{film_page}' — empty wikitext",
                         remediation="Film skipped; trying next Wikidata result")
        return None
    m = PLOT_HEADER_RE.search(wikitext)
    if m:
        text = clean_wikitext(m.group(1))
        if len(text) >= MIN_PLOT:
            return text
        if alert:
            alert.record('PLOT_TOO_SHORT',
                         f"'{film_page}' — {len(text)} chars (need {MIN_PLOT})",
                         remediation="Trying next film candidate")
        return None
    if alert:
        alert.record('PLOT_MISSING', f"'{film_page}' — no Plot/Synopsis section",
                     remediation="Trying next film candidate")
    return None

def extract_plot_via_api(film_page, alert):
    """
    Fallback for Passes 2–4 where we may not have pre-fetched wikitext.
    Tries section API first, then full wikitext regex.
    """
    # Try sections API
    data = api_get({"action": "parse", "page": film_page, "prop": "sections"})
    time.sleep(RATE_LIMIT)
    sections = data.get("parse", {}).get("sections", []) if data else []

    PLOT_NAMES = {"plot","synopsis","story","narrative","plot summary","film synopsis"}
    plot_sec = next(
        (s for s in sections if s.get("line","").lower().strip() in PLOT_NAMES), None)

    if plot_sec:
        sec_data = api_get({"action": "parse", "page": film_page,
                            "prop": "wikitext", "section": plot_sec["index"]})
        time.sleep(RATE_LIMIT)
        if sec_data:
            raw = sec_data.get("parse", {}).get("wikitext", {}).get("*", "")
            text = clean_wikitext(raw)
            if len(text) >= MIN_PLOT:
                return text

    # Fallback: full wikitext + regex
    wt = _single_wikitext(film_page)
    return extract_plot_from_wikitext(wt, alert, film_page)

# ──────────────────────────────────────────────────────────────────────────────
# PASS 1 — WIKIDATA P144  (batch-optimised pipeline)
#
# Phase A: Wikipedia search for each book  (1 API call/book — unavoidable)
# Phase B: Batch QID lookup               (1 call per 50 pages)
# Phase C: Batch SPARQL P144              (1 SPARQL call per 50 QIDs)
# Phase D: Batch wikitext for film pages  (1 API call per 20 films)
# Phase E: Plot extraction + state update (local — no API calls)
# ──────────────────────────────────────────────────────────────────────────────

def run_pass1(state):
    alert   = AlertTracker()
    pending = get_pending(state, pass_number=1)

    # Resume support: skip Phase A for books that already have cached candidates
    need_search  = {k: v for k, v in pending.items() if not v.get("_candidates")}
    have_search  = {k: v for k, v in pending.items() if     v.get("_candidates")}

    log(f"\n{'='*60}")
    log(f"PASS 1 — Wikidata P144 (batch pipeline)")
    log(f"  {len(pending)} books pending | "
        f"{len(need_search)} need search | {len(have_search)} cached")
    log(f"  Est. time: {len(need_search)*1.2/60:.0f} min (search) + "
        f"{len(pending)//SPARQL_BATCH_SIZE * SPARQL_RATE_LIMIT / 60:.0f} min (SPARQL)")
    log(f"{'='*60}")

    # ── Phase A: Wikipedia search ──────────────────────────────────────────
    log(f"\n─── Phase A: Searching Wikipedia ({len(need_search)} books)...")
    for i, (key, entry) in enumerate(need_search.items()):
        book_title = entry["book_title"]
        if i % 100 == 0 and i > 0:
            log(f"  Search progress: {i}/{len(need_search)}")
            save_state(state)

        candidates, query_used = search_book_wikipedia(
            book_title,
            entry.get("book_genres", ""),
            entry.get("author", ""),
            alert
        )
        state[key]["_candidates"]   = candidates
        state[key]["_search_query"] = query_used
        log(f"  [{i+1:4d}] '{book_title}' → {len(candidates)} candidates "
            f"(query: '{query_used}')", also_print=False)

    save_state(state)
    log(f"  Phase A done. Saving state.")

    # ── Phase B: Batch QID lookup ──────────────────────────────────────────
    log(f"\n─── Phase B: Batch QID lookup...")
    all_candidate_pages = list(dict.fromkeys(
        p for v in pending.values() for p in v.get("_candidates", [])
    ))
    log(f"  {len(all_candidate_pages)} unique candidate pages → "
        f"{(len(all_candidate_pages)+QID_BATCH_SIZE-1)//QID_BATCH_SIZE} API calls")

    page_qid_map = batch_wiki_qid_lookup(all_candidate_pages, alert)
    log(f"  Phase B done: {sum(1 for q in page_qid_map.values() if q)} QIDs found "
        f"out of {len(all_candidate_pages)} pages")

    # ── Phase C: Match QIDs to books ──────────────────────────────────────
    log(f"\n─── Phase C: Matching QIDs to books...")
    book_key_to_qid = {}
    no_qid_count    = 0

    for key, entry in pending.items():
        book_title  = entry["book_title"]
        candidates  = entry.get("_candidates", [])
        matched_qid = None

        for page in candidates:
            page_clean = strip_wiki_qualifier(page)
            score = fuzz.token_sort_ratio(book_title.lower(), page_clean.lower())
            qid   = page_qid_map.get(page)
            if score >= BOOK_PAGE_FUZZY_THRESHOLD and qid:
                matched_qid = qid
                log(f"  '{book_title}' → '{page}' (score={score}) QID={qid}",
                    also_print=False)
                break

        state[key]["pass_tried"] = 1

        if matched_qid:
            book_key_to_qid[key] = matched_qid
        else:
            no_qid_count += 1
            if candidates:
                best_page  = candidates[0]
                best_score = fuzz.token_sort_ratio(
                    book_title.lower(),
                    strip_wiki_qualifier(best_page).lower()
                )
                alert.record('FUZZY_MISS',
                             f"'{book_title}' best='{best_page}' score={best_score}",
                             remediation="No QID — will be tried in Pass 2")

    log(f"  Phase C done: {len(book_key_to_qid)} books have QIDs, "
        f"{no_qid_count} have no QID (→ Pass 2)")

    # ── Phase D: Batch SPARQL ──────────────────────────────────────────────
    log(f"\n─── Phase D: Batch SPARQL P144 "
        f"({len(book_key_to_qid)} QIDs in "
        f"{(len(book_key_to_qid)+SPARQL_BATCH_SIZE-1)//SPARQL_BATCH_SIZE} batches)...")

    film_results = batch_sparql_p144(book_key_to_qid, alert)

    # Update sparql_failed flag
    sparql_fail_count = 0
    for key, result in film_results.items():
        if result is None:
            state[key]["sparql_failed"] = True
            sparql_fail_count += 1
        else:
            state[key]["sparql_failed"] = False

    total_film_candidates = sum(
        len(v) for v in film_results.values() if v
    )
    log(f"  Phase D done: {total_film_candidates} film candidates, "
        f"{sparql_fail_count} batches failed (run --retry-sparql)")

    # ── Phase E: Batch wikitext + plot extraction ──────────────────────────
    log(f"\n─── Phase E: Batch wikitext + plot extraction...")
    all_film_pages = list(dict.fromkeys(
        fp for pages in film_results.values()
        if pages for fp in pages
    ))
    log(f"  Fetching wikitext for {len(all_film_pages)} film pages "
        f"in {(len(all_film_pages)+WIKITEXT_BATCH_SIZE-1)//WIKITEXT_BATCH_SIZE} batches...")

    wikitext_cache = batch_wiki_wikitext(all_film_pages, alert)
    log(f"  Wikitext fetched for {len(wikitext_cache)} pages")

    found_this_pass = 0
    for key, film_pages in film_results.items():
        if film_pages is None or not film_pages:
            continue
        entry = state[key]
        for fp in film_pages:
            wt   = wikitext_cache.get(fp, "")
            plot = extract_plot_from_wikitext(wt, alert, fp)
            if plot:
                state[key].update({
                    "status":      "found",
                    "film_page":   fp,
                    "strategy":    "pass1_wikidata_P144_batch",
                    "based_on":    f"wikidata QID {book_key_to_qid.get(key,'')}",
                    "movie_plot":  plot,
                    "plot_length": len(plot),
                    "sparql_failed": False,
                })
                found_this_pass += 1
                log(f"  '{entry['book_title']}' → '{fp}' ({len(plot)} chars)")
                break

    save_state(state)
    log(f"\nPass 1 complete. Found: {found_this_pass} new pairs.")
    alert.summary("Pass 1 alert summary")
    return state, alert

# ──────────────────────────────────────────────────────────────────────────────
# PASS 2 — BOOK PAGE → ADAPTATIONS SECTION
# Speedup: batch wikitext for infobox verification + plot in one pass
# ──────────────────────────────────────────────────────────────────────────────

def run_pass2(state):
    alert   = AlertTracker()
    pending = get_pending(state, pass_number=2)
    log(f"\n{'='*60}")
    log(f"PASS 2 — Book page adaptations | {len(pending)} books pending")
    log(f"{'='*60}")

    found_this_pass = 0

    for i, (key, entry) in enumerate(pending.items()):
        book_title = entry["book_title"]
        author     = entry.get("author")
        book_start = time.time()

        if i % 50 == 0 and i > 0:
            log(f"  Progress: {i}/{len(pending)} | found: {found_this_pass}")
            save_state(state)

        log(f"  [{i+1:4d}/{len(pending)}] '{book_title}'", also_print=True)
        state[key]["pass_tried"] = 2

        # Find book's Wikipedia page
        candidates, _ = search_book_wikipedia(
            book_title, entry.get("book_genres",""), author, alert)
        book_page = candidates[0] if candidates else None

        if not book_page:
            alert.record('SEARCH_EMPTY', f"'{book_title}'",
                         remediation="Skipping — no book page found in Pass 2")
            alert.tick(time.time() - book_start, book_title)
            continue

        # Get adaptation sections
        data = api_get({"action": "parse", "page": book_page, "prop": "sections"})
        time.sleep(RATE_LIMIT)
        sections = data.get("parse", {}).get("sections", []) if data else []

        adapt_sections = [s for s in sections
                          if any(kw in s.get("line","").lower()
                                 for kw in ADAPTATION_KEYWORDS)]

        candidate_links = []
        if adapt_sections:
            for sec in adapt_sections:
                sec_data = api_get({"action": "parse", "page": book_page,
                                    "prop": "wikitext", "section": sec["index"]})
                time.sleep(RATE_LIMIT)
                if sec_data:
                    raw   = sec_data.get("parse",{}).get("wikitext",{}).get("*","")
                    links = re.findall(r'\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]', raw)
                    candidate_links.extend(
                        l.strip() for l in links
                        if not any(s in l.lower()
                                   for s in ['category:','file:','image:','template:'])
                    )
        else:
            # Fall back to film links in full wikitext
            wt    = _single_wikitext(book_page)
            links = re.findall(r'\[\[([^\]|]+\((?:film|movie)\)[^\]]*)\]\]',
                               wt, re.IGNORECASE)
            candidate_links.extend(l.strip() for l in links)

        if not candidate_links:
            log(f"         No film links on book page '{book_page}'", also_print=False)
            alert.tick(time.time() - book_start, book_title)
            continue

        # Batch fetch wikitext for all candidates at once
        deduped = list(dict.fromkeys(candidate_links))
        wt_cache = batch_wiki_wikitext(deduped, alert)

        found = False
        for link in deduped:
            wt        = wt_cache.get(link, "")
            confirmed, based_on = verify_infobox(link, book_title, author, wt)

            if not confirmed and "(film)" not in link.lower():
                alt_link = f"{link} (film)"
                alt_wt   = wt_cache.get(alt_link)
                if alt_wt is None:
                    alt_wt = _single_wikitext(alt_link)
                confirmed, based_on = verify_infobox(alt_link, book_title, author, alt_wt)
                if confirmed:
                    link = alt_link
                    wt   = alt_wt or ""

            if confirmed:
                plot = (extract_plot_from_wikitext(wt, alert, link)
                        or extract_plot_via_api(link, alert))
                if plot:
                    log(f"         '{link}' ({len(plot)} chars)")
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
            log(f"         No verified film+plot found", also_print=False)

        alert.tick(time.time() - book_start, book_title)

    save_state(state)
    log(f"\nPass 2 complete. Found: {found_this_pass} new pairs.")
    alert.summary("Pass 2 alert summary")
    return state, alert

# ──────────────────────────────────────────────────────────────────────────────
# PASS 3 — AUTHOR WIKIPEDIA CATEGORY
# Speedup: batch wikitext for all category members at once
# ──────────────────────────────────────────────────────────────────────────────

def run_pass3(state):
    alert   = AlertTracker()
    pending = {k: v for k, v in get_pending(state, pass_number=3).items()
               if v.get("author")}
    log(f"\n{'='*60}")
    log(f"PASS 3 — Author category | {len(pending)} books pending (with author)")
    log(f"{'='*60}")

    found_this_pass      = 0
    author_category_cache = {}

    for i, (key, entry) in enumerate(pending.items()):
        book_title = entry["book_title"]
        author     = entry["author"]
        book_start = time.time()

        if i % 50 == 0 and i > 0:
            log(f"  Progress: {i}/{len(pending)} | found: {found_this_pass}")
            save_state(state)

        log(f"  [{i+1:4d}/{len(pending)}] '{book_title}' by {author}", also_print=True)
        state[key]["pass_tried"] = 3

        parts        = author.strip().split()
        short_author = f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else author
        category     = f"Films based on works by {short_author}"

        if category not in author_category_cache:
            data = api_get({
                "action":  "query",
                "list":    "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": 200,
                "cmtype":  "page",
            })
            time.sleep(RATE_LIMIT)
            members = ([m["title"] for m in
                        data.get("query",{}).get("categorymembers",[])]
                       if data else [])
            author_category_cache[category] = members
            log(f"         Category '{category}': {len(members)} films", also_print=False)
        else:
            members = author_category_cache[category]

        if not members:
            log(f"         Category empty or not found", also_print=False)
            alert.tick(time.time() - book_start, book_title)
            continue

        matches = sorted(
            [(m, fuzz.token_set_ratio(book_title.lower(), m.lower()))
             for m in members],
            key=lambda x: -x[1]
        )
        top_matches = [(m, s) for m, s in matches if s >= 50][:3]

        if not top_matches:
            log(f"         No title match ≥ 50 in author category", also_print=False)
            alert.tick(time.time() - book_start, book_title)
            continue

        # Batch fetch wikitext for top matches
        top_pages = [m for m, _ in top_matches]
        wt_cache  = batch_wiki_wikitext(top_pages, alert)

        found = False
        for film_page, sim in top_matches:
            wt        = wt_cache.get(film_page, "")
            confirmed, based_on = verify_infobox(film_page, book_title, author, wt)
            if confirmed:
                plot = (extract_plot_from_wikitext(wt, alert, film_page)
                        or extract_plot_via_api(film_page, alert))
                if plot:
                    log(f"         '{film_page}' sim={sim} ({len(plot)} chars)")
                    state[key].update({
                        "status":      "found",
                        "film_page":   film_page,
                        "strategy":    "pass3_author_category",
                        "based_on":    based_on,
                        "movie_plot":  plot,
                        "plot_length": len(plot),
                    })
                    found_this_pass += 1
                    found = True
                    break

        if not found:
            log(f"         No verified film+plot in category", also_print=False)

        alert.tick(time.time() - book_start, book_title)

    save_state(state)
    log(f"\nPass 3 complete. Found: {found_this_pass} new pairs.")
    alert.summary("Pass 3 alert summary")
    return state, alert

# ──────────────────────────────────────────────────────────────────────────────
# PASS 4 — FUZZY TITLE + MANDATORY INFOBOX (last resort)
# ──────────────────────────────────────────────────────────────────────────────

def run_pass4(state):
    alert   = AlertTracker()
    pending = get_pending(state, pass_number=4)
    log(f"\n{'='*60}")
    log(f"PASS 4 — Fuzzy+infobox | {len(pending)} books pending")
    log(f"{'='*60}")

    found_this_pass = 0

    for i, (key, entry) in enumerate(pending.items()):
        book_title = entry["book_title"]
        author     = entry.get("author")
        book_start = time.time()

        if i % 50 == 0 and i > 0:
            log(f"  Progress: {i}/{len(pending)} | found: {found_this_pass}")
            save_state(state)

        log(f"  [{i+1:4d}/{len(pending)}] '{book_title}'", also_print=True)
        state[key]["pass_tried"] = 4

        found = False
        for query in [f"{book_title} film", f"{book_title} (film)"]:
            results = wiki_search(query, n=5)
            time.sleep(RATE_LIMIT)

            candidates = [r for r in results
                          if fuzz.token_set_ratio(
                              book_title.lower(), r.lower()
                          ) >= FILM_PAGE_FUZZY_THRESHOLD]

            if not candidates:
                continue

            wt_cache = batch_wiki_wikitext(candidates, alert)

            for result in candidates:
                wt        = wt_cache.get(result, "")
                confirmed, based_on = verify_infobox(result, book_title, author, wt)
                if confirmed:
                    plot = (extract_plot_from_wikitext(wt, alert, result)
                            or extract_plot_via_api(result, alert))
                    if plot:
                        log(f" '{result}' ({len(plot)} chars)")
                        state[key].update({
                            "status":      "found",
                            "film_page":   result,
                            "strategy":    "pass4_fuzzy_verified",
                            "based_on":    based_on,
                            "movie_plot":  plot,
                            "plot_length": len(plot),
                        })
                        found_this_pass += 1
                        found = True
                        break
            if found:
                break

        if not found:
            state[key]["status"] = "exhausted"

        alert.tick(time.time() - book_start, book_title)

    save_state(state)
    log(f"\nPass 4 complete. Found: {found_this_pass} new pairs.")
    alert.summary("Pass 4 alert summary")
    return state, alert

# ──────────────────────────────────────────────────────────────────────────────
# RETRY SPARQL FAILURES
# ──────────────────────────────────────────────────────────────────────────────

def retry_sparql_failures(state):
    failures = get_sparql_failures(state)
    if not failures:
        log("No SPARQL failures found — nothing to retry.")
        return state
    log(f"\nRetrying {len(failures)} SPARQL-failed books from Pass 1...")
    for k in failures:
        state[k]["pass_tried"]    = 0
        state[k]["sparql_failed"] = False
    state, alert = run_pass1(state)
    print_state_summary(state, "After SPARQL retry")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# BUILD FINAL DATASET
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(state):
    log(f"\n{'='*60}\nBUILDING FINAL DATASET\n{'='*60}")
    records = [
        {
            "book_title":          v["book_title"],
            "movie_title":         v["film_page"]
                                     .replace(" (film)","")
                                     .replace(" (movie)",""),
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
    log(f"Length ratio  — {ratio:.1f}x")
    log("\nVerification breakdown:")
    for method, cnt in df["verification_method"].value_counts().items():
        log(f"  {method:<40} {cnt:4d} ({cnt/len(df)*100:.1f}%)")
    print("\nTop 10 by plot length:")
    print(df[["book_title","movie_title","verification_method","plot_length_chars"]]
          .sort_values("plot_length_chars", ascending=False)
          .head(10).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# INITIALISE STATE
# ──────────────────────────────────────────────────────────────────────────────

def initialise_state():
    os.makedirs(DATA_DIR, exist_ok=True)
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
            "book_title":    book_title,
            "author":        (str(row[author_col]).strip()
                              if author_col and pd.notna(row.get(author_col))
                              else None),
            "book_summary":  str(row[summary_col]),
            "book_genres":   (str(row[genre_col])
                              if genre_col and pd.notna(row.get(genre_col))
                              else "[]"),
            "status":        "pending",
            "pass_tried":    0,
            "sparql_failed": False,
            "_candidates":   None,   # populated in Pass 1 Phase A
            "_search_query": None,
            "film_page":     None,
            "strategy":      None,
            "based_on":      None,
            "movie_plot":    None,
            "plot_length":   None,
            "_pri":          int(row["_pri"]),
        }

    save_state(state)
    n_pri = sum(1 for v in state.values() if v.get("_pri",0))
    log(f"Initialised state: {len(state)} books ({n_pri} priority)")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Multi-pass Wikipedia book→movie scraper V4")
    parser.add_argument("--pass",         type=int, choices=[1,2,3,4],
                        dest="run_pass",  help="Which pass to run")
    parser.add_argument("--init",         action="store_true",
                        help="Initialise state from cmu_cleaned.csv")
    parser.add_argument("--build",        action="store_true",
                        help="Build final CSV from current state")
    parser.add_argument("--status",       action="store_true",
                        help="Print state summary and exit")
    parser.add_argument("--retry-sparql", action="store_true",
                        help="Retry books that failed due to SPARQL errors in Pass 1")
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

    elif args.retry_sparql:
        state = load_state()
        state = retry_sparql_failures(state)

    elif args.run_pass:
        state = load_state()
        if not state:
            print("State is empty. Run --init first.")
            exit(1)

        print_state_summary(state, f"Before Pass {args.run_pass}")

        if   args.run_pass == 1: state, alert = run_pass1(state)
        elif args.run_pass == 2: state, alert = run_pass2(state)
        elif args.run_pass == 3: state, alert = run_pass3(state)
        elif args.run_pass == 4: state, alert = run_pass4(state)

        print_state_summary(state, f"After Pass {args.run_pass}")

        found   = sum(1 for v in state.values() if v["status"] == "found")
        pending = sum(1 for v in state.values() if v["status"] == "pending")
        sfail   = sum(1 for v in state.values() if v.get("sparql_failed"))

        print(f"\n{'='*60}")
        print(f"  DECISION POINT")
        print(f"{'='*60}")
        print(f"  Pairs found:        {found}")
        print(f"  Books pending:      {pending}")
        print(f"  SPARQL failures:    {sfail}  ← run --retry-sparql if > 0")
        print(f"\n  Recommendation:")
        if found >= 1000:
            print(f"   {found} pairs — sufficient. Run --build or continue.")
        elif found >= 500:
            print(f"    {found} pairs — workable. Consider next pass.")
        else:
            print(f"   {found} pairs — run next pass.")
        if args.run_pass < 4:
            print(f"\n  Next:  python {os.path.basename(__file__)} "
                  f"--pass {args.run_pass+1}")
        if sfail:
            print(f"  Retry: python {os.path.basename(__file__)} --retry-sparql")
        print(f"  Build: python {os.path.basename(__file__)} --build")
        print(f"{'='*60}")

    else:
        parser.print_help()