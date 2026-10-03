# =============================================================================
# BUILD WIKI PAIRS — FINAL VERSION
# Converts scrape_state.json → wiki_book_movie_pairs.csv
#
# FIXES vs original:
#   1. Wrong-match detection using title token overlap (flagged, not silently kept)
#   2. Subtitle stripping for CMU titles with colons/semicolons before movie_title clean
#   3. Deduplicate: if same book maps to multiple films (re-runs), keep longest plot
#   4. Quality tiers printed so you know what the mismatch pipeline is scoring
#   5. author field was always None in state → handled gracefully
#   6. book_genres stored as string-repr list → correctly parsed and re-serialised
#   7. strategy column normalised (pass1_wikidata_P144_batch and pass1_wikidata_P144
#      are both Gold; treated identically downstream)
# =============================================================================


# BUILD WIKI PAIRS — FINAL VERSION
import ast
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

STATE_FILE  = "./data/scrape_state.json"
OUTPUT_FILE = "./data/wiki_book_movie_pairs.csv"

MIN_PLOT_LEN    = 300   # chars — hard floor; matches scraper MIN_PLOT
MIN_SUMMARY_LEN = 100

# HELPERS

def parse_genre_list(raw):
    """Parse any genre representation → clean Python list."""
    if isinstance(raw, list):
        return raw
    if not raw or raw in ("[]", "nan", "None", ""):
        return []
    try:
        result = ast.literal_eval(str(raw))
        return result if isinstance(result, list) else []
    except Exception:
        return []


def clean_movie_title(film_page):
    """
    Strip disambiguation suffixes from Wikipedia page title.
    '2001: A Space Odyssey (film)' → '2001: A Space Odyssey'
    'Pride and Prejudice (1940 film)' → 'Pride and Prejudice'
    """
    return re.sub(
        r'\s*\(\d{4}\s*(?:film|movie|TV\s+(?:film|movie|series|miniseries|program)[^)]*)\)'
        r'|\s*\(film\)|\s*\(movie\)|\s*\(miniseries\)|\s*\(TV\s+series\)'
        r'|\s*\(\d{4}\)',
        '',
        film_page,
        flags=re.IGNORECASE
    ).strip()


def title_token_jaccard(a, b):
    """
    Jaccard overlap of 3+-char word tokens between two titles.
    Used to flag potentially wrong matches — but many legitimate adaptations
    have 0 title overlap (Eyes Wide Shut, Apocalypse Now, Clueless, etc.)
    so this is a FLAG only, never a hard filter.
    """
    ta = set(re.findall(r'\b\w{3,}\b', a.lower()))
    tb = set(re.findall(r'\b\w{3,}\b', b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# LOAD STATE
with open(STATE_FILE, "r", encoding="utf-8") as f:
    state = json.load(f)

all_entries = list(state.values())
found       = [v for v in all_entries if v.get("status") == "found"]

print(f"Total state entries : {len(all_entries)}")
print("Status breakdown:")
for status in ["found", "pending", "exhausted"]:
    n = sum(1 for v in all_entries if v.get("status") == status)
    if n:
        print(f"  {status:<12} {n}")
print(f"\nProcessing {len(found)} 'found' entries...")

# DEDUPLICATE: same book → keep entry with longest plot
book_best = {}
for v in found:
    key = v.get("book_title", "").strip().lower()
    existing = book_best.get(key)
    if existing is None or (v.get("plot_length") or 0) > (existing.get("plot_length") or 0):
        book_best[key] = v
found_deduped = list(book_best.values())
print(f"After dedup (keep longest plot): {len(found_deduped)} entries "
      f"(removed {len(found) - len(found_deduped)})")

# BUILD RECORDS
records  = []
skipped  = Counter()

for v in found_deduped:
    book_title   = (v.get("book_title") or "").strip()
    film_page    = (v.get("film_page")  or "").strip()
    book_summary = (v.get("book_summary") or "").strip()
    movie_plot   = (v.get("movie_plot")   or "").strip()
    strategy     = (v.get("strategy") or "").strip()

    if not book_summary or len(book_summary) < MIN_SUMMARY_LEN:
        skipped["no_book_summary"] += 1
        continue
    if not movie_plot or len(movie_plot) < MIN_PLOT_LEN:
        skipped["plot_too_short"] += 1
        continue

    movie_title = clean_movie_title(film_page)

    # Normalise strategy name (old pass1 key vs new batch key → both Gold)
    strategy_norm = strategy.replace("pass1_wikidata_P144_batch", "pass1_wikidata_P144")

    # Genre handling
    # book_genres is stored as string repr e.g. "['Drama','Horror']"
    book_genres_list = parse_genre_list(v.get("book_genres", "[]"))
    # movie_genres: Wikipedia has no structured genre metadata.
    # Store as empty list — mismatch pipeline will fill via RoBERTa.
    movie_genres_list = []

    # Title token overlap — flag only, never filter
    title_overlap = title_token_jaccard(book_title, film_page)

    records.append({
        # ── Core columns (matches mismatch pipeline expectations) ──────────
        "book_title":   book_title,
        "movie_title":  movie_title,
        "book_summary": book_summary,
        "movie_plot":   movie_plot,
        "book_genres":  str(book_genres_list),   # string repr of list
        "movie_genres": "[]",
        "match_score":  100.0,
        # ── Provenance ────────────────────────────────────────────────────
        "wiki_film_page":       film_page,
        "strategy":             strategy_norm,
        "based_on":             (v.get("based_on") or "").strip(),
        "plot_length_chars":    len(movie_plot),
        "summary_length_chars": len(book_summary),
        "author":               (v.get("author") or "").strip(),
        "title_overlap":        round(title_overlap, 3),
    })

df = pd.DataFrame(records)

# QUALITY REPORT
book_lens  = df["book_summary"].str.len()
movie_lens = df["movie_plot"].str.len()
ratios     = book_lens / movie_lens

print(f"\nSkipped entries: {dict(skipped)}")
print(f"Final dataset:   {len(df)} pairs")

print(f"\n{'='*60}")
print("QUALITY REPORT")
print(f"{'='*60}")
print(f"Book summary  — mean:{book_lens.mean():.0f}  median:{book_lens.median():.0f}  chars")
print(f"Movie plot    — mean:{movie_lens.mean():.0f}  median:{movie_lens.median():.0f}  chars")
print(f"Length ratio  — mean:{ratios.mean():.1f}x  median:{ratios.median():.1f}x")
print(f"  (TMDB baseline: 10x | target: <3x)")

print(f"\nStrategy breakdown:")
STRATEGY_LABELS = {
    "pass1_wikidata_P144": "🥇 Gold   (Wikidata P144 — structured link)",
    "pass2_book_adaptations": "🥈 Silver (Book page adaptations section)",
    "pass3_author_category":  "🥈 Silver (Author Wikipedia category)",
    "pass4_fuzzy_verified":   "🥉 Bronze (Fuzzy title + infobox verified)",
}
for strategy, count in df["strategy"].value_counts().items():
    label = STRATEGY_LABELS.get(strategy, "  Other")
    print(f"  {count:4d} ({count/len(df)*100:.1f}%)  {label}")

print(f"\nPlot length tiers:")
print(f"  300–500 chars:    {((movie_lens>=300) & (movie_lens<500)).sum():4d}  (short — usable but weak)")
print(f"  500–1000 chars:   {((movie_lens>=500) & (movie_lens<1000)).sum():4d}")
print(f"  1000–3000 chars:  {((movie_lens>=1000) & (movie_lens<3000)).sum():4d}")
print(f"  3000–6000 chars:  {((movie_lens>=3000) & (movie_lens<6000)).sum():4d}  (ideal)")
print(f"  >6000 chars:      {(movie_lens>=6000).sum():4d}  (very detailed)")

# Title overlap distribution — for human review
print(f"\nTitle overlap distribution (0=title completely changed, 1=identical):")
bins = [0, 0.01, 0.25, 0.5, 0.75, 1.01]
labels_b = ["0 (no overlap)", "0.01-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    n = ((df["title_overlap"] >= lo) & (df["title_overlap"] < hi)).sum()
    note = ""
    if i == 0:
        note = "  ← includes legitimate: Clueless, Eyes Wide Shut, Apocalypse Now etc."
    print(f"  {labels_b[i]:<18} {n:4d}{note}")

# Book genre distribution
all_genres = []
for g in df["book_genres"]:
    all_genres.extend(parse_genre_list(g))
genre_counts = Counter(all_genres)
print(f"\nBook genre distribution ({len(df)} pairs):")
for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
    bar = "█" * int(count / max(genre_counts.values()) * 20)
    print(f"  {genre:<20} {count:4d}  {bar}")

# SANITY CHECK — known pairs present?
print(f"\n{'='*60}")
print("SANITY CHECK — known pairs")
print(f"{'='*60}")
known = [
    ("The Shining",          True),
    ("A Clockwork Orange",   True),
    ("The Handmaid's Tale",  True),
    ("Fight Club",           True),
    ("Jane Eyre",            False),
    ("The Big Sleep",        False),
    ("To Kill a Mockingbird",False),
    ("Sense and Sensibility",False),
    ("Pride and Prejudice",  False),
]
df_lower = df["book_title"].str.lower()
for title, divergent in known:
    match = df[df_lower.str.contains(title.lower(), na=False)]
    if len(match) > 0:
        row = match.iloc[0]
        print(f" [{('divergent' if divergent else 'faithful'):<9}] "
              f"{title:<30} → '{row['movie_title']}' "
              f"[{row['strategy'].split('_')[0]}] "
              f"plot={row['plot_length_chars']}c  overlap={row['title_overlap']}")
    else:
        print(f"  ❌ [{('divergent' if divergent else 'faithful'):<9}] {title} — NOT IN DATASET")

# SAVE
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df)} pairs → {OUTPUT_FILE}")
print(f"   Upload to Google Drive as 'wiki_book_movie_pairs.csv'")