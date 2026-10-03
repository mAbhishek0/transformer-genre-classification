# =============================================================================
# TOPIC PROBABILITY MATRIX GENERATION — VERSION 2
#
# Fixes vs V1:
#   [FIX 1] sigma calibration: percentile-based (15th pct) instead of median.
#           Median sigma was too wide → flat RBF → every doc gets identical
#           entropy = log(TOP_K). Confirmed by T4 std=0.000 in V1 validation.
#   [FIX 2] sigma sweep over [0.10× … 0.50×] auto-sigma, pick by target
#           top-1 mass in [0.40, 0.60]. Replaces the blind median heuristic.
#   [FIX 3] TOP_K reduced 5 → 3. Top-3 is sufficient for multi-genre blending
#           and directly raises primary-topic mass without other changes.
#   [FIX 4] Centroid coherence report added (mean intra-cluster cosine sim).
#           Low coherence → topics are noisy → warns before wasting training.
# =============================================================================


# TOPIC PROBABILITY MATRIX GENERATION — VERSION 2
import os, pickle, warnings
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

# CONFIG
DATA_DIR     = "./data/"
K_RANGE      = [30, 40, 50, 60, 75]
TOP_K        = 3                      # FIX 3: reduced from 5
RANDOM_STATE = 42

# Sigma sweep: fractions of auto-sigma (15th-pct distance) to try
# Target: mean top-1 mass in [TARGET_MASS_LO, TARGET_MASS_HI]
SIGMA_FRACTIONS    = [0.10, 0.20, 0.30, 0.40, 0.50]
TARGET_MASS_LO     = 0.40
TARGET_MASS_HI     = 0.60

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — LOADING DATA")
print("=" * 70)

with open(f"{DATA_DIR}data_splits.pkl", "rb") as f:
    splits = pickle.load(f)
with open(f"{DATA_DIR}mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

X_train      = list(splits["X_train"])
X_val        = list(splits["X_val"])
X_test       = list(splits["X_test"])
y_train      = splits["y_train"]
all_texts    = X_train + X_val + X_test
TARGET_GENRES = list(mlb.classes_)

embeddings       = np.load(f"{DATA_DIR}all_embeddings.npy").astype(np.float32)
embeddings_normed = normalize(embeddings, norm="l2")

n_train = len(X_train); n_val = len(X_val); n_test = len(X_test)
n_total = len(all_texts)
assert n_total == len(embeddings)
print(f"Loaded: {n_train} train | {n_val} val | {n_test} test")
print(f"Embedding shape: {embeddings.shape}")

# 2. SIGMA BASE — 15th PERCENTILE (FIX 1)
print("\n" + "=" * 70)
print("STEP 2 — SIGMA BASE (15th-percentile pairwise distance)")
print("=" * 70)

rng        = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(n_total, size=min(2000, n_total), replace=False)
sample_emb = embeddings_normed[sample_idx]

sq_dists  = cdist(sample_emb, sample_emb, metric="sqeuclidean")
upper_tri = sq_dists[np.triu_indices_from(sq_dists, k=1)]

sigma_base_sq = np.percentile(upper_tri, 15)   # FIX 1: 15th pct, not median
sigma_base    = np.sqrt(sigma_base_sq)

print(f"15th-pct pairwise squared distance: {sigma_base_sq:.4f}")
print(f"sigma_base (15th-pct L2):           {sigma_base:.4f}")
print(f"V1 median sigma was:                1.3064  (this should be much smaller)")

# ──────────────────────────────────────────────────────────────────────────────
# 3. K SWEEP
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — K SWEEP (silhouette + genre alignment)")
print("=" * 70)

def genre_alignment(train_topics, y_train, K, TARGET_GENRES):
    scores = []
    for g in range(len(TARGET_GENRES)):
        mask = y_train[:, g].astype(bool)
        if mask.sum() < 10:
            continue
        counts = np.bincount(train_topics[mask], minlength=K).astype(float)
        counts /= counts.sum()
        H = -np.sum(counts[counts > 0] * np.log(counts[counts > 0]))
        scores.append(H / np.log(K))
    return 1.0 - np.mean(scores)

sweep_results = []
for K in K_RANGE:
    print(f"  Testing K={K}...", end="", flush=True)
    km = MiniBatchKMeans(K, random_state=RANDOM_STATE, batch_size=4096,
                         n_init=10, max_iter=300)
    km.fit(embeddings_normed)
    train_topics = km.labels_[:n_train]

    sil_idx = rng.choice(n_total, size=min(5000, n_total), replace=False)
    sil = silhouette_score(embeddings_normed[sil_idx],
                           km.predict(embeddings_normed[sil_idx]),
                           metric="euclidean")
    align    = genre_alignment(train_topics, y_train, K, TARGET_GENRES)
    composite = 0.6 * sil + 0.4 * align
    sweep_results.append({"K": K, "sil": sil, "align": align, "composite": composite})
    print(f"  Sil={sil:.4f}  Align={align:.4f}  Composite={composite:.4f}")

best_K_entry = max(sweep_results, key=lambda x: x["composite"])
K_FINAL      = best_K_entry["K"]
print(f"\n  → Best K={K_FINAL}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. FINAL K-MEANS FIT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"STEP 4 — FINAL K-MEANS FIT (K={K_FINAL})")
print("=" * 70)

km_final = MiniBatchKMeans(K_FINAL, random_state=RANDOM_STATE, batch_size=4096,
                            n_init=20, max_iter=500)
km_final.fit(embeddings_normed)
centroids_normed = normalize(km_final.cluster_centers_, norm="l2")
print(f"Centroids shape: {centroids_normed.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4b. CENTROID COHERENCE REPORT (FIX 4)
#
# Mean cosine similarity of docs within each topic to their centroid.
# < 0.5 → topic is a loose bag of unrelated docs (K-Means forcing structure
#           where none exists) — topic features will carry noise, not signal.
# > 0.7 → tight, coherent cluster — high-quality thematic signal expected.
# ──────────────────────────────────────────────────────────────────────────────
print("\nCentroid Coherence (mean intra-cluster cosine similarity):")
train_assignments = km_final.predict(embeddings_normed[:n_train])
coherences = []
for k in range(K_FINAL):
    mask = train_assignments == k
    if mask.sum() == 0:
        continue
    cluster_embs = embeddings_normed[:n_train][mask]
    centroid     = centroids_normed[k]
    cos_sims     = cluster_embs @ centroid        # unit vectors → dot = cosine
    coherences.append(cos_sims.mean())

mean_coh = np.mean(coherences)
min_coh  = np.min(coherences)
print(f"  Mean coherence across {K_FINAL} topics: {mean_coh:.4f}")
print(f"  Min coherence (worst topic):           {min_coh:.4f}")
if mean_coh < 0.50:
    print("  LOW coherence — topics are geometrically loose. "
          "Topic features may be noisy. Consider sentence-debias or "
          "UMAP reduction before clustering.")
elif mean_coh > 0.70:
    print("   HIGH coherence — topics are tight and semantically focused.")
else:
    print("   Moderate coherence — acceptable for soft-fusion features.")

# ──────────────────────────────────────────────────────────────────────────────
# 5. SIGMA SWEEP — FIND SHARPNESS TARGET (FIX 2)
#
# For each fraction f of sigma_base, compute RBF probs on a sample,
# measure mean top-1 mass. Pick f where top-1 mass ∈ [TARGET_LO, TARGET_HI].
# This replaces the blind median heuristic with a closed-loop calibration.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — SIGMA SWEEP (target top-1 mass in "
      f"[{TARGET_MASS_LO}, {TARGET_MASS_HI}])")
print("=" * 70)

def rbf_probs_sample(emb, centroids, sigma_sq, top_k):
    dots   = emb @ centroids.T
    sq_d   = 2.0 - 2.0 * dots
    rbf    = np.exp(-sq_d / (2.0 * sigma_sq))
    if top_k < centroids.shape[0]:
        thresh = np.sort(rbf, axis=1)[:, -top_k]
        rbf    = rbf * (rbf >= thresh[:, None])
    rs  = rbf.sum(axis=1, keepdims=True)
    rs  = np.where(rs == 0, 1.0, rs)
    return (rbf / rs).astype(np.float32)

# Use a 3000-doc sample for speed
sigma_sweep_idx  = rng.choice(n_total, size=min(3000, n_total), replace=False)
sigma_sweep_emb  = embeddings_normed[sigma_sweep_idx]

sigma_results = []
best_sigma_sq = None
print(f"  {'Fraction':<10} {'sigma_sq':<12} {'mean top-1 mass':<18} {'mean H':<12} {'verdict'}")
print(f"  {'-'*8:<10} {'-'*10:<12} {'-'*15:<18} {'-'*8:<12} {'-'*12}")

for frac in SIGMA_FRACTIONS:
    sq  = sigma_base_sq * (frac ** 2)
    p   = rbf_probs_sample(sigma_sweep_emb, centroids_normed, sq, TOP_K)
    t1  = p.max(axis=1).mean()
    H   = -np.sum(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1)), axis=1).mean()
    std_H = (-np.sum(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1)), axis=1)).std()
    ok  = TARGET_MASS_LO <= t1 <= TARGET_MASS_HI
    verdict = " TARGET" if ok else ("too flat" if t1 < TARGET_MASS_LO else "too sharp")
    print(f"  {frac:<10.2f} {sq:<12.4f} {t1:<18.4f} {H:<12.4f} {verdict}  std(H)={std_H:.4f}")
    sigma_results.append({"frac": frac, "sq": sq, "top1": t1, "H": H, "ok": ok})
    if ok and best_sigma_sq is None:
        best_sigma_sq = sq

# Fallback: pick closest to target midpoint
if best_sigma_sq is None:
    target_mid = (TARGET_MASS_LO + TARGET_MASS_HI) / 2
    best_sigma_sq = min(sigma_results, key=lambda x: abs(x["top1"] - target_mid))["sq"]
    print(f"\n  No exact match → using closest: sigma_sq={best_sigma_sq:.4f}")
else:
    print(f"\n  → Selected sigma_sq={best_sigma_sq:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. FULL PROBABILITY COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — FULL PROBABILITY COMPUTATION")
print("=" * 70)

BATCH = 2000
all_probs = []
for i in range(int(np.ceil(n_total / BATCH))):
    s, e  = i * BATCH, min((i + 1) * BATCH, n_total)
    p     = rbf_probs_sample(embeddings_normed[s:e], centroids_normed,
                              best_sigma_sq, TOP_K)
    all_probs.append(p)
    if (i + 1) % 10 == 0 or e == n_total:
        print(f"  {e}/{n_total} docs processed...")

all_probs = np.vstack(all_probs)
print(f"\nFull matrix shape:   {all_probs.shape}")
print(f"Row sum range:       [{all_probs.sum(1).min():.6f}, {all_probs.sum(1).max():.6f}]")
print(f"Global top-1 mass:   {all_probs.max(axis=1).mean():.4f}")
print(f"Global mean entropy: "
      f"{(-np.sum(np.clip(all_probs,1e-12,1)*np.log(np.clip(all_probs,1e-12,1)),axis=1)).mean():.4f}")
print(f"Entropy std:         "
      f"{(-np.sum(np.clip(all_probs,1e-12,1)*np.log(np.clip(all_probs,1e-12,1)),axis=1)).std():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. SLICE & SAVE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7 — SLICING AND SAVING")
print("=" * 70)

probs_train = all_probs[:n_train]
probs_val   = all_probs[n_train : n_train + n_val]
probs_test  = all_probs[n_train + n_val:]

np.save(f"{DATA_DIR}topic_probs_train.npy", probs_train)
np.save(f"{DATA_DIR}topic_probs_val.npy",   probs_val)
np.save(f"{DATA_DIR}topic_probs_test.npy",  probs_test)

metadata = {
    "K": K_FINAL, "sigma_sq": float(best_sigma_sq),
    "sigma_base": float(sigma_base), "sigma_base_sq": float(sigma_base_sq),
    "top_k": TOP_K, "sweep_results": sweep_results,
    "sigma_sweep_results": sigma_results,
    "n_train": n_train, "n_val": n_val, "n_test": n_test,
    "centroid_coherences": coherences,
}
with open(f"{DATA_DIR}topic_model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Saved train:{probs_train.shape}  val:{probs_val.shape}  test:{probs_test.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. TOPIC COHERENCE REPORT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8 — TOPIC COHERENCE REPORT")
print("=" * 70)

vectorizer = TfidfVectorizer(stop_words="english", max_features=20000,
                             ngram_range=(1, 2), min_df=3)
tfidf_matrix = vectorizer.fit_transform(X_train)
vocab        = np.array(vectorizer.get_feature_names_out())

lines = []
for k in range(K_FINAL):
    mask  = train_assignments == k
    n_doc = mask.sum()
    if n_doc == 0:
        lines.append(f"Topic {k:02d}: EMPTY  coh=N/A"); continue
    tf    = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
    terms = ", ".join(vocab[tf.argsort()[-10:][::-1]])
    coh   = coherences[k] if k < len(coherences) else float("nan")
    lines.append(f"Topic {k:02d} (n={n_doc:4d}, coh={coh:.3f}): {terms}")

for l in lines:
    print(" ", l)

with open(f"{DATA_DIR}topic_words_report_v2.txt", "w") as f:
    f.write(f"K={K_FINAL} | sigma_sq={best_sigma_sq:.4f} | top_k={TOP_K} | "
            f"mean_coh={mean_coh:.4f}\n\n" + "\n".join(lines))

print("\n V2 generation complete. Run validate_topic_probs_v2.py next.")