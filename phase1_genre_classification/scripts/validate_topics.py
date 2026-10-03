# =============================================================================
# TOPIC PROBABILITY MATRIX — VALIDATION SUITE V2
#
# New vs V1:
#   T4b: std(entropy) < 0.10 is now a FAIL — catches the V1 degenerate case
#        where every doc had identical entropy = log(TOP_K) (std=0.000).
#        V1's T4 PASSED this despite the distributions being completely flat
#        and meaningless. This test would have caught it.
#   T5 : top-1 mass threshold raised to 0.40 (was 0.30) — reflects TOP_K=3.
#   T11: Centroid coherence — loaded from metadata, flagged if mean < 0.50.
#   T12: Per-doc distribution sharpness variance — are some docs genuinely
#        more topic-concentrated than others? Low variance → uniform → bad.
# =============================================================================


# TOPIC PROBABILITY MATRIX — VALIDATION SUITE V2

import pickle, warnings
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
warnings.filterwarnings("ignore")

DATA_DIR = "./data/"
PASS = "PASS"; WARN = "WARN"; FAIL = "FAIL"

# Thresholds (V2 values — updated from V1)
TOL_NORM          = 1e-5
COLLAPSE_THRESH   = 0.40
DEAD_THRESH       = 1e-6
ENTROPY_LOW_WARN  = 0.5
ENTROPY_HIGH_WARN = 2.5     # tighter: TOP_K=3, log(3)=1.099 is max theoretical
STD_ENTROPY_MIN   = 0.05    # NEW: std(H) < 0.05 → degenerate uniform (V1 bug)
TOP1_MASS_MIN     = 0.40    # raised from 0.30 (TOP_K=3 should produce sharper)
TOP3_MASS_MIN     = 0.90    # top-3 should capture ≥90% with TOP_K=3
KL_SHIFT_MAX      = 0.10
MI_MIN            = 0.005
COH_MIN           = 0.50    # min acceptable mean centroid coherence
SHARPNESS_VAR_MIN = 0.001   # min variance of top-1 mass across docs

results = []
def record(tid, name, status, detail):
    results.append((tid, name, status, detail))
    print(f"  {status}  [{tid}] {name}\n         {detail}\n")

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING FILES")
print("=" * 70)

probs_train = np.load(f"{DATA_DIR}topic_probs_train.npy")
probs_val   = np.load(f"{DATA_DIR}topic_probs_val.npy")
probs_test  = np.load(f"{DATA_DIR}topic_probs_test.npy")

with open(f"{DATA_DIR}data_splits.pkl", "rb") as f:
    splits = pickle.load(f)
y_train = splits["y_train"]

with open(f"{DATA_DIR}mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
TARGET_GENRES = list(mlb.classes_)

with open(f"{DATA_DIR}topic_model_metadata.pkl", "rb") as f:
    meta = pickle.load(f)

K = meta["K"]
print(f"K={K} | sigma_sq={meta['sigma_sq']:.4f} | top_k={meta['top_k']}")
print(f"Shapes — train:{probs_train.shape}  val:{probs_val.shape}  test:{probs_test.shape}\n")
print("=" * 70)
print("RUNNING VALIDATION SUITE V2")
print("=" * 70 + "\n")

all_splits = {"train": probs_train, "val": probs_val, "test": probs_test}

# T1 — Row normalization
for name, P in all_splits.items():
    dev = np.abs(P.sum(axis=1) - 1.0)
    n_bad = (dev > TOL_NORM).sum()
    record("T1", f"Row normalization [{name}]",
           PASS if n_bad == 0 else FAIL,
           f"{'All' if n_bad==0 else n_bad} rows | max dev: {dev.max():.2e}")

# T2 — Topic collapse
for name, P in all_splits.items():
    dom = P.argmax(axis=1)
    counts = np.bincount(dom, minlength=K)
    pct = counts.max() / len(P)
    status = FAIL if pct > COLLAPSE_THRESH else (WARN if pct > 0.25 else PASS)
    record("T2", f"Topic collapse [{name}]", status,
           f"Largest topic {counts.argmax()}: {pct*100:.1f}%")

# T3 — Dead features
for name, P in all_splits.items():
    max_per = P.max(axis=0)
    dead = (max_per < DEAD_THRESH).sum()
    record("T3", f"Dead features [{name}]",
           FAIL if dead > 0 else PASS,
           f"{dead} dead | min topic-max: {max_per.min():.4f} | mean: {max_per.mean():.4f}")

# T4 — Entropy + T4b: std(entropy) degenerate check (NEW CRITICAL TEST)
for name, P in all_splits.items():
    Ps  = np.clip(P, 1e-12, 1.0)
    H   = -np.sum(Ps * np.log(Ps), axis=1)
    mH  = H.mean(); sH = H.std()
    max_H = np.log(K)

    if sH < STD_ENTROPY_MIN:
        # THIS IS THE V1 FAILURE: std=0 means every doc has identical distribution.
        # The RBF kernel was so flat that TOP_K sparsity made all docs uniform
        # over top-K — zero per-document information.
        record("T4", f"Entropy (mean+std) [{name}]", FAIL,
               f"std(H)={sH:.4f} < {STD_ENTROPY_MIN} — DEGENERATE: all docs have "
               f"identical distribution (=log(TOP_K)). sigma is too large. "
               f"mean H={mH:.3f}, max possible={max_H:.3f}")
    elif mH < ENTROPY_LOW_WARN:
        record("T4", f"Entropy (mean+std) [{name}]", WARN,
               f"mean H={mH:.3f} ± {sH:.3f} — too sharp. May underperform vs lookup.")
    elif mH > ENTROPY_HIGH_WARN:
        record("T4", f"Entropy (mean+std) [{name}]", WARN,
               f"mean H={mH:.3f} ± {sH:.3f} — too blurry. Increase sigma or TOP_K.")
    else:
        record("T4", f"Entropy (mean+std) [{name}]", PASS,
               f"mean H={mH:.3f} ± {sH:.3f} (max={max_H:.2f}, ratio={mH/max_H:.1%})")

# T5 — Top-K concentration mass (updated thresholds for TOP_K=3)
for name, P in all_splits.items():
    sP       = np.sort(P, axis=1)[:, ::-1]
    t1       = sP[:, 0].mean()
    t3       = sP[:, :3].sum(axis=1).mean()
    nonzero  = (P > 1e-8).sum(axis=1).mean()
    t1_var   = sP[:, 0].var()      # variance of top-1 mass across docs

    issues, status = [], PASS
    if t1 < TOP1_MASS_MIN:
        issues.append(f"top-1 mass {t1:.3f} < {TOP1_MASS_MIN}"); status = WARN
    if t3 < TOP3_MASS_MIN:
        issues.append(f"top-3 mass {t3:.3f} < {TOP3_MASS_MIN}"); status = FAIL
    if t1_var < SHARPNESS_VAR_MIN:
        issues.append(f"var(top-1)={t1_var:.5f} < {SHARPNESS_VAR_MIN} — all docs equally sharp"); status = WARN

    record("T5", f"Top-K concentration [{name}]", status,
           (", ".join(issues) + ". " if issues else "") +
           f"top-1={t1:.3f} | top-3={t3:.3f} | active={nonzero:.1f}/{K} | var(top-1)={t1_var:.5f}")

# T6 — Distribution shift
marg_train = probs_train.mean(axis=0)
for name, P in [("val", probs_val), ("test", probs_test)]:
    jsd = jensenshannon(marg_train, P.mean(axis=0), base=2)
    status = FAIL if jsd > KL_SHIFT_MAX else (WARN if jsd > KL_SHIFT_MAX/2 else PASS)
    record("T6", f"Distribution shift train vs {name}", status, f"JSD={jsd:.4f}")

# T7 — Genre discriminability
train_hard = probs_train.argmax(axis=1)
mi_scores  = {}
print("  Per-genre Mutual Information:")
for g, genre in enumerate(TARGET_GENRES):
    mi = mutual_info_score(train_hard, y_train[:, g].astype(int))
    mi_scores[genre] = mi
    bar  = "█" * int(mi / max(mi_scores.values() or [1]) * 20)
    flag = " ← LOW" if mi < MI_MIN else ""
    print(f"    {genre:<20} MI={mi:.5f}  {bar}{flag}")

low_mi   = [g for g, mi in mi_scores.items() if mi < MI_MIN]
mean_mi  = np.mean(list(mi_scores.values()))
status   = FAIL if mean_mi < MI_MIN else (WARN if low_mi else PASS)
record("T7", "Genre discriminability (MI)", status,
       f"Mean MI={mean_mi:.5f}. Low-MI genres: {low_mi if low_mi else 'None'}")

# T8 — Multi-label consistency
label_counts     = y_train.sum(axis=1)
Ps_train         = np.clip(probs_train, 1e-12, 1.0)
H_train          = -np.sum(Ps_train * np.log(Ps_train), axis=1)
sl_mask, ml_mask = label_counts == 1, label_counts >= 2
if sl_mask.sum() > 0 and ml_mask.sum() > 0:
    mH_sl = H_train[sl_mask].mean(); mH_ml = H_train[ml_mask].mean()
    diff  = mH_ml - mH_sl
    status = FAIL if diff < 0 else (WARN if diff < 0.05 else PASS)
    record("T8", "Multi-label consistency", status,
           f"Single-label H={mH_sl:.3f} | Multi-label H={mH_ml:.3f} | Δ={diff:.3f}")

# T9 — Numerical stability
for name, P in all_splits.items():
    issues = []
    if np.isnan(P).any(): issues.append("NaN")
    if np.isinf(P).any(): issues.append("Inf")
    if (P < 0).any():     issues.append(f"negatives (min={P.min():.2e})")
    record("T9", f"Numerical stability [{name}]",
           FAIL if issues else PASS,
           " | ".join(issues) if issues else f"Range: [{P.min():.2e}, {P.max():.4f}]")

# T10 — Scale compatibility
std_p = probs_train.std()
record("T10", "Fusion scale compatibility",
       WARN if std_p < 0.05 else PASS,
       f"Global std={std_p:.4f} | mean={probs_train.mean():.4f} | "
       f"mean top-1={probs_train.max(1).mean():.4f}")

# T11 — Centroid coherence (NEW)
if "centroid_coherences" in meta:
    cohs    = meta["centroid_coherences"]
    mean_c  = np.mean(cohs); min_c = np.min(cohs)
    n_low   = sum(1 for c in cohs if c < COH_MIN)
    status  = FAIL if mean_c < COH_MIN else (WARN if n_low > K // 4 else PASS)
    record("T11", "Centroid coherence", status,
           f"Mean={mean_c:.4f} | Min={min_c:.4f} | "
           f"Topics with coh<{COH_MIN}: {n_low}/{K}")
else:
    record("T11", "Centroid coherence", WARN,
           "Not found in metadata — re-run V2 generator to populate.")

# T12 — Per-doc sharpness variance (NEW)
# Does top-1 mass vary meaningfully across documents?
# Degenerate: all docs have identical top-1 mass → topic features are constant
# features, not per-doc features → zero discriminative power.
for name, P in all_splits.items():
    t1_per_doc = P.max(axis=1)
    var_t1     = t1_per_doc.var()
    min_t1     = t1_per_doc.min(); max_t1 = t1_per_doc.max()
    status = FAIL if var_t1 < SHARPNESS_VAR_MIN else PASS
    record("T12", f"Per-doc sharpness variance [{name}]", status,
           f"var(top-1 mass)={var_t1:.5f} | range=[{min_t1:.3f}, {max_t1:.3f}] | "
           f"(threshold: {SHARPNESS_VAR_MIN})")

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("VALIDATION SUMMARY V2")
print("=" * 70)

n_pass = sum(1 for r in results if PASS in r[2])
n_warn = sum(1 for r in results if WARN in r[2])
n_fail = sum(1 for r in results if FAIL in r[2])

for r in results:
    print(f"  {r[2]}  [{r[0]}] {r[1]}")

print(f"\nTotal: {len(results)} checks — {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL")

if n_fail > 0:
    print("\n MATRICES NOT READY. Fix FAIL items (especially T4 std(H) if near 0).")
    print("   Most likely cause: sigma still too large — reduce SIGMA_FRACTIONS range.")
elif n_warn > 0:
    print("\n  Matrices usable. Review WARNs before committing to 3-hour training run.")
else:
    print("\n All checks passed. Ready for RoBERTa fusion training.")