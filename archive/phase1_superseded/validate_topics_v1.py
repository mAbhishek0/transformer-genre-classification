import numpy as np
import os
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

DATA_DIR = r"D:\NLP_Project\NLP_Project\data"

train = np.load(os.path.join(DATA_DIR, "topic_probs_train.npy"))
val   = np.load(os.path.join(DATA_DIR, "topic_probs_val.npy"))
test  = np.load(os.path.join(DATA_DIR, "topic_probs_test.npy"))

print("========================================")
print("ADVANCED TOPIC VALIDATION SUITE")
print("========================================")

# =========================================
# BASIC INFO
# =========================================
print("\n=== BASIC INFO ===")
print(f"Train: {train.shape}")
print(f"Val:   {val.shape}")
print(f"Test:  {test.shape}")

num_topics = train.shape[1]

# =========================================
# 1. ENTROPY (REFINED)
# =========================================
print("\n=== ENTROPY ANALYSIS ===")

def entropy_analysis(name, arr):
    ent = entropy(arr.T)
    max_ent = np.log(num_topics)

    print(f"{name}:")
    print(f"  Mean Entropy: {ent.mean():.4f} / {max_ent:.4f}")
    print(f"  Normalized: {(ent.mean()/max_ent):.4f}")

    if ent.mean()/max_ent > 0.85:
        print("  ⚠️ Too uniform (weak signal)")
    elif ent.mean()/max_ent < 0.4:
        print("  ❌ Too sharp (overconfident)")
    else:
        print("  ✅ Good balance")

entropy_analysis("Train", train)
entropy_analysis("Val", val)
entropy_analysis("Test", test)

# =========================================
# 2. TOP-K MASS (VERY IMPORTANT)
# =========================================
print("\n=== TOP-K MASS ANALYSIS ===")

def topk_mass(name, arr, k=3):
    sorted_probs = np.sort(arr, axis=1)[:, ::-1]
    topk = sorted_probs[:, :k].sum(axis=1)

    print(f"{name} Top-{k} mass:")
    print(f"  Mean: {topk.mean():.4f}")

    if topk.mean() < 0.3:
        print("  ❌ Weak topic signal (too flat)")
    elif topk.mean() > 0.8:
        print("  ⚠️ Too concentrated")
    else:
        print("  ✅ Good concentration")

topk_mass("Train", train)
topk_mass("Val", val)
topk_mass("Test", test)

# =========================================
# 3. VARIANCE PER TOPIC
# =========================================
print("\n=== TOPIC VARIANCE ===")

def topic_variance(name, arr):
    var = arr.var(axis=0)
    low_var = np.sum(var < 1e-5)

    print(f"{name}:")
    print(f"  Mean variance: {var.mean():.6f}")
    print(f"  Low variance topics: {low_var}")

    if low_var > 5:
        print("  ❌ Many uninformative topics")
    else:
        print("  ✅ Topics carry signal")

topic_variance("Train", train)

# =========================================
# 4. CORRELATION CHECK (REDUNDANCY)
# =========================================
print("\n=== TOPIC CORRELATION ===")

corr = np.corrcoef(train.T)
high_corr = np.sum((corr > 0.9) & (corr < 0.999))

print(f"Highly correlated topic pairs (>0.9): {high_corr}")

if high_corr > 20:
    print("❌ Too many redundant topics")
else:
    print("✅ Topics are diverse")

# =========================================
# 5. KL DIVERGENCE FROM UNIFORM
# =========================================
print("\n=== INFORMATION CONTENT ===")

uniform = np.ones(num_topics) / num_topics

def kl_check(name, arr):
    kl_vals = []
    for row in arr:
        kl_vals.append(np.sum(row * np.log(row / uniform + 1e-12)))

    kl_vals = np.array(kl_vals)

    print(f"{name}:")
    print(f"  Mean KL divergence: {kl_vals.mean():.4f}")

    if kl_vals.mean() < 0.1:
        print("  ❌ Too close to uniform (useless features)")
    elif kl_vals.mean() > 3:
        print("  ⚠️ Too spiky")
    else:
        print("  ✅ Good information content")

kl_check("Train", train)

# =========================================
# 6. EFFECTIVE NUMBER OF TOPICS
# =========================================
print("\n=== EFFECTIVE TOPIC COUNT ===")

def effective_topics(arr):
    ent = entropy(arr.T)
    return np.exp(ent.mean())

eff_topics = effective_topics(train)

print(f"Effective topics used: {eff_topics:.2f} / {num_topics}")

if eff_topics > num_topics * 0.9:
    print("❌ Almost uniform usage (bad)")
elif eff_topics < num_topics * 0.3:
    print("⚠️ Too few topics used")
else:
    print("✅ Healthy topic usage")

# =========================================
# 7. JS DIVERGENCE BETWEEN SPLITS
# =========================================
print("\n=== SPLIT CONSISTENCY (JS DIVERGENCE) ===")

train_mean = train.mean(axis=0)
val_mean   = val.mean(axis=0)
test_mean  = test.mean(axis=0)

js_val = jensenshannon(train_mean, val_mean)
js_test = jensenshannon(train_mean, test_mean)

print(f"JS(train, val):  {js_val:.4f}")
print(f"JS(train, test): {js_test:.4f}")

if js_val > 0.1:
    print("❌ Distribution mismatch")
else:
    print("✅ Stable distributions")

# =========================================
# 8. SAMPLE INSPECTION (CRITICAL)
# =========================================
print("\n=== SAMPLE INSPECTION ===")

for i in range(3):
    top5 = np.sort(train[i])[-5:][::-1]
    print(f"Sample {i} top-5 probs: {top5}")

print("\n========================================")
print("FINAL INTERPRETATION GUIDE")
print("========================================")
print("""
❌ = WILL hurt model performance
⚠️ = Might hurt performance
✅ = Good

KEY SIGNALS TO WATCH:
- Top-3 mass < 0.3  → weak topics
- KL < 0.1         → useless features
- Effective topics ~50 → too uniform
- High correlation → redundant topics
""")