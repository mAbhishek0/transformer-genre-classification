# ROBERTA INFERENCE TEST 
import torch
import torch.nn as nn
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update paths if needed
# ─────────────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS = "roberta_base_best.pt"   # your .pt file
MLB_PATH      = "mlb.pkl"               # genre label decoder
THRESHOLD     = 0.75                    # best threshold from your run
MAX_LEN       = 512
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# TEST INPUTS — 3 hardcoded texts
# ─────────────────────────────────────────────────────────────────────────────
TEST_TEXTS = [
    # 1 — clear action/sci-fi
    (
        "In a distant future, a retired soldier is forced back into action "
        "when an alien fleet launches a surprise invasion on Earth. Armed with "
        "experimental weaponry, he leads a small squad behind enemy lines to "
        "destroy the alien mothership before humanity is wiped out."
    ),
    # 2 — romance/drama
    (
        "Two strangers meet on a train journey across Europe. She is a "
        "concert pianist fleeing a broken engagement. He is a journalist "
        "chasing a story he no longer believes in. Over three days they fall "
        "in love, but their pasts threaten to pull them apart before the "
        "journey ends."
    ),
    # 3 — horror
    (
        "A family moves into an old Victorian mansion after inheriting it "
        "from a distant relative. Strange noises at night, doors that open "
        "by themselves, and a locked basement no one can explain. When the "
        "youngest daughter starts talking to someone no one else can see, "
        "the father begins to unravel the mansion's dark and murderous history."
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION — must match exactly what was used during training
# ─────────────────────────────────────────────────────────────────────────────
class RoBERTaClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.drop    = nn.Dropout(0.3)
        self.clf     = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.clf(self.drop(cls))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MLB, TOKENIZER, MODEL
# ─────────────────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")

# Load genre label decoder
with open(MLB_PATH, "rb") as f:
    mlb = pickle.load(f)
TARGET_GENRES = list(mlb.classes_)
NUM_LABELS    = len(TARGET_GENRES)
print(f"Genres ({NUM_LABELS}): {TARGET_GENRES}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Load model and weights
print("Loading model weights...")
model = RoBERTaClassifier(num_labels=NUM_LABELS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()
print("Model ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(text, threshold=THRESHOLD):
    """
    Takes a raw text string.
    Returns a list of predicted genre strings.
    """
    # Tokenize
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)         # shape: (1, 12)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]   # shape: (12,)

    # Apply threshold
    binary_preds = (probs > threshold).astype(int)

    # Decode to genre names
    predicted_genres = [
        TARGET_GENRES[i] for i, v in enumerate(binary_preds) if v == 1
    ]

    # Also return per-genre probabilities for transparency
    prob_breakdown = {
        TARGET_GENRES[i]: f"{probs[i]*100:.1f}%"
        for i in range(NUM_LABELS)
    }

    return predicted_genres, prob_breakdown


# ─────────────────────────────────────────────────────────────────────────────
# RUN ON TEST INPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("INFERENCE RESULTS")
print("=" * 65)

for i, text in enumerate(TEST_TEXTS, 1):
    genres, probs_dict = predict(text)

    print(f"\nInput {i}:")
    print(f"  Text   : {text[:100]}...")
    print(f"  Genres : {', '.join(genres) if genres else 'None (all below threshold)'}")
    print(f"  Scores :")
    # Print only genres above 10% to keep output clean
    for genre, pct in probs_dict.items():
        prob_val = float(pct.replace("%", ""))
        if prob_val >= 10.0:
            marker = " ✓" if genre in genres else ""
            print(f"    {genre:<20} {pct}{marker}")
    print("─" * 65)