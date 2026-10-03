# MULTI-MODEL INFERENCE SCRIPT
# Supports: RoBERTa-Optimal, DistilBERT-Optimal, RoBERTa-Fusion, Longformer

import torch
import torch.nn as nn
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MLB_PATH = "mlb.pkl"

# Prediction behavior
THRESHOLD        = 0.5        # base threshold
TOP_K_FALLBACK   = 3          # ensure at least K labels
MIN_PROB_DISPLAY = 0.05       # for printing
BATCH_SIZE       = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS (must match training architectures)
# ─────────────────────────────────────────────────────────────────────────────

class RoBERTaClassifier(nn.Module):
    """Standard RoBERTa classifier (roberta_optimal_best.pt)"""
    def __init__(self, num_labels):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.drop    = nn.Dropout(0.3)
        self.clf     = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.clf(self.drop(cls))


class DistilBERTClassifier(nn.Module):
    """DistilBERT classifier (distilbert_optimal_best.pt)"""
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.drop = nn.Dropout(0.3)
        self.clf  = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.clf(self.drop(cls))


class RoBERTaFusionClassifier(nn.Module):
    """
    Two-stream RoBERTa + Topic Fusion classifier (roberta_fusion_best.pt)

    Stream A — RoBERTa [CLS]:
        roberta(input_ids, mask) → cls (768)
        → Linear(768, 256) → LayerNorm(256) → GELU → Dropout(0.3) → cls_proj (256)

    Stream B — Topic probs:
        topic_probs (30)
        → Linear(30, 64) → LayerNorm(64) → GELU → Dropout(0.2) → topic_proj (64)

    Fusion:
        Concat([cls_proj, topic_proj]) (320) → Linear(320, 12)
    """
    def __init__(self, topic_dim, num_labels):
        super().__init__()

        # Stream A — RoBERTa encoder
        self.roberta = AutoModel.from_pretrained("roberta-base")

        self.cls_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Stream B — Topic projection
        self.topic_proj = nn.Sequential(
            nn.Linear(topic_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Fusion head
        self.classifier = nn.Linear(256 + 64, num_labels)

    def forward(self, input_ids, attention_mask, topic_probs=None):
        # Stream A
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        cls_out = self.cls_proj(cls)            # (B, 256)

        # Stream B — use zeros if no topic probs provided (inference fallback)
        if topic_probs is None:
            topic_probs = torch.zeros(cls.size(0), 30, device=cls.device)
        topic_out = self.topic_proj(topic_probs)  # (B, 64)

        # Fuse
        fused = torch.cat([cls_out, topic_out], dim=1)   # (B, 320)
        return self.classifier(fused)                      # (B, num_labels)


class LongformerClassifier(nn.Module):
    """Longformer classifier (longformer_best.pt)"""
    def __init__(self, num_labels):
        super().__init__()
        self.longformer = AutoModel.from_pretrained("allenai/longformer-base-4096")
        self.drop = nn.Dropout(0.3)
        self.clf  = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, global_attention_mask=None):
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
        out = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.clf(self.drop(cls))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "roberta_optimal": {
        "weights": "roberta_optimal_best.pt",
        "base_model": "roberta-base",
        "max_len": 512,
        "model_class": "RoBERTaClassifier",
        "display_name": "RoBERTa-Optimal",
    },
    "distilbert_optimal": {
        "weights": "distilbert_optimal_best.pt",
        "base_model": "distilbert-base-uncased",
        "max_len": 512,
        "model_class": "DistilBERTClassifier",
        "display_name": "DistilBERT-Optimal",
    },
    "roberta_fusion": {
        "weights": "roberta_fusion_best.pt",
        "base_model": "roberta-base",
        "max_len": 512,
        "model_class": "RoBERTaFusionClassifier",
        "display_name": "RoBERTa-Fusion",
        "topic_dim": 30,
    },
    "longformer": {
        "weights": "longformer_best.pt",
        "base_model": "allenai/longformer-base-4096",
        "max_len": 768,
        "model_class": "LongformerClassifier",
        "display_name": "Longformer",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def load_mlb(path):
    with open(path, "rb") as f:
        mlb = pickle.load(f)
    return mlb, list(mlb.classes_)


def load_model(config, num_labels):
    """Instantiate and load weights for any supported model."""
    class_name = config["model_class"]
    weights_path = config["weights"]

    print(f"  Loading {config['display_name']} from: {weights_path}")

    # Instantiate the correct architecture
    if class_name == "RoBERTaClassifier":
        model = RoBERTaClassifier(num_labels).to(DEVICE)
    elif class_name == "DistilBERTClassifier":
        model = DistilBERTClassifier(num_labels).to(DEVICE)
    elif class_name == "RoBERTaFusionClassifier":
        topic_dim = config.get("topic_dim", 30)
        model = RoBERTaFusionClassifier(topic_dim, num_labels).to(DEVICE)
    elif class_name == "LongformerClassifier":
        model = LongformerClassifier(num_labels).to(DEVICE)
    else:
        raise ValueError(f"Unknown model class: {class_name}")

    checkpoint = torch.load(weights_path, map_location=DEVICE)

    # Handle different save formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZATION (BATCHED)
# ─────────────────────────────────────────────────────────────────────────────
def tokenize_batch(texts, tokenizer, max_len=512):
    return tokenizer(
        texts,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def decode_predictions(
    probs: np.ndarray,
    labels: List[str],
    threshold: float = THRESHOLD,
    top_k: int = TOP_K_FALLBACK
) -> Tuple[List[str], List[Tuple[str, float]]]:

    # threshold selection
    selected = np.where(probs >= threshold)[0]

    # fallback if nothing selected
    if len(selected) == 0:
        selected = np.argsort(probs)[-top_k:]

    # sort by confidence
    sorted_idx = sorted(selected, key=lambda i: probs[i], reverse=True)

    pred_labels = [labels[i] for i in sorted_idx]
    ranked = [(labels[i], float(probs[i])) for i in sorted_idx]

    return pred_labels, ranked


def predict_batch(texts: List[str], model, tokenizer, labels, config):
    """Run inference for a batch of texts using the specified model config."""
    results = []
    max_len = config.get("max_len", 512)
    class_name = config["model_class"]

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        enc = tokenize_batch(batch, tokenizer, max_len=max_len)
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                if class_name == "LongformerClassifier":
                    # Longformer needs global_attention_mask on [CLS]
                    global_attention_mask = torch.zeros_like(input_ids)
                    global_attention_mask[:, 0] = 1
                    logits = model(input_ids, attention_mask, global_attention_mask)
                elif class_name == "RoBERTaFusionClassifier":
                    # Fusion model — pass zeros for topic probs during inference
                    topic_probs = torch.zeros(input_ids.size(0), 30, device=DEVICE)
                    logits = model(input_ids, attention_mask, topic_probs)
                else:
                    # Standard models (RoBERTa, DistilBERT)
                    logits = model(input_ids, attention_mask)

                probs = torch.sigmoid(logits).cpu().numpy()

        for text, p in zip(batch, probs):
            preds, ranked = decode_predictions(p, labels)

            full_scores = {
                labels[j]: float(p[j]) for j in range(len(labels))
            }

            results.append({
                "text": text,
                "predictions": preds,
                "ranked": ranked,
                "scores": full_scores
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINTING
# ─────────────────────────────────────────────────────────────────────────────
def print_results(results, model_name=""):
    print("=" * 80)
    print(f"INFERENCE RESULTS — {model_name}")
    print("=" * 80)

    for idx, res in enumerate(results, 1):
        print(f"\nInput {idx}")
        print(f"Text: {res['text'][:120]}...")

        print("\nTop Predictions:")
        for label, prob in res["ranked"]:
            print(f"  {label:<20} {prob*100:.2f}%")

        print("\nOther notable scores (>5%):")
        for label, prob in res["scores"].items():
            if prob >= MIN_PROB_DISPLAY and label not in res["predictions"]:
                print(f"  {label:<20} {prob*100:.2f}%")

        print("-" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print(f"Using device: {DEVICE}")

    # Load label encoder (shared across all models)
    mlb, labels = load_mlb(MLB_PATH)
    print(f"Loaded {len(labels)} labels: {labels}\n")

    # Test inputs
    TEST_TEXTS = [
        "A cybernetic soldier fights rogue AI controlling a dystopian megacity.",
        "Two childhood friends reunite years later and rediscover their love.",
        "An abandoned hospital hides a terrifying supernatural entity."
    ]

    # ── Run inference for each available model ────────────────────────────────
    models_to_run = [
        "roberta_optimal",
        "distilbert_optimal",
        "roberta_fusion",
        "longformer",
    ]

    for model_key in models_to_run:
        config = MODEL_CONFIGS[model_key]
        weights_path = config["weights"]

        # Skip if weights file doesn't exist
        if not os.path.exists(weights_path):
            print(f"\n{'='*80}")
            print(f"SKIPPING {config['display_name']} — {weights_path} not found")
            print(f"{'='*80}")
            continue

        print(f"\n{'─'*80}")
        print(f"Loading {config['display_name']}...")
        print(f"{'─'*80}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        model = load_model(config, len(labels))
        print(f"  Model loaded successfully.\n")

        # Run predictions
        results = predict_batch(TEST_TEXTS, model, tokenizer, labels, config)
        print_results(results, model_name=config["display_name"])

        # Free memory before loading next model
        del model
        del tokenizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ALL INFERENCE COMPLETE")
    print("=" * 80)