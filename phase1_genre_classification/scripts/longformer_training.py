# KAGGLE LONGFORMER-BASE-4096 TRAINING PIPELINE 
import os
import time
import pickle
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
# Swapped to Cosine Scheduler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, hamming_loss, classification_report

# Enable cuDNN benchmark for faster training
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 1. KAGGLE PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA = "/kaggle/input/datasets/mabhishek0/nlp-genre-project-data/"
SAVE = "/kaggle/working/"

print(f"Loading data from: {DATA}")

with open(f"{DATA}data_splits.pkl", "rb") as f:
    splits = pickle.load(f)

X_train, y_train = splits["X_train"], splits["y_train"]
X_val,   y_val   = splits["X_val"],   splits["y_val"]
X_test,  y_test  = splits["X_test"],  splits["y_test"]

with open(f"{DATA}mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

TARGET_GENRES = list(mlb.classes_)
NUM_LABELS    = len(TARGET_GENRES)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Labels ({NUM_LABELS}): {TARGET_GENRES}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TIMING & UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def fmt(seconds):
    return str(timedelta(seconds=int(seconds)))

def eta(elapsed_steps, total_steps, elapsed_seconds):
    if elapsed_steps == 0: return "calculating..."
    pace = elapsed_seconds / elapsed_steps
    return fmt(pace * (total_steps - elapsed_steps))

def get_pos_weight(y_train, device):
    counts = y_train.sum(axis=0)
    weight = (len(y_train) - counts) / (counts + 1e-8)
    return torch.FloatTensor(weight).to(device)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET & MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "allenai/longformer-base-4096"
MAX_LEN = 768  # Ideal balance of Context vs Speed for 4 Epochs

class UltraFastLongformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=768, device="cuda"):
        self.labels = torch.FloatTensor(labels).to(device)
        print(f"Tokenizing {len(texts)} texts for Longformer (max_len={max_len})...")
        
        encodings = tokenizer(
            list(texts),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        self.input_ids = encodings["input_ids"].to(device)
        self.attention_mask = encodings["attention_mask"].to(device)
        
        # Longformer REQUIRED: Global attention on the [CLS] token
        global_attention_mask = torch.zeros_like(self.input_ids)
        global_attention_mask[:, 0] = 1 
        self.global_attention_mask = global_attention_mask.to(device)

    def __len__(self): 
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "global_attention_mask": self.global_attention_mask[idx],
            "labels": self.labels[idx]
        }

class LongformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.longformer = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(0.3)
        self.clf  = nn.Linear(768, NUM_LABELS)

    def forward(self, input_ids, attention_mask, global_attention_mask):
        out = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.clf(self.drop(cls))

# ─────────────────────────────────────────────────────────────────────────────
# 4. INITIALIZATION & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\n--- Loading data directly into GPU VRAM ---")
tok_start = time.time()
train_ds = UltraFastLongformerDataset(X_train, y_train, tokenizer, max_len=MAX_LEN, device=device)
val_ds   = UltraFastLongformerDataset(X_val,   y_val,   tokenizer, max_len=MAX_LEN, device=device)
test_ds  = UltraFastLongformerDataset(X_test,  y_test,  tokenizer, max_len=MAX_LEN, device=device)
print(f"All splits loaded in {fmt(time.time() - tok_start)}")

BATCH_SIZE = 8   
ACCUM_STEPS = 4  
EPOCHS = 4

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = LongformerClassifier().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight(y_train, device))
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

total_optim_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS

# Optimization: Cosine Decay Scheduler instead of Linear
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_optim_steps),
    num_training_steps=total_optim_steps
)

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# We will save the best thresholds as an array
global_best_thresholds = np.full(NUM_LABELS, 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP WITH LAYER FREEZING
# ─────────────────────────────────────────────────────────────────────────────
best_macro = 0.0
global_step = 0
train_start = time.time()

print("\n" + "=" * 65)
print(f"Longformer Ultimate Training Started.")
print(f"Physical Batch: {BATCH_SIZE} | Accum Steps: {ACCUM_STEPS} | Eff Batch: {BATCH_SIZE * ACCUM_STEPS}")
print(f"Max len: {MAX_LEN} | Epochs: {EPOCHS} | Scheduler: Cosine")
print("=" * 65)

for epoch in range(EPOCHS):
    epoch_start = time.time()
    running_loss = 0.0
    
    # --- Optimization: Dynamic Layer Freezing ---
    if epoch < 2:
        # Freeze embeddings and bottom 6 layers for first 2 epochs
        for param in model.longformer.embeddings.parameters(): param.requires_grad = False
        for i in range(6):
            for param in model.longformer.encoder.layer[i].parameters(): param.requires_grad = False
        freeze_status = "[❄️ Bottom 6 Layers FROZEN for speed]"
    else:
        # Unfreeze all layers for final fine-tuning
        for param in model.longformer.parameters(): param.requires_grad = True
        freeze_status = "[🔥 All Layers UNFREEZED for fine-tuning]"
    
    model.train()
    optimizer.zero_grad()
    
    print(f"\n{'─'*65}")
    print(f"EPOCH {epoch+1}/{EPOCHS}  |  Elapsed so far: {fmt(time.time() - train_start)}")
    print(f"{freeze_status}")
    print(f"{'─'*65}")

    for step, batch in enumerate(train_loader):
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(
                batch["input_ids"], 
                batch["attention_mask"], 
                batch["global_attention_mask"]
            )
            loss = criterion(logits, batch["labels"]) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if ((step + 1) % ACCUM_STEPS == 0) or ((step + 1) == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * ACCUM_STEPS)
        global_step  += 1

        if (step + 1) % (50 * ACCUM_STEPS) == 0:
            avg_loss = running_loss / (step + 1)
            epoch_elapsed = time.time() - epoch_start
            total_elapsed = time.time() - train_start
            steps_left_ep = len(train_loader) - (step + 1)
            pace_ep = epoch_elapsed / (step + 1)
            
            print(f"  Step {step+1:>5}/{len(train_loader)}"
                  f"  |  Loss: {avg_loss:.4f}"
                  f"  |  Epoch ETA: {fmt(pace_ep * steps_left_ep)}"
                  f"  |  Total ETA: {eta(global_step, len(train_loader)*EPOCHS, total_elapsed)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. PER-GENRE THRESHOLD VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nEvaluating Epoch {epoch+1}...")
    model.eval()
    all_logits, all_labels_val = [], []

    with torch.no_grad():
        for b in val_loader:
            logits = model(b["input_ids"], b["attention_mask"], b["global_attention_mask"])
            all_logits.append(logits.cpu().numpy())
            all_labels_val.append(b["labels"].cpu().numpy())

    all_logits_np = np.vstack(all_logits)
    all_labels_np = np.vstack(all_labels_val)
    all_probs_np  = 1 / (1 + np.exp(-all_logits_np)) # Sigmoid

    # --- Optimization: Per-Genre Threshold Tuning ---
    best_t_ep = np.zeros(NUM_LABELS)
    for i in range(NUM_LABELS):
        best_f1_col = 0.0
        best_t_col = 0.5
        for t in np.arange(0.20, 0.85, 0.05):
            preds = (all_probs_np[:, i] > t).astype(int)
            f1 = f1_score(all_labels_np[:, i], preds, zero_division=0)
            if f1 > best_f1_col:
                best_f1_col = f1
                best_t_col = t
        best_t_ep[i] = best_t_col

    # Apply the unique thresholds across all columns to get overall Macro-F1
    best_preds_ep = (all_probs_np > best_t_ep).astype(int)
    ep_macro_f1 = f1_score(all_labels_np, best_preds_ep, average="macro", zero_division=0)

    print(f"Epoch {epoch+1} Val Macro-F1: {ep_macro_f1:.4f}")
    
    if ep_macro_f1 > best_macro:
        best_macro = ep_macro_f1
        global_best_thresholds = best_t_ep.copy() # Save the winning array
        torch.save(model.state_dict(), f"{SAVE}longformer_best.pt")
        print(f"  -> Saved new best model to {SAVE}longformer_best.pt")
        
        # Print the winning thresholds to show the user it worked
        print("  -> Best Thresholds per Genre:")
        t_strings = [f"{TARGET_GENRES[i]}: {best_t_ep[i]:.2f}" for i in range(min(5, NUM_LABELS))]
        print(f"     {', '.join(t_strings)} ...")

print(f"\n{'='*65}")
print(f"Training Complete! Total Session Time: {fmt(time.time() - train_start)}")
print(f"Best Validation Macro-F1: {best_macro:.4f}")
print(f"{'='*65}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL TEST SET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading best model for final test set evaluation...")
model.load_state_dict(torch.load(f"{SAVE}longformer_best.pt", map_location=device))
model.eval()

all_logits_test, all_labels_test = [], []
with torch.no_grad():
    for b in test_loader:
        logits = model(b["input_ids"], b["attention_mask"], b["global_attention_mask"])
        all_logits_test.append(logits.cpu().numpy())
        all_labels_test.append(b["labels"].cpu().numpy())

L = np.vstack(all_labels_test)
Test_Probs = 1 / (1 + np.exp(-np.vstack(all_logits_test)))

# Apply the PER-GENRE best thresholds found during validation
P = (Test_Probs > global_best_thresholds).astype(int)

print(f"\n=== Longformer Final Test Results (Per-Genre Tuned) ===")
print(f"Micro-F1:     {f1_score(L, P, average='micro', zero_division=0):.4f}")
print(f"Macro-F1:     {f1_score(L, P, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(L, P):.4f}")
print("\n" + classification_report(L, P, target_names=TARGET_GENRES, zero_division=0))