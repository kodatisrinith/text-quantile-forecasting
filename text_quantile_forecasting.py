import os
import numpy as np
import pandas as pd
import pyreadstat
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

FILE_PATH  = "/path/to/your/data.dta"

TEXT_COL   = "TEXT"
LABEL_COL  = "SAL_ACTUAL_ibes"
ID_COL     = "fiscal_year"

MODEL_NAME = "roberta-large"
SPLIT_YEAR = 2013

MAX_LENGTH = 512
STRIDE     = 128

BEST_LR     = 3e-5
BEST_EPOCHS = 30

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

SAVE_DIR = "/path/to/output/directory"

TRAIN_EMB_PATH  = f"{SAVE_DIR}/SAL_qr_train_emb.pt"
TEST_EMB_PATH   = f"{SAVE_DIR}/SAL_qr_test_emb.pt"
NORM_STATS_PATH = f"{SAVE_DIR}/SAL_qr_norm_stats.pt"
TRAIN_SEQ_PATH  = f"{SAVE_DIR}/SAL_qr_train_seq.pt"
TEST_SEQ_PATH   = f"{SAVE_DIR}/SAL_qr_test_seq.pt"
FINAL_SAVE_PATH = f"{SAVE_DIR}/SAL_quantile_predictions_v2.dta"
ATTN_DTA_PATH   = f"{SAVE_DIR}/SAL_attended_text.dta"

REGENERATE_EMBEDDINGS = True
REBUILD_SEQUENCES     = True
RETRAIN_MODEL         = True

TOP_K = 6
TEMP  = 3.5

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)
print("Device:", device)


# =============================================================================
# DATA LOADING
# =============================================================================

df, _ = pyreadstat.read_dta(FILE_PATH)

df = df[df[TEXT_COL].notna()]
df = df[df[TEXT_COL].astype(str).str.strip() != ""]
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
df = df[df[LABEL_COL].notna()]
df = df[df[LABEL_COL] > 0].copy()

df["label_log"] = np.log(df[LABEL_COL])
df = df.sort_values(ID_COL).reset_index(drop=True)
df["original_row_id"] = df.index

print("Rows:", len(df))


# =============================================================================
# TRAIN / TEST SPLIT AND NORMALISATION
# =============================================================================

train_df = df[df[ID_COL] <= SPLIT_YEAR].copy()
test_df  = df[df[ID_COL] >  SPLIT_YEAR].copy()

label_mean = train_df["label_log"].mean()
label_std  = train_df["label_log"].std()
label_std  = label_std if label_std > 1e-8 else 1e-8

train_df["label_norm"] = (train_df["label_log"] - label_mean) / label_std
test_df["label_norm"]  = (test_df["label_log"]  - label_mean) / label_std

torch.save({"label_mean": label_mean, "label_std": label_std}, NORM_STATS_PATH)


# =============================================================================
# TOKENISER
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

def generate_embeddings(split_df, name):
    roberta = AutoModel.from_pretrained(MODEL_NAME).to(device)
    roberta.eval()

    reps, ids, labels, labels_orig = [], [], [], []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=name):
        text      = str(row[TEXT_COL])
        label     = float(row["label_norm"])
        label_og  = float(row[LABEL_COL])
        rid       = int(row["original_row_id"])

        enc = tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt"
        )

        with torch.no_grad():
            for i in range(enc["input_ids"].shape[0]):
                out = roberta(
                    input_ids=enc["input_ids"][i].unsqueeze(0).to(device),
                    attention_mask=enc["attention_mask"][i].unsqueeze(0).to(device)
                )
                cls = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
                reps.append(cls)
                ids.append(rid)
                labels.append(label)
                labels_orig.append(label_og)

    return {"reps": reps, "ids": ids, "labels": labels, "labels_orig": labels_orig}


if REGENERATE_EMBEDDINGS:
    train_cache = generate_embeddings(train_df, "train")
    test_cache  = generate_embeddings(test_df,  "test")
    torch.save(train_cache, TRAIN_EMB_PATH)
    torch.save(test_cache,  TEST_EMB_PATH)
else:
    train_cache = torch.load(TRAIN_EMB_PATH, weights_only=False)
    test_cache  = torch.load(TEST_EMB_PATH,  weights_only=False)


# =============================================================================
# DOCUMENT SEQUENCE CONSTRUCTION
# =============================================================================

def build_sequences(reps, ids, labels, labels_orig):
    seqs     = defaultdict(list)
    lab      = {}
    lab_orig = {}

    for r, i, y, yo in zip(reps, ids, labels, labels_orig):
        seqs[i].append(r)
        lab[i]      = y
        lab_orig[i] = yo

    X      = [torch.stack(v) for v in seqs.values()]
    y      = torch.tensor([lab[i]      for i in seqs.keys()])
    y_orig = torch.tensor([lab_orig[i] for i in seqs.keys()])
    rid    = list(seqs.keys())

    return X, y, y_orig, rid


if REBUILD_SEQUENCES:
    X_train, y_train, y_train_orig, rid_train = build_sequences(**train_cache)
    X_test,  y_test,  y_test_orig,  rid_test  = build_sequences(**test_cache)

    torch.save({"X": X_train, "y": y_train, "y_orig": y_train_orig, "rid": rid_train}, TRAIN_SEQ_PATH)
    torch.save({"X": X_test,  "y": y_test,  "y_orig": y_test_orig,  "rid": rid_test},  TEST_SEQ_PATH)
else:
    tr = torch.load(TRAIN_SEQ_PATH, weights_only=False)
    te = torch.load(TEST_SEQ_PATH,  weights_only=False)

    X_train, y_train, y_train_orig, rid_train = tr["X"], tr["y"], tr["y_orig"], tr["rid"]
    X_test,  y_test,  y_test_orig,  rid_test  = te["X"], te["y"], te["y_orig"], te["rid"]


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class MultiQuantileModel(nn.Module):

    def __init__(self, dim, nq):
        super().__init__()
        self.lstm = nn.LSTM(
            dim, dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )
        self.base  = nn.Linear(dim, 1)
        self.delta = nn.Linear(dim, nq - 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        w      = torch.softmax(self.attn(out), dim=1)
        pooled = (out * w).sum(dim=1)
        q1     = self.base(pooled)
        d      = F.softplus(self.delta(pooled))
        qs     = torch.cat([q1, q1 + torch.cumsum(d, dim=1)], dim=1)
        return qs, w.squeeze(-1)


class MultiPinballLoss(nn.Module):

    def __init__(self, quantiles):
        super().__init__()
        self.q = torch.tensor(quantiles).view(1, -1)

    def forward(self, preds, target):
        q    = self.q.to(preds.device)
        y    = target.view(-1, 1).expand_as(preds)
        e    = y - preds
        loss = torch.maximum(q * e, (q - 1) * e)
        return loss.mean()


# =============================================================================
# TRAINING
# =============================================================================

if RETRAIN_MODEL:
    model = MultiQuantileModel(
        dim=X_train[0].shape[1],
        nq=len(QUANTILES)
    ).to(device)

    nn.init.constant_(model.delta.bias, 0.5)

    opt     = torch.optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=0.01)
    loss_fn = MultiPinballLoss(QUANTILES)

    for epoch in range(BEST_EPOCHS):
        model.train()
        total_loss = 0

        for x, y in zip(X_train, y_train):
            x = x.unsqueeze(0).to(device)
            y = torch.tensor([y.item()], device=device)

            preds, _ = model(x)
            loss     = loss_fn(preds, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1:>3}  loss  {total_loss / len(X_train):.6f}")


# =============================================================================
# INFERENCE
# =============================================================================

norm_stats = torch.load(NORM_STATS_PATH, weights_only=False)
label_mean = norm_stats["label_mean"]
label_std  = norm_stats["label_std"]
label_std  = label_std if label_std > 1e-8 else 1e-8


def scale_quantile_predictions(preds_actual):
    median       = preds_actual[4]
    preds_scaled = preds_actual.copy()
    for i, val in enumerate(preds_actual):
        if val < median:
            preds_scaled[i] = median + 1.5  * (val - median)
        else:
            preds_scaled[i] = median + TEMP * (val - median)
    return preds_scaled


def predict_quantiles(model, x_doc):
    with torch.no_grad():
        preds_norm, _ = model(x_doc.unsqueeze(0).to(device))
        preds_norm    = preds_norm.squeeze(0).cpu().numpy()
        preds_actual  = np.exp(preds_norm * label_std + label_mean)
    preds_scaled = scale_quantile_predictions(preds_actual)
    return {f"q{int(q * 100):03d}_pred": v for q, v in zip(QUANTILES, preds_scaled)}


model.eval()
all_results = []

for rid, x_doc in tqdm(zip(rid_test, X_test), total=len(X_test)):
    preds = predict_quantiles(model, x_doc)
    preds["original_row_id"] = rid
    all_results.append(preds)

pred_df = pd.DataFrame(all_results)
pred_df["q_median"]       = pred_df["q050_pred"]
pred_df["q_interval"]     = pred_df["q090_pred"] - pred_df["q010_pred"]
pred_df["q_rel_interval"] = pred_df["q_interval"] / pred_df["q050_pred"]

test_meta = df[df[ID_COL] > SPLIT_YEAR][[
    "original_row_id", ID_COL, LABEL_COL
]].copy()

pred_df  = pred_df.merge(test_meta, on="original_row_id", how="left")
df_final = df.merge(
    pred_df.drop(columns=[ID_COL, LABEL_COL]),
    on="original_row_id",
    how="left"
)
df_final = df_final.drop(columns=["label_log"], errors="ignore")

for col in df_final.columns:
    if df_final[col].dtype == "object":
        df_final[col] = df_final[col].astype(str)

df_final.to_stata(FINAL_SAVE_PATH, write_index=False, version=118)
print("Saved:", FINAL_SAVE_PATH)


# =============================================================================
# ATTENDED TEXT EXTRACTION
# =============================================================================

def get_chunk_texts(text):
    enc = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )
    return [
        tokenizer.decode(
            enc["input_ids"][i],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        for i in range(enc["input_ids"].shape[0])
    ]


rid_to_text  = dict(zip(
    test_df["original_row_id"].astype(int).tolist(),
    test_df[TEXT_COL].astype(str).tolist()
))
rid_to_fyear = dict(zip(
    test_df["original_row_id"].astype(int).tolist(),
    test_df[ID_COL].tolist()
))

rows = []
model.eval()

for rid, x_doc in tqdm(zip(rid_test, X_test), total=len(X_test), desc="Extracting attended text"):
    rid = int(rid)
    with torch.no_grad():
        _, w = model(x_doc.unsqueeze(0).to(device))
        w    = w.squeeze(0).cpu().numpy()

    n_chunks    = len(w)
    chunk_texts = get_chunk_texts(rid_to_text.get(rid, ""))

    if len(chunk_texts) > n_chunks:
        chunk_texts = chunk_texts[:n_chunks]
    elif len(chunk_texts) < n_chunks:
        chunk_texts += [""] * (n_chunks - len(chunk_texts))

    k             = min(TOP_K, n_chunks)
    top_idx       = sorted(np.argsort(w)[::-1][:k])
    attended_text = " ".join(chunk_texts[i] for i in top_idx)

    rows.append({
        "original_row_id": rid,
        ID_COL:            rid_to_fyear.get(rid),
        "attended_text":   attended_text,
        "n_chunks":        n_chunks
    })

attn_df = pd.DataFrame(rows)
print(f"Extraction complete: {len(attn_df):,} firm-years")

for col in attn_df.select_dtypes("object").columns:
    attn_df[col] = attn_df[col].astype(str)

attn_df.to_stata(ATTN_DTA_PATH, write_index=False, version=118)
print(f"Attended text saved: {ATTN_DTA_PATH}")
