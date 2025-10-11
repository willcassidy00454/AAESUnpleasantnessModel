# Run from here to prep everything up to training
import os
import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# ---------------------------
# User-editable config
# ---------------------------
DATA_CSV: Optional[str] = "/Users/willcassidy/Development/GitHub/AAESUnpleasantnessModel-Evaluation/Data/all_results.txt"
RANDOM_SEED = 30

TARGET_PROG_ITEM = 1
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 1000
PATIENCE = 12  # early stopping on val RMSE
EMBED_STIMULUS = False  # set False if you don't want stimulus embedding
EMBEDDING_DIM = 16
HIDDEN_SIZES = [16, 32, 64]#, 8]
DROPOUTS = [0.2, 0.1, 0.0]#, 0.0]
TEST_SIZE = 0.2
VAL_SIZE = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "Src/DeepLearning/best_mlp_model.pt"

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ---------------------------
# Data loading / synthetic data
# ---------------------------
def load_data_or_synth(path: Optional[str]):
    if path and os.path.exists(path):
        print(f"Loading CSV from {path}")
        df = pd.read_csv(path)
        # Expecting columns: feature_0..feature_4, mean_score, stimulus_id
        required = ["stimulus_id", "prog_item", "rating", "colouration", "flutter_echo", "asymmetry", "curvature", "hf_damping"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        # Convert stimulus_id to integer codes if necessary
        if not np.issubdtype(df["stimulus_id"].dtype, np.integer):
            df["stimulus_id"] = pd.Categorical(df["stimulus_id"]).codes
        return df
    else:
        warnings.warn("No CSV provided or path not found.")
        # return make_synthetic_data(total_mean_scores=5544, n_stimuli=238, n_features=5)

df = load_data_or_synth(DATA_CSV)

# Filter for a specific prog_item value

if "prog_item" not in df.columns:
    raise ValueError("Expected a column named 'prog_item' in the dataset!")

df = df[df["prog_item"] == TARGET_PROG_ITEM].reset_index(drop=True)

if len(df) == 0:
    raise ValueError(f"No rows found for prog_item == {TARGET_PROG_ITEM}")

print(f"Filtered for prog_item == {TARGET_PROG_ITEM}")

feature_names = ["colouration", "flutter_echo", "curvature", "hf_damping"]

# Omit spatial asymmetry feature from saxophone
if TARGET_PROG_ITEM == 1:
    feature_names.append("asymmetry")

print(f"Dataset: {len(df)} ratings, {df['stimulus_id'].nunique()} unique stimuli, {len([c for c in df.columns if c in feature_names])} features")

# ---------------------------
# Grouped split by stimulus
# ---------------------------
def grouped_split(df, test_size=0.15, val_size=0.15, group_col="stimulus_id", random_state=RANDOM_SEED):
    """Split so groups (stimuli) are disjoint across train/val/test."""
    groups = df[group_col].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Now split train into train+val
    groups_train = df_train[group_col].values
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=random_state+1)
    train_idx2, val_idx = next(splitter2.split(df_train, groups=groups_train))
    df_train_final = df_train.iloc[train_idx2].reset_index(drop=True)
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    return df_train_final, df_val, df_test

df_train, df_val, df_test = grouped_split(df, test_size=TEST_SIZE, val_size=VAL_SIZE)
print(f"Split sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
print(f"Stimuli in splits — train: {df_train['stimulus_id'].nunique()}, val: {df_val['stimulus_id'].nunique()}, test: {df_test['stimulus_id'].nunique()}")

# ---------------------------
# Dataset / DataLoader
# ---------------------------
class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, group_col="stimulus_id"):
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = (df["rating"].values.astype(np.float32).reshape(-1, 1)) / 100.0
        self.groups = df[group_col].values.astype(np.int64)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.features[idx]),
            "target": torch.from_numpy(self.targets[idx]),
            "stimulus": int(self.groups[idx])
        }

feature_cols = [c for c in df.columns if c in feature_names]
train_ds = RatingsDataset(df_train, feature_cols)
val_ds = RatingsDataset(df_val, feature_cols)
test_ds = RatingsDataset(df_test, feature_cols)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

n_stimuli = int(df["stimulus_id"].nunique())
n_features = len(feature_cols)



# ---------------------------
# Model
# ---------------------------

# class LeakyTanh(nn.Module):
#     def __init__(self):
#         super(LeakyTanh, self).__init__()
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return torch.tanh(input) + 0.1 * input

class MLPRegressor(nn.Module):
    def __init__(self, n_features, hidden_sizes, dropout_rates, use_embedding=True, n_stimuli=None, embed_dim=16):
        super().__init__()
        self.use_embedding = use_embedding and (n_stimuli is not None)
        input_dim = n_features

        if self.use_embedding:
            self.embed = nn.Embedding(n_stimuli, embed_dim)
            input_dim += embed_dim

        layers = []

        # Layer 1
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rates[0]))

        # Layer 2
        layers.append(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        # layers.append(nn.BatchNorm1d(hidden_sizes[1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rates[1]))

        # Layer 3
        layers.append(nn.Linear(hidden_sizes[1], hidden_sizes[2]))
        # layers.append(nn.LayerNorm(hidden_sizes[2]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rates[2]))
        #
        # # Layer 4
        # layers.append(nn.Linear(hidden_sizes[2], hidden_sizes[3]))
        # # layers.append(nn.LayerNorm(hidden_sizes[3]))
        # layers.append(nn.ReLU(hidden_sizes[3]))
        # layers.append(nn.Dropout(dropout_rates[3]))

        # Output
        # layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_sizes[2], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, features, stimulus_ids=None):
        x = features
        if self.use_embedding:
            if stimulus_ids is None:
                raise ValueError("stimulus_ids required when using embedding")
            emb = self.embed(stimulus_ids)
            x = torch.cat([x, emb], dim=1)
        return self.net(x)

model = MLPRegressor(
    n_features=n_features,
    hidden_sizes=HIDDEN_SIZES,
    dropout_rates=DROPOUTS,
    use_embedding=EMBED_STIMULUS,
    n_stimuli=n_stimuli if EMBED_STIMULUS else None,
    embed_dim=EMBEDDING_DIM
).to(DEVICE)

print(model)

# ---------------------------
# Training utilities
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)


def eval_model(loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(DEVICE)
            targs = batch["target"].to(DEVICE)
            stim = batch["stimulus"].to(DEVICE)
            out = model(feats, stim if EMBED_STIMULUS else None)
            preds.append(out.cpu().numpy())
            trues.append(targs.cpu().numpy())
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return {"mse": mse, "mae": mae, "r2": r2, "preds": preds, "trues": trues}

#%% -------------------------
# Training loop with early stopping
# ---------------------------
best_val_mse = float("inf")
patience_counter = 0

training_loss = []
validation_loss = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        feats = batch["features"].to(DEVICE)
        targs = batch["target"].to(DEVICE)
        stim = batch["stimulus"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(feats, stim if EMBED_STIMULUS else None)
        loss = criterion(outputs, targs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * feats.size(0)

    train_loss = running_loss / len(train_ds)
    val_metrics = eval_model(val_loader)
    val_mse = val_metrics["mse"]

    training_loss.append(train_loss)
    validation_loss.append(val_mse)

    scheduler.step(val_mse)

    print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | val_mse: {val_mse:.4f} | val_mae: {val_metrics['mae']:.4f} | val_r2: {val_metrics['r2']:.4f}")

    # early stopping check
    if val_mse < best_val_mse - 1e-6:
        best_val_mse = val_mse
        patience_counter = 0
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_mse": val_mse
        }, MODEL_SAVE_PATH)
        print(f"  --> New best model saved (val_mse={val_mse:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs (best val_mse={best_val_mse:.4f})")
            break

plt.plot(range(len(training_loss)), training_loss, "-", label="Training Loss")
plt.plot(range(len(training_loss)), validation_loss, "--", label="Validation Loss")
plt.legend()
plt.show()

# ---------------------------
# Load best model & evaluate on test
# ---------------------------
ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
test_metrics = eval_model(test_loader)
print("\nTest set performance (using best saved model):")
print(f"  RMSE: {test_metrics['mse']:.4f}")
print(f"  MAE : {test_metrics['mae']:.4f}")
print(f"  R2  : {test_metrics['r2']:.4f}")

# Optional: save predictions to CSV
out_df = pd.DataFrame({
    "pred": test_metrics["preds"],
    "true": test_metrics["trues"]
})
out_df.to_csv("Src/DeepLearning/test_preds.csv", index=False)
print("Wrote test_preds.csv with predictions and true ratings.")

#%% -------------------------
# Stimulus-level evaluation and plot
# ---------------------------
from sklearn.linear_model import LinearRegression
import seaborn as sns

print("\nEvaluating at stimulus level (mean per stimulus)...")

# 1. Get predictions for all test samples
model.eval()
all_preds, all_trues, all_stimuli = [], [], []
with torch.no_grad():
    for batch in test_loader:
        feats = batch["features"].to(DEVICE)
        targs = batch["target"].cpu().numpy().ravel()
        stim = batch["stimulus"].cpu().numpy().ravel()
        preds = model(feats, batch["stimulus"].to(DEVICE) if EMBED_STIMULUS else None).cpu().numpy().ravel()

        all_preds.extend(preds)
        all_trues.extend(targs)
        all_stimuli.extend(stim)

df_pred = pd.DataFrame({
    "stimulus_id": all_stimuli,
    "true_rating": all_trues,
    "pred_rating": all_preds
})

# 2. Compute mean predicted and true rating per stimulus
df_mean = df_pred.groupby("stimulus_id", as_index=False).agg(
    mean_true=("true_rating", "mean"),
    mean_pred=("pred_rating", "mean")
)

# 3. Fit linear regression (predicted vs true)
X = df_mean[["mean_true"]].values
y = df_mean["mean_pred"].values
reg = LinearRegression().fit(X, y)
slope, intercept = reg.coef_[0], reg.intercept_
r2 = reg.score(X, y)

print(f"Stimulus-level regression: pred = {slope:.3f} * true + {intercept:.3f}")
print(f"R² = {r2:.4f}")

# 4. Plot
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df_mean, x="mean_true", y="mean_pred", s=60, alpha=0.7)
x_line = np.linspace(df_mean["mean_true"].min(), df_mean["mean_true"].max(), 100)
y_line = reg.predict(x_line.reshape(-1, 1))
plt.plot(x_line, y_line, color="red", lw=2, label=f"Linear fit (R²={r2:.2f})")

# Add a diagonal y=x line for perfect agreement
plt.plot(x_line, x_line, color="gray", lw=1.5, ls="--", label="Ideal (y=x)")

plt.title("Stimulus-level Mean Prediction vs. True Rating")
plt.xlabel("Mean True Rating")
plt.ylabel("Mean Predicted Rating")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
