import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import os

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


def generate_base_board():
    return np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])


def shuffle_board(board):
    b = board.copy()
    nums = np.arange(1, 10)
    np.random.shuffle(nums)
    mapping = {i + 1: nums[i] for i in range(9)}
    b = np.vectorize(mapping.get)(b)
    for i in range(0, 9, 3):
        p = np.random.permutation(3)
        b[i:i + 3] = b[i:i + 3][p]
    for i in range(0, 9, 3):
        p = np.random.permutation(3)
        b[:, i:i + 3] = b[:, i:i + 3][:, p]
    p = np.random.permutation(3)
    b = np.vstack([
        b[p[0] * 3:p[0] * 3 + 3],
        b[p[1] * 3:p[1] * 3 + 3],
        b[p[2] * 3:p[2] * 3 + 3],
    ])
    b = np.rot90(b, k=np.random.randint(0, 4))
    return b


def create_dataset(num_samples=10000, empty_cells=40):
    base = generate_base_board()
    X, Y = [], []
    print(f"Generating {num_samples} unique Sudoku boards with {empty_cells} empty cells...")
    for _ in tqdm(range(num_samples)):
        solved = shuffle_board(base)
        puzzle = solved.copy()
        mask_indices = np.random.choice(81, empty_cells, replace=False)
        puzzle.flat[mask_indices] = 0
        X.append(puzzle.flatten())
        Y.append(solved.flatten())
    return torch.tensor(np.array(X), dtype=torch.long), torch.tensor(np.array(Y), dtype=torch.long)


class SudokuPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        quarter_dim = embed_dim // 4

        self.row_embed = nn.Embedding(9, quarter_dim)
        self.col_embed = nn.Embedding(9, quarter_dim)
        self.box_embed = nn.Embedding(9, quarter_dim)

        remaining = embed_dim - quarter_dim * 3
        pos = torch.arange(81).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, remaining, 2) * -(np.log(10000.0) / remaining))
        pe = torch.zeros(1, 81, remaining)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("sin_pe", pe)

        rows = torch.arange(9).view(9, 1).expand(9, 9).flatten()
        cols = torch.arange(9).view(1, 9).expand(9, 9).flatten()
        boxes = (rows // 3) * 3 + (cols // 3)
        self.register_buffer("row_ids", rows)
        self.register_buffer("col_ids", cols)
        self.register_buffer("box_ids", boxes)

    def forward(self, batch_size, device):
        row = self.row_embed(self.row_ids.to(device)).unsqueeze(0).expand(batch_size, -1, -1)
        col = self.col_embed(self.col_ids.to(device)).unsqueeze(0).expand(batch_size, -1, -1)
        box = self.box_embed(self.box_ids.to(device)).unsqueeze(0).expand(batch_size, -1, -1)
        sin = self.sin_pe.to(device).expand(batch_size, -1, -1)
        return torch.cat([row, col, box, sin], dim=-1)


class TinyRecursiveSudoku(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=192, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = SudokuPositionalEncoding(embed_dim)
        self.input_proj = nn.Linear(embed_dim * 3, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=6,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, steps=8, noise_scale=0.0):
        batch_size, seq_len = x.shape
        pos = self.pos_encoding(batch_size, x.device)
        x_emb = self.embedding(x) + pos
        y_emb = self.embedding(x) + pos
        z = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        step_predictions = []
        for _ in range(steps):
            if noise_scale > 0.0 and self.training is False:
                z = z + torch.randn_like(z) * noise_scale
            combined = torch.cat([x_emb, y_emb, z], dim=-1)
            z_in = F.gelu(self.input_proj(combined))
            z = self.transformer(z_in)
            z = self.layer_norm(z)
            logits = self.output_head(z)
            step_predictions.append(logits)
            probs = F.softmax(logits, dim=-1)
            y_emb = torch.matmul(probs, self.embedding.weight) + pos
        return step_predictions


def compute_mode(tensor, dim=1):
    one_hot = F.one_hot(tensor, num_classes=10)
    counts = one_hot.sum(dim=dim)
    return counts.argmax(dim=-1), None


EPOCHS = 60
RECURSIVE_STEPS = 8
CURRICULUM_START = 20
CURRICULUM_END = 50
BATCH_SIZE = 128
TRAIN_SAMPLES = 50000
NUM_WORKERS = 0

val_x, val_y = create_dataset(2000, empty_cells=45)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

train_losses, val_losses = [], []
lr_hist = []

model = TinyRecursiveSudoku().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)
criterion = nn.CrossEntropyLoss()

use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=use_amp)

start_epoch = 0
checkpoint_path = "trm_checkpoint.pt"
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"]
    train_losses = ckpt["train_losses"]
    val_losses = ckpt["val_losses"]
    lr_hist = ckpt["lr_hist"]
    print(f"Resumed at epoch {start_epoch}/{EPOCHS}")

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(start_epoch, EPOCHS):
    curriculum_empty = int(CURRICULUM_START + (CURRICULUM_END - CURRICULUM_START) * (epoch / (EPOCHS - 1)))
    train_x, train_y = create_dataset(TRAIN_SAMPLES, empty_cells=curriculum_empty)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
            step_outputs = model(x_batch, steps=RECURSIVE_STEPS)
            loss = 0
            for logits in step_outputs:
                loss += criterion(logits.view(-1, 10), y_batch.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                outputs = model(x_batch, steps=RECURSIVE_STEPS)
                final_logits = outputs[-1]
                val_loss += criterion(final_logits.view(-1, 10), y_batch.view(-1)).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    lr_hist.append(current_lr)

    print(f"Epoch {epoch+1}/{EPOCHS} | Empty: {curriculum_empty} | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lr_hist": lr_hist,
    }, checkpoint_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train Loss (All Steps)")
axes[0].plot(val_losses, label="Val Loss (Final Step)")
axes[0].set_title("TRM Training Curve (Deep Supervision)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

ax2 = axes[0].twinx()
ax2.plot(lr_hist, "g--", alpha=0.5, label="Learning Rate")
ax2.set_ylabel("LR", color="g")
ax2.tick_params(axis="y", labelcolor="g")

axes[1].plot(train_losses, label="Train Loss (All Steps)")
axes[1].plot(val_losses, label="Val Loss (Final Step)")
axes[1].set_title("Loss (Zoomed)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(bottom=min(val_losses) - 0.05)

plt.tight_layout()
plt.savefig("trm_training_curve.png")
plt.close()


def evaluate_accuracy(model, loader, steps=RECURSIVE_STEPS, noise=0.0, parallel_votes=1):
    model.eval()
    correct_boards = 0
    total_boards = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            if parallel_votes > 1:
                B, L = x_batch.shape
                x_expanded = x_batch.repeat_interleave(parallel_votes, dim=0)
                outputs = model(x_expanded, steps=steps, noise_scale=noise)
                final_logits = outputs[-1]
                preds = torch.argmax(final_logits, dim=-1)
                preds = preds.view(B, parallel_votes, L)
                final_preds, _ = compute_mode(preds, dim=1)
            else:
                outputs = model(x_batch, steps=steps, noise_scale=noise)
                final_preds = torch.argmax(outputs[-1], dim=-1)
            board_matches = (final_preds == y_batch).all(dim=1)
            correct_boards += board_matches.sum().item()
            total_boards += x_batch.size(0)
    return (correct_boards / total_boards) * 100


print("\nRunning Deterministic TRM Evaluation...")
trm_acc = evaluate_accuracy(model, val_loader, steps=RECURSIVE_STEPS, noise=0.0, parallel_votes=1)
print(f"Standard TRM Accuracy (Perfect Boards): {trm_acc:.2f}%\n")

print("Running Stochastic PTRM Evaluation (Injecting latent noise & voting)...")
ptrm_acc = evaluate_accuracy(model, val_loader, steps=RECURSIVE_STEPS, noise=0.1, parallel_votes=5)
print(f"PTRM Accuracy (Perfect Boards): {ptrm_acc:.2f}%")

metrics = {
    "trm_accuracy": trm_acc,
    "ptrm_accuracy": ptrm_acc,
    "epochs_trained": EPOCHS,
    "model_dim": model.embed_dim,
    "num_layers": model.transformer.num_layers,
    "recursive_steps": RECURSIVE_STEPS,
}
with open("evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

model_path = "trm_sudoku_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model successfully saved to: {os.path.abspath(model_path)}")
print("Files generated: trm_training_curve.png, evaluation_metrics.json, trm_sudoku_model.pth")

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("Checkpoint cleaned up.")
