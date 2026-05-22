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
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
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
    print(f"Generating {num_samples} unique Sudoku boards...")
    for _ in tqdm(range(num_samples)):
        solved = shuffle_board(base)
        puzzle = solved.copy()
        mask_indices = np.random.choice(81, empty_cells, replace=False)
        puzzle.flat[mask_indices] = 0
        X.append(puzzle.flatten())
        Y.append(solved.flatten())
    return torch.tensor(np.array(X), dtype=torch.long), torch.tensor(np.array(Y), dtype=torch.long)

train_x, train_y = create_dataset(20000, empty_cells=45)
val_x, val_y = create_dataset(2000, empty_cells=45)

class SudokuDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

BATCH_SIZE = 128
train_loader = DataLoader(SudokuDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SudokuDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

class TinyRecursiveSudoku(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=128, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_proj = nn.Linear(embed_dim * 3, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, steps=6, noise_scale=0.0):
        batch_size, seq_len = x.shape
        x_emb = self.embedding(x)
        y_emb = self.embedding(x)
        z = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        step_predictions = []
        for _ in range(steps):
            if noise_scale > 0.0 and self.training is False:
                z = z + torch.randn_like(z) * noise_scale
            combined = torch.cat([x_emb, y_emb, z], dim=-1)
            z_in = F.gelu(self.input_proj(combined))
            z = self.transformer(z_in)
            logits = self.output_head(z)
            step_predictions.append(logits)
            probs = F.softmax(logits, dim=-1)
            y_emb = torch.matmul(probs, self.embedding.weight)
        return step_predictions

model = TinyRecursiveSudoku().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 15
RECURSIVE_STEPS = 6
train_losses, val_losses = [], []

print("Starting Deep Supervision Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        step_outputs = model(x_batch, steps=RECURSIVE_STEPS)
        loss = 0
        for logits in step_outputs:
            loss += criterion(logits.view(-1, 10), y_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(x_batch, steps=RECURSIVE_STEPS)
            final_logits = outputs[-1]
            val_loss += criterion(final_logits.view(-1, 10), y_batch.view(-1)).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss (Summed): {avg_train_loss:.4f} | Val Loss (Final Step): {avg_val_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss (All Steps)')
plt.plot(val_losses, label='Val Loss (Final Step)')
plt.title('TRM Training Curve (Deep Supervision)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('trm_training_curve.png')
plt.close()

def evaluate_accuracy(model, loader, steps=6, noise=0.0, parallel_votes=1):
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
                final_preds, _ = torch.mode(preds, dim=1)
            else:
                outputs = model(x_batch, steps=steps, noise_scale=noise)
                final_preds = torch.argmax(outputs[-1], dim=-1)
            board_matches = (final_preds == y_batch).all(dim=1)
            correct_boards += board_matches.sum().item()
            total_boards += x_batch.size(0)
    return (correct_boards / total_boards) * 100

print("Running Deterministic TRM Evaluation...")
trm_acc = evaluate_accuracy(model, val_loader, steps=6, noise=0.0, parallel_votes=1)
print(f"Standard TRM Accuracy (Perfect Boards): {trm_acc:.2f}%\n")

print("Running Stochastic PTRM Evaluation (Injecting latent noise & voting)...")
ptrm_acc = evaluate_accuracy(model, val_loader, steps=6, noise=0.1, parallel_votes=5)
print(f"PTRM Accuracy (Perfect Boards): {ptrm_acc:.2f}%")

metrics = {
    "trm_accuracy": trm_acc,
    "ptrm_accuracy": ptrm_acc,
    "epochs_trained": EPOCHS
}
with open('evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

model_path = "trm_sudoku_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model successfully saved to: {os.path.abspath(model_path)}")
print("Files generated in directory: trm_training_curve.png, evaluation_metrics.json, trm_sudoku_model.pth")
