# RLM-Sudoku

A recursive language model approach for solving Sudoku puzzles using a Transformer with deep supervision and curriculum learning.

## Architecture

The model (`TinyRecursiveSudoku`) uses a **recursive reasoning loop** inspired by the "Thinking Recursively" (TRM) paradigm:

```
Input → [X_emb | Y_emb | Z] → Project → Transformer → Output → Y_emb update → repeat
```

### Components

| Component | Details |
|-----------|---------|
| **Embedding** | `nn.Embedding(10, 192)` — maps digits 0–9 to vectors |
| **Positional Encoding** | Learned row (9), column (9), and box (9) embeddings + sinusoidal PE (81 positions) |
| **Recurrent State Z** | Zero-initialized scratchpad that carries information across steps |
| **Transformer** | 3-layer `nn.TransformerEncoder`, 6 heads, 192-dim, GELU activation, 768-dim FFN |
| **LayerNorm** | Applied after transformer before output head |
| **Output Head** | `nn.Linear(192, 10)` — predicts digit logits per cell |
| **Recursive Steps** | 8 steps with deep supervision (loss computed at every step) |

**Total parameters:** 1,450,906 (~5.6MB as float32)

### Training Strategy

- **Deep Supervision** — loss is summed across all 8 recursive steps, providing gradient signal at every refinement
- **Curriculum Learning** — empty cells ramp linearly from 20 → 50 over 60 epochs, starting easy and gradually increasing difficulty
- **ReduceLROnPlateau** — LR starts at 1e-3, halves when val loss plateaus (patience=3, min_lr=1e-6)
- **Automatic Mixed Precision** — enabled on CUDA for faster training
- **Checkpointing** — saves every epoch; interrupted runs resume automatically

## Training Results

| Run | Samples/Epoch | Epochs | Val Loss | TRM Accuracy | PTRM Accuracy |
|-----|:------------:|:------:|:--------:|:------------:|:-------------:|
| Baseline | 10k | 30 | 0.1363 | 24.45% | 24.55% |
| Tier 2 | **50k** | **60** | **0.1092** | **25.25%** | **25.50%** |

Both runs use the same 1.45M-param model; Tier 2 benefits from 5x more data per epoch and 2x more epochs.

## Files

| File | Description |
|------|-------------|
| `run_trm.py` | Headless training script — run with `python run_trm.py` |
| `trm_sudoku_notebook.ipynb` | Jupyter notebook with same pipeline split into cells |
| `sudoku_game.py` | PyGame visualizer — interactive play with Solve & Step-by-Step modes |
| `trm_sudoku_model.pth` | Trained model weights (~5.6MB) |
| `evaluation_metrics.json` | Final TRM / PTRM accuracy and model config |
| `trm_training_curve.png` | Training and validation loss curves |

## Usage

### Train
```bash
python run_trm.py
```
Checkpoints are saved automatically to `trm_checkpoint.pt`. If interrupted, re-run and it resumes from the last epoch.

### Test with PyGame
```bash
python sudoku_game.py
```
- **New Puzzle** — generates a random board with 45 empty cells  
- **Solve** — runs the model and shows solved board (green = correct, red = wrong)  
- **Step-by-Step** — runs the model and advances through each of the 8 recursive steps  
- Click a cell and type **1–9** to edit, **Backspace / 0** to clear  

### Run in Jupyter
Open `trm_sudoku_notebook.ipynb` and run all cells.

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- NumPy, Matplotlib, tqdm
- Pygame (for visualizer only)

Install with:
```bash
pip install torch numpy matplotlib tqdm pygame
```

## System Compatibility

- **Training:** GPU with 8GB+ VRAM recommended. MPS (Apple Silicon) and CPU supported.
- **Inference:** Runs on any device with PyTorch — no GPU needed for inference.
- **Tested on:** M2 MacBook Air (16GB), Tesla P100 (16GB)
