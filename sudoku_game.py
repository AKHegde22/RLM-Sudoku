import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    DEVICE = torch.device("cpu")


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
            d_model=embed_dim, nhead=6, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu"
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


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trm_sudoku_model.pth")

CELL_SIZE = 60
GRID_SIZE = CELL_SIZE * 9
PADDING = 40
BUTTON_HEIGHT = 40
BTN_GAP = 10
WINDOW_WIDTH = GRID_SIZE + PADDING * 2
WINDOW_HEIGHT = GRID_SIZE + PADDING * 2 + BUTTON_HEIGHT + 40

BG_COLOR = (245, 245, 240)
GRID_COLOR = (40, 40, 40)
CELL_COLOR = (255, 255, 255)
CELL_ACTIVE = (220, 235, 255)
FIXED_COLOR = (120, 120, 120)
USER_COLOR = (60, 60, 255)
SOLVED_CORRECT = (40, 180, 40)
SOLVED_WRONG = (255, 80, 80)
BTN_COLOR = (70, 130, 180)
BTN_HOVER = (100, 160, 210)
BTN_TEXT = (255, 255, 255)
STATUS_COLOR = (100, 100, 100)
ACCENT_GREEN = (200, 240, 200)

pygame.init()
font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 16)
tiny_font = pygame.font.SysFont("Arial", 13)


def generate_puzzle(empty_cells=45):
    base = np.array([
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
    board = base.copy()
    nums = np.arange(1, 10)
    np.random.shuffle(nums)
    mapping = {i + 1: nums[i] for i in range(9)}
    board = np.vectorize(mapping.get)(board)
    for i in range(0, 9, 3):
        p = np.random.permutation(3)
        board[i:i + 3] = board[i:i + 3][p]
    for i in range(0, 9, 3):
        p = np.random.permutation(3)
        board[:, i:i + 3] = board[:, i:i + 3][:, p]
    p = np.random.permutation(3)
    board = np.vstack([
        board[p[0] * 3:p[0] * 3 + 3],
        board[p[1] * 3:p[1] * 3 + 3],
        board[p[2] * 3:p[2] * 3 + 3],
    ])
    board = np.rot90(board, k=np.random.randint(0, 4))
    solved = board.copy()
    mask_indices = np.random.choice(81, empty_cells, replace=False)
    board.flat[mask_indices] = 0
    return board, solved


class SudokuGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Recursive Sudoku Solver (TRM)")

        self.board = np.zeros((9, 9), dtype=int)
        self.solved_board = np.zeros((9, 9), dtype=int)
        self.fixed_mask = np.zeros((9, 9), dtype=bool)
        self.predicted = np.zeros((9, 9), dtype=bool)
        self.correct = np.ones((9, 9), dtype=bool)
        self.selected = None
        self.status_text = "Loading model..."
        self.model_loaded = False
        self.model = None

        self.buttons = {}
        x = PADDING
        for label, text in [("new", "New Puzzle"), ("solve", "Solve"), ("step", "Step-by-Step")]:
            self.buttons[label] = (pygame.Rect(x, WINDOW_HEIGHT - PADDING - BUTTON_HEIGHT, 130, BUTTON_HEIGHT), text)
            x += 140

        self.step_mode = False
        self.current_step = 0
        self.max_steps = 0
        self.step_boards = []

        self.clock = pygame.time.Clock()

        self.load_model()
        self.new_puzzle()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.status_text = "No model found. Train first! (trm_sudoku_model.pth missing)"
            return
        try:
            self.model = TinyRecursiveSudoku().to(DEVICE)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
            self.model.eval()
            self.model_loaded = True
            self.status_text = "Model loaded. Press 'New' for a puzzle."
        except Exception as e:
            self.status_text = f"Model load error: {e}"

    def new_puzzle(self):
        puzzle, solution = generate_puzzle(45)
        self.board = puzzle.copy()
        self.solved_board = solution.copy()
        self.fixed_mask = puzzle != 0
        self.predicted = np.zeros((9, 9), dtype=bool)
        self.correct = np.ones((9, 9), dtype=bool)
        self.selected = None
        self.step_mode = False
        self.current_step = 0
        self.max_steps = 0
        self.step_boards = []
        self.status_text = "New puzzle. Press 'Solve' or 'Step-by-Step'."

    def run_model(self, step_mode=False):
        if not self.model_loaded:
            self.status_text = "Model not loaded!"
            return

        x = torch.tensor(self.board.flatten(), dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            all_steps = self.model(x, steps=8)

        self.step_boards = []
        for step_idx, logits in enumerate(all_steps):
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
            self.step_boards.append(preds.copy())

        final_preds = self.step_boards[-1]
        self.predicted = np.zeros((9, 9), dtype=bool)
        self.correct = np.ones((9, 9), dtype=bool)
        for i in range(9):
            for j in range(9):
                if not self.fixed_mask[i, j]:
                    self.predicted[i, j] = True
                    self.board[i, j] = int(final_preds[i * 9 + j])
                    self.correct[i, j] = (self.board[i, j] == self.solved_board[i, j])

        correct_cells = self.correct[self.predicted].sum()
        total_empty = self.predicted.sum()
        perfect = correct_cells == total_empty
        self.status_text = f"Solved: {correct_cells}/{total_empty} cells correct" + (
            "  --  PERFECT BOARD!" if perfect else ""
        )

        if step_mode:
            self.step_mode = True
            self.current_step = 0
            self.max_steps = len(self.step_boards)
            self.status_text = f"Step {self.current_step + 1}/{self.max_steps}. Press 'Step-by-Step' to advance."

    def apply_step(self):
        if not self.step_mode or self.current_step >= self.max_steps - 1:
            self.step_mode = False
            return
        self.current_step += 1
        step_preds = self.step_boards[self.current_step]
        for i in range(9):
            for j in range(9):
                if not self.fixed_mask[i, j]:
                    self.board[i, j] = int(step_preds[i * 9 + j])
                    self.correct[i, j] = (self.board[i, j] == self.solved_board[i, j])
        correct_cells = self.correct[self.predicted].sum()
        total_empty = self.predicted.sum()
        self.status_text = f"Step {self.current_step + 1}/{self.max_steps} | {correct_cells}/{total_empty} correct"

    def draw_grid(self):
        for i in range(9):
            for j in range(9):
                x = PADDING + j * CELL_SIZE
                y = PADDING + i * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                if self.selected == (i, j):
                    color = CELL_ACTIVE
                elif self.fixed_mask[i, j]:
                    color = (235, 235, 235)
                elif self.step_mode and self.predicted[i, j]:
                    color = ACCENT_GREEN if self.correct[i, j] else (255, 220, 220)
                else:
                    color = CELL_COLOR
                pygame.draw.rect(self.screen, color, rect)

                if self.board[i, j] != 0:
                    if self.fixed_mask[i, j]:
                        color = FIXED_COLOR
                    elif self.step_mode or self.predicted[i, j]:
                        color = SOLVED_CORRECT if self.correct[i, j] else SOLVED_WRONG
                    else:
                        color = USER_COLOR
                    text = font.render(str(self.board[i, j]), True, color)
                    text_rect = text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    self.screen.blit(text, text_rect)

        for i in range(0, 10):
            thickness = 3 if i % 3 == 0 else 1
            pygame.draw.line(self.screen, GRID_COLOR,
                             (PADDING + i * CELL_SIZE, PADDING),
                             (PADDING + i * CELL_SIZE, PADDING + GRID_SIZE), thickness)
            pygame.draw.line(self.screen, GRID_COLOR,
                             (PADDING, PADDING + i * CELL_SIZE),
                             (PADDING + GRID_SIZE, PADDING + i * CELL_SIZE), thickness)

    def draw_buttons(self):
        mouse = pygame.mouse.get_pos()
        for _, (rect, text) in self.buttons.items():
            color = BTN_HOVER if rect.collidepoint(mouse) else BTN_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            rendered = small_font.render(text, True, BTN_TEXT)
            self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def draw_status(self):
        text = small_font.render(self.status_text, True, STATUS_COLOR)
        self.screen.blit(text, (PADDING, WINDOW_HEIGHT - PADDING - BUTTON_HEIGHT - 18))

    def handle_key(self, key):
        if self.selected is None:
            return
        i, j = self.selected
        if self.fixed_mask[i, j]:
            return
        if key == pygame.K_BACKSPACE or key == pygame.K_DELETE:
            self.board[i, j] = 0
        elif pygame.K_1 <= key <= pygame.K_9:
            self.board[i, j] = key - pygame.K_0
        elif key == pygame.K_0:
            self.board[i, j] = 0

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    gx = (mx - PADDING) // CELL_SIZE
                    gy = (my - PADDING) // CELL_SIZE
                    if 0 <= gx < 9 and 0 <= gy < 9:
                        if not self.fixed_mask[gy, gx]:
                            self.selected = (gy, gx)
                            self.step_mode = False
                    for label, (rect, _) in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            if label == "new":
                                self.new_puzzle()
                            elif label == "solve":
                                self.step_mode = False
                                self.run_model(step_mode=False)
                            elif label == "step":
                                if not self.step_mode and self.model_loaded:
                                    self.run_model(step_mode=True)
                                elif self.step_mode:
                                    self.apply_step()
                            break
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)

            self.screen.fill(BG_COLOR)
            self.draw_grid()
            self.draw_buttons()
            self.draw_status()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    game = SudokuGame()
    game.run()
