"""Move selection from a trained policy -- pure imitation.

Given a board, the engine scores only the **legal** moves with the policy network and
plays the one you'd most likely make (argmax). A temperature > 0 samples instead,
which is handy if you want a little variety while staying in-character.
"""

import os

import chess
import numpy as np
import torch

from encoding import board_to_tensor, legal_move_index_map
from model import PolicyNet

MODELS_DIR = "models"


class ChessEngine:
    def __init__(self, username, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNet().to(self.device)
        model_path = os.path.join(MODELS_DIR, f"policy_{username.lower()}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run training first."
            )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def _legal_logits(self, board):
        """Return (moves, logits) aligned over the current legal moves."""
        index_map = legal_move_index_map(board)
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)[0].cpu().numpy()
        moves = list(index_map.values())
        scores = np.array([logits[idx] for idx in index_map])
        return moves, scores

    def select_move(self, board, temperature=0.0):
        """Pick a legal move. temperature=0 -> argmax (most like you)."""
        if board.is_game_over():
            return None
        moves, scores = self._legal_logits(board)
        if temperature <= 0:
            return moves[int(np.argmax(scores))]
        # Softmax sampling with temperature.
        logits = scores / temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return moves[int(np.random.choice(len(moves), p=probs))]

    def top_moves(self, board, k=5):
        """Return [(move, probability), ...] sorted high-to-low -- for inspection."""
        moves, scores = self._legal_logits(board)
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        order = np.argsort(probs)[::-1][:k]
        return [(moves[i], float(probs[i])) for i in order]


if __name__ == "__main__":
    import sys

    user = sys.argv[1] if len(sys.argv) > 1 else "AlanZhang"
    engine = ChessEngine(user)
    board = chess.Board()
    print(f"Top predicted moves for {user} from the start position:")
    for move, prob in engine.top_moves(board, k=8):
        print(f"  {board.san(move):6s} {prob:.1%}")
