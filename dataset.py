"""Turn cached PGNs into a behavioral-cloning dataset.

For every game we figure out which colour *you* played (by matching your username
against the PGN White/Black headers) and emit one training example
``(board_tensor, move_index)`` for each position where it was **your** turn -- i.e.
the move *you* actually chose. Positions on the opponent's move are ignored: we are
modelling your decisions, not theirs.

Examples are cached to ``data/dataset_<user>.npz`` so training can iterate quickly.
"""

import io
import os

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

from encoding import board_to_tensor, is_underpromotion, move_to_index

DATA_DIR = "data"


def _user_color(headers, username):
    """Return chess.WHITE / chess.BLACK if username played, else None."""
    user = username.lower()
    if headers.get("White", "").lower() == user:
        return chess.WHITE
    if headers.get("Black", "").lower() == user:
        return chess.BLACK
    return None


def examples_from_pgn(pgn_text, username):
    """Yield (tensor, move_index) for each of the user's moves in one game."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return

    color = _user_color(game.headers, username)
    if color is None:
        return

    board = game.board()
    for move in game.mainline_moves():
        if board.turn == color and not is_underpromotion(move):
            yield board_to_tensor(board), move_to_index(move, board)
        board.push(move)


def build_dataset(games_pgn, username, use_cache=True):
    """Build (X, y) arrays of the user's moves, with on-disk caching."""
    cache_path = os.path.join(DATA_DIR, f"dataset_{username.lower()}.npz")
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"Loaded {len(data['y'])} cached examples from {cache_path}")
        return data["X"], data["y"]

    xs, ys = [], []
    for pgn_text in tqdm(games_pgn, desc="Parsing games"):
        for tensor, idx in examples_from_pgn(pgn_text, username):
            xs.append(tensor)
            ys.append(idx)

    if not xs:
        raise RuntimeError(
            f"No examples produced for '{username}'. Check the username matches "
            "the White/Black PGN headers."
        )

    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"Built {len(y)} examples -> {cache_path}")
    return X, y
