"""Board / move encoding for behavioral cloning.

Everything is expressed from the **side-to-move's perspective** ("my" pieces always
occupy planes 0-5 and move *up* the board). When it is Black to move we mirror the
board vertically so the network sees one consistent point of view regardless of the
colour you actually played. This is what lets a single policy capture *your* style.

Board tensor : (12, 8, 8) float32 -- 6 piece types x {mine, theirs}.
Move label   : an int in [0, 4096) encoding (from_square, to_square) in the
               canonical frame: index = from * 64 + to. Promotions are collapsed to
               the from->to pair (queen is assumed at inference).
"""

import chess
import numpy as np

NUM_MOVES = 64 * 64  # 4096
BOARD_PLANES = 12

# piece_type (1..6) -> plane offset (0..5)
_PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def _canonical_square(square, turn):
    """Map a real square into the side-to-move frame (vertical mirror for Black)."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def board_to_tensor(board):
    """Return a (12, 8, 8) float32 tensor from the side-to-move's perspective."""
    tensor = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)
    turn = board.turn
    for square, piece in board.piece_map().items():
        csq = _canonical_square(square, turn)
        rank, file = divmod(csq, 8)
        is_mine = piece.color == turn
        plane = _PIECE_TO_PLANE[piece.piece_type] + (0 if is_mine else 6)
        tensor[plane, rank, file] = 1.0
    return tensor


def move_to_index(move, board):
    """Encode a move as from*64 + to in the canonical frame."""
    turn = board.turn
    cf = _canonical_square(move.from_square, turn)
    ct = _canonical_square(move.to_square, turn)
    return cf * 64 + ct


def is_underpromotion(move):
    """True for knight/bishop/rook promotions (collapsed away by our encoding)."""
    return move.promotion is not None and move.promotion != chess.QUEEN


def legal_move_index_map(board):
    """Map {canonical_index: chess.Move} for the current legal moves.

    When several legal moves share a from->to pair (only under-promotions), the
    queen promotion wins so inference never emits an under-promotion.
    """
    mapping = {}
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        if idx in mapping and is_underpromotion(move):
            continue  # keep the already-stored (queen / non-promo) move
        mapping[idx] = move
    return mapping
