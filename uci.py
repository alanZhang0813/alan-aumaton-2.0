"""Minimal UCI engine wrapping the trained policy.

Implements just enough of the Universal Chess Interface to load into a GUI
(CuteChess, Arena, BanksiaGUI) or drive with lichess-bot:

    uci / isready / ucinewgame / position [startpos|fen ...] [moves ...] / go / quit

Usage:
    python uci.py [username]            # defaults to AlanZhang
A "Temperature" UCI spin option (0..100, as percent) controls move sampling;
0 = pure imitation (argmax).
"""

import sys

import chess

from engine import ChessEngine


def _apply_position(board, tokens):
    """Set up `board` from a UCI `position` command's tokens (after 'position')."""
    if not tokens:
        return
    if tokens[0] == "startpos":
        board.reset()
        moves_start = tokens.index("moves") + 1 if "moves" in tokens else len(tokens)
    elif tokens[0] == "fen":
        fen = " ".join(tokens[1:7])
        board.set_fen(fen)
        moves_start = tokens.index("moves") + 1 if "moves" in tokens else len(tokens)
    else:
        return
    for uci_move in tokens[moves_start:]:
        board.push(chess.Move.from_uci(uci_move))


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else "AlanZhang"
    engine = None  # lazy-load on first use so `uci`/`isready` stay instant
    board = chess.Board()
    temperature = 0.0

    def out(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            out(f"id name alan-aumaton ({username})")
            out("id author alan-aumaton")
            out("option name Temperature type spin default 0 min 0 max 100")
            out("uciok")
        elif cmd == "isready":
            if engine is None:
                engine = ChessEngine(username)
            out("readyok")
        elif cmd == "setoption":
            # setoption name Temperature value 30
            if "Temperature" in tokens and "value" in tokens:
                temperature = int(tokens[tokens.index("value") + 1]) / 100.0
        elif cmd == "ucinewgame":
            board.reset()
        elif cmd == "position":
            _apply_position(board, tokens[1:])
        elif cmd == "go":
            if engine is None:
                engine = ChessEngine(username)
            move = engine.select_move(board, temperature=temperature)
            out(f"bestmove {move.uci() if move else '0000'}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
