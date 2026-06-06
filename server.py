"""HTTP API that serves moves from your trained clone.

A thin FastAPI wrapper around `ChessEngine`: the browser keeps the game state and
asks this server what move *you* would play next. All the chess logic (encoding,
canonicalization, legal-move masking) is reused from engine.py / encoding.py.

Run:
    .\.venv\Scripts\python.exe server.py            # serves AlanZhang on :8000
    set BOT_USER=SomeUser & .\.venv\Scripts\python.exe server.py
    # or: uvicorn server:app --reload --port 8000

Endpoints:
    GET  /health            -> {status, model_loaded, username}
    POST /move  {fen, temperature?}
                            -> {move, san, fen, is_game_over, result, top_moves}
"""

import os

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import ChessEngine

USERNAME = os.environ.get("BOT_USER", "AlanZhang")

app = FastAPI(title=f"alan-aumaton ({USERNAME})")

# Allow your separate website to call this API from the browser. Lock this down to
# your real domain(s) in production via the BOT_ORIGINS env var (comma-separated).
_origins = os.environ.get("BOT_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None  # lazy-loaded so the server boots even before training finishes


def get_engine():
    global _engine
    if _engine is None:
        try:
            _engine = ChessEngine(USERNAME)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _engine


class MoveRequest(BaseModel):
    fen: str
    temperature: float = 0.0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "username": USERNAME,
        "model_loaded": _engine is not None,
    }


@app.post("/move")
def move(req: MoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {req.fen!r}")

    if board.is_game_over():
        return {
            "move": None,
            "san": None,
            "fen": board.fen(),
            "is_game_over": True,
            "result": board.result(),
            "top_moves": [],
        }

    engine = get_engine()
    chosen = engine.select_move(board, temperature=req.temperature)
    top = [
        {"uci": m.uci(), "san": board.san(m), "prob": round(p, 4)}
        for m, p in engine.top_moves(board, k=5)
    ]
    san = board.san(chosen)
    board.push(chosen)

    return {
        "move": chosen.uci(),
        "san": san,
        "fen": board.fen(),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "top_moves": top,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
