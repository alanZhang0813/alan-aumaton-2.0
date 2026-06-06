# alan-aumaton 2.0

A chess bot that **plays like you**. It learns from your Chess.com game history by
*behavioral cloning*: a policy network is trained to predict the move *you* would make
in a given position (your turns only), so it reproduces your style — openings, habits,
and blunders included.

## How it works

| Stage | File | What it does |
|-------|------|--------------|
| Fetch | `chesscom.py` | Pulls every game from the public Chess.com API, caches PGNs to `data/`. |
| Encode | `encoding.py` | Board → `(12,8,8)` tensor and move → `(from,to)` index, both **canonicalized to the side-to-move's perspective**. |
| Dataset | `dataset.py` | Emits `(position, your_move)` for every position where it was **your** turn. |
| Model | `model.py` | A small residual CNN policy network (4096-way move classifier). |
| Train | `train.py` | Cross-entropy training; reports val top-1/top-3 move accuracy. |
| Play | `engine.py` | Loads the model and picks the move you'd most likely make (legal-masked). |
| UCI | `uci.py` | Exposes the engine over the UCI protocol for any chess GUI / lichess-bot. |

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

```powershell
.\.venv\Scripts\python.exe main.py --user AlanZhang
# options: --epochs 30  --batch-size 256  --lr 1e-3  --refresh
```

This fetches your games (cached after the first run), builds the dataset, trains, and
saves `models/policy_<user>.pt`.

## Play against your clone

Inspect what it would play from the start position:

```powershell
.\.venv\Scripts\python.exe engine.py AlanZhang
```

Use it as a UCI engine — point CuteChess / Arena / BanksiaGUI (or lichess-bot) at:

```
.\.venv\Scripts\python.exe uci.py AlanZhang
```

The `Temperature` UCI option (0–100) adds move variety; `0` is pure imitation.

## Play from a website (HTTP API)

`server.py` exposes the clone over HTTP so an interactive board on your own site can
ask it for moves. The browser owns the game state (use [chess.js](https://github.com/jhlywa/chess.js)
for rules + a board UI like [chessboard.js](https://chessboardjs.com/) or
[chessground](https://github.com/lichess-org/chessground)); on each of your moves it
POSTs the position's FEN and applies the move that comes back.

```powershell
.\.venv\Scripts\python.exe server.py        # http://localhost:8000
# env: BOT_USER=AlanZhang  PORT=8000  BOT_ORIGINS=https://yoursite.com (CORS allowlist)
```

A minimal standalone test page lives in `web/index.html` — open it in a browser
(while the server runs) to play against your clone and confirm the API end-to-end.

### Endpoint contract (for integrating into your existing site)

`POST /move`
```jsonc
// request
{ "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "temperature": 0 }            // 0 = pure imitation; >0 samples for variety

// response
{ "move": "e2e4",               // UCI; apply with chess.js {from,to,promotion}
  "san": "e4",
  "fen": "...",                 // position after the clone's move
  "is_game_over": false,
  "result": null,               // "1-0" | "0-1" | "1/2-1/2" when over
  "top_moves": [ {"uci":"e2e4","san":"e4","prob":0.962}, ... ] }
```

`GET /health` → `{ "status": "ok", "username": "...", "model_loaded": true|false }`.
Before training has saved a checkpoint, `/move` returns **503** until the model exists.

Frontend move flow: your UI validates the human move with chess.js → `POST /move`
with `game.fen()` → apply `response.move` via
`game.move({from: m.slice(0,2), to: m.slice(2,4), promotion: m.slice(4,5) || 'q'})`.
Deploy `server.py` on any host (Render / Railway / Fly.io / a VPS); set `BOT_ORIGINS`
to your site's domain to lock down CORS.
