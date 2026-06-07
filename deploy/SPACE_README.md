---
title: Alan Aumaton Chess
emoji: ♟️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# alan-aumaton — move API

Serves moves from a chess policy network trained to imitate a specific player.
Backend for an interactive board hosted elsewhere (e.g. GitHub Pages).

- `GET  /health` → `{status, username, model_loaded}`
- `POST /move` `{fen, temperature}` → `{move, san, fen, is_game_over, result, top_moves}`

Built with FastAPI + PyTorch (CPU). See the project repo for training code.
