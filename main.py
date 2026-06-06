"""End-to-end pipeline: fetch your Chess.com games, build a behavioral-cloning
dataset of *your* moves, and train a policy network that plays like you.

    python main.py --user AlanZhang
    python main.py --user AlanZhang --epochs 30 --refresh

After training, play against the clone by loading `uci.py` as an engine in a chess
GUI, or inspect it with `python engine.py <user>`.
"""

import argparse

from chesscom import ChessComBot
from dataset import build_dataset
from train import train


def parse_args():
    p = argparse.ArgumentParser(description="Train a chess bot that plays like you.")
    p.add_argument("--user", default="AlanZhang", help="Chess.com username")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore caches and re-fetch games / rebuild the dataset",
    )
    return p.parse_args()


def main():
    args = parse_args()

    bot = ChessComBot(args.user)
    games = bot.load_player_data(use_cache=not args.refresh)
    print(f"Profile: {bot.name} ({bot.title or 'no title'}) | {len(games)} games")

    X, y = build_dataset(games, args.user, use_cache=not args.refresh)

    train(
        X,
        y,
        args.user,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
