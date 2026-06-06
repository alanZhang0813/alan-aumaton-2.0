"""Fetch a player's games from the public Chess.com API, with on-disk caching.

The Chess.com API exposes games per player per month ("archives"). We pull every
archive in parallel, keep only the PGN of standard-chess games, and cache the raw
PGNs to ``data/games_<user>.json`` so subsequent runs never re-hit the network.
"""

import concurrent.futures
import json
import os
import time

import requests

DATA_DIR = "data"


class ChessComBot:
    def __init__(self, username, max_workers=8, user_agent=None):
        self.username = username
        self.name = None
        self.title = None
        self.archived_months = []
        self.games_pgn = []
        self.unparseable_games = set()
        self.max_workers = max_workers
        self.session = requests.Session()
        self.headers = {
            "User-Agent": user_agent
            or "alan-aumaton/2.0 (chess style-cloning project)"
        }

    # ----- public API -------------------------------------------------------

    def load_player_data(self, use_cache=True):
        """Populate ``self.games_pgn``. Uses the on-disk cache when available."""
        cache_path = self._cache_path()
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            self.games_pgn = cached.get("games_pgn", [])
            self.unparseable_games = set(cached.get("unparseable_games", []))
            print(f"Loaded {len(self.games_pgn)} cached games from {cache_path}")
            return self.games_pgn

        start_time = time.time()
        self.get_profile_info()
        self.get_archives_list()
        if self.archived_months:
            print(f"Loading games from {len(self.archived_months)} months...")
            self.process_game_archives()

        elapsed = time.time() - start_time
        print(f"Fetched {len(self.games_pgn)} games in {elapsed:.1f}s")
        if self.unparseable_games:
            print(f"Skipped non-standard game types: {self.unparseable_games}")

        self._save_cache(cache_path)
        return self.games_pgn

    # ----- network helpers --------------------------------------------------

    def get_profile_info(self):
        url = f"https://api.chess.com/pub/player/{self.username}"
        response = self.session.get(url, headers=self.headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch profile for {self.username} "
                f"(status {response.status_code})"
            )
        data = response.json()
        self.name = data.get("name")
        self.title = data.get("title")

    def get_archives_list(self):
        url = f"https://api.chess.com/pub/player/{self.username}/games/archives"
        response = self.session.get(url, headers=self.headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch archives for {self.username} "
                f"(status {response.status_code})"
            )
        self.archived_months = response.json().get("archives", [])

    def get_month_games(self, month_url):
        try:
            response = self.session.get(month_url, headers=self.headers)
            if response.status_code == 200:
                return response.json().get("games", [])
            print(f"Error retrieving {month_url}: {response.status_code}")
        except requests.RequestException as e:
            print(f"Exception retrieving {month_url}: {e}")
        return []

    def process_game_archives(self):
        """Fetch every monthly archive in parallel and collect PGNs."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(self.get_month_games, month): month
                for month in self.archived_months
            }
            for future in concurrent.futures.as_completed(futures):
                month = futures[future]
                try:
                    self.process_games(future.result())
                except Exception as e:  # noqa: BLE001 - log and keep going
                    print(f"Error processing {month}: {e}")

    def process_games(self, games):
        for game in games:
            # Only standard chess; variants (e.g. chess960, bughouse) are skipped.
            if game.get("rules") != "chess":
                self.unparseable_games.add(game.get("rules", "unknown"))
                continue
            if "pgn" in game:
                self.games_pgn.append(game["pgn"])

    # ----- caching ----------------------------------------------------------

    def _cache_path(self):
        return os.path.join(DATA_DIR, f"games_{self.username.lower()}.json")

    def _save_cache(self, cache_path):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "games_pgn": self.games_pgn,
                    "unparseable_games": sorted(self.unparseable_games),
                },
                f,
            )
        print(f"Cached games to {cache_path}")


if __name__ == "__main__":
    import sys

    user = sys.argv[1] if len(sys.argv) > 1 else "AlanZhang"
    bot = ChessComBot(user)
    bot.load_player_data()
    print("Name:", bot.name)
    print("Title:", bot.title)
    print("# games:", len(bot.games_pgn))
