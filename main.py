import requests
import chess
import chess.pgn
import io
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import defaultdict
import pickle
import os
import concurrent.futures
import time

class ChessComBot:
    def __init__(self, username, max_workers=10):
        self.username = username
        self.name = None
        self.title = None
        self.archived_months = None
        self.games_pgn = []
        self.unparseable_games = set()
        self.user_agent = "Alan's Application for Designing a Chess Bot"
        self.max_workers = max_workers
        self.session = requests.Session()
        self.headers = {
            "User-Agent": self.user_agent
        }

        self.model = None
        self.trained = False
        self.piece_values = {
            'p': -1, 'P': 1,   # pawns
            'n': -3, 'N': 3,   # knights
            'b': -3, 'B': 3,   # bishops
            'r': -5, 'R': 5,   # rooks
            'q': -9, 'Q': 9,   # queens
            'k': 0, 'K': 0     # kings
        }

        self.load_player_data()
    
    def load_player_data(self):
        start_time = time.time()
        self.get_profile_info()
        self.get_archives_list()
        # Process months in parallel
        if self.archived_months:
            print(f"Loading games from {len(self.archived_months)} months...")
            self._process_game_archives()
        
        elapsed = time.time() - start_time
        print(f"Loaded {len(self.games_pgn)} games in {elapsed:.2f} seconds")
        
        if self.unparseable_games:
            print(f"Unparseable game types: {self.unparseable_games}")

    def get_profile_info(self):
        # Get profile information
        profile_url = f"https://api.chess.com/pub/player/{self.username}"
        response = self.session.get(profile_url, headers=self.headers)
        if response.status_code == 200:
            profile_response = response.json()
            self.name = profile_response.get('name')
            self.title = profile_response.get('title')
        else:
            raise Exception(f"Failed to fetch profile data for {self.username}. Status code: {response.status_code}")

    def get_archives_list(self):
        # Get player game archives (by month)
        archives_url = f"https://api.chess.com/pub/player/{self.username}/games/archives"
        response = self.session.get(archives_url, headers=self.headers)
        if response.status_code == 200: # if the response is valid, parse the response
            archives_data = response.json()
            # Obtain every month of play
            self.archived_months = archives_data.get("archives", [])
        else:
            raise Exception(f"Failed to fetch archives for {self.username}. Status code: {response.status_code}")
    
    def get_month_games(self, month_url):
        """Get games for a specific month"""
        try:
            response = self.session.get(month_url, headers=self.headers)
            if response.status_code == 200:
                return response.json().get("games", [])
            else:
                print(f"Error retrieving games for {month_url}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception while retrieving {month_url}: {e}")
            return []

    def process_game_archives(self):
        """Process all game archives in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start all fetch tasks
            future_to_month = {
                executor.submit(self.get_month_games, month): month 
                for month in self.archived_months
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_month):
                month = future_to_month[future]
                try:
                    games = future.result()
                    self.process_games(games)
                except Exception as e:
                    print(f"Error processing {month}: {e}")

    def process_games(self, games):
        """Process list of games"""
        for game in games:
            if 'pgn' in game:
                self.games_pgn.append(game['pgn'])
            else:
                self.unparseable_games.add(game.get('rules', 'unknown'))

    def create_model(self):
        """Create a neural network for chess position evaluation"""
        input_shape = (8, 8, 12)  # 8x8 board with 12 channels (6 piece types x 2 colors)
        
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='tanh')  # Output between -1 (black winning) and 1 (white winning)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        return model
    
    def board_to_input(self, board):
        """Convert a chess board to neural network input format"""
        # Initialize 8x8x12 input array (6 piece types for each color)
        x = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Piece type order: pawn, knight, bishop, rook, queen, king
        # First 6 channels: white pieces
        # Next 6 channels: black pieces
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Fill in the board representation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank, file = divmod(square, 8)
                color_offset = 0 if piece.color == chess.WHITE else 6
                x[rank, file, piece_idx[piece.piece_type] + color_offset] = 1
                
        return x
    
    def evaluate_position(self, board):
        """Evaluate chess position - combines material count with neural network evaluation"""
        if not self.trained:
            # If model isn't trained, use basic material evaluation
            material_score = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    material_score += self.piece_values.get(piece.symbol(), 0)
            return np.tanh(material_score / 15)
        
        # Basic material counting
        material_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                material_score += self.piece_values.get(piece.symbol(), 0)
        
        # Neural network evaluation
        board_input = np.expand_dims(self.board_to_input(board), axis=0)
        model_score = self.model.predict(board_input, verbose=0)[0][0]
        
        # Combine scores
        material_weight = 0.3
        model_weight = 0.7
        normalized_material = np.tanh(material_score / 15)
        
        return material_weight * normalized_material + model_weight * model_score
    
    def preprocess_pgn_game(self, pgn_text):
        """Extract training examples from a PGN game"""
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return []
        
        # Get result for use as the label
        result = game.headers.get("Result", "*")
        if result == "1-0":
            result_value = 1.0  # White win
        elif result == "0-1":
            result_value = -1.0  # Black win
        elif result == "1/2-1/2":
            result_value = 0.0  # Draw
        else:
            return []  # Unclear result, skip this game
        
        # Extract training examples from the game
        examples = []
        board = game.board()
        
        for move in game.mainline_moves():
            board.push(move)
            
            # Don't use very early game positions or positions after the game is effectively decided
            if board.fullmove_number < 5 or board.is_game_over():
                continue
                
            # Skip positions with obvious material imbalance
            material_diff = sum(self.piece_values.get(p.symbol(), 0) for p in board.piece_map().values())
            if abs(material_diff) > 10:
                continue
                
            # Create input features
            x = self.board_to_input(board)
            
            # Create target value (adjusted based on move number)
            move_num = board.fullmove_number
            progress_weight = min(move_num / 40, 1.0)
            target = progress_weight * result_value
            
            examples.append((x, target))
            
        return examples

if __name__ == "__main__":
    chess_bot = ChessComBot("AlanZhang", max_workers=8)
    print("Name:", chess_bot.name)
    print("Title:", chess_bot.title)
    print("# of Archived Months:", len(chess_bot.archived_months))
    print("# of Games:", len(chess_bot.games_pgn))
    print("Unparseable Game Types:", chess_bot.unparseable_games)
