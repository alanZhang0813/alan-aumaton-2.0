import requests

class ChessComPlayer:
    def __init__(self, username):
        self.username = username
        self.name = None
        self.title = None
        self.archived_months = None
        self.games_pgn = []
        self.unparseable_games = set()
        self.user_agent = "Alan's Application for Designing a Chess Bot"
        self.load_player_data()
    
    def load_player_data(self):
        headers = {"User-Agent": self.user_agent}

        # Get profile information
        profile_url = f"https://api.chess.com/pub/player/{self.username}"
        response = requests.get(profile_url, headers=headers)
        if response.status_code == 200:
            profile_response = response.json()
            self.name = profile_response.get('name')
            self.title = profile_response.get('title')
        
        # Get player game archives (by month)
        archives_url = f"https://api.chess.com/pub/player/{self.username}/games/archives"
        response = requests.get(archives_url, headers=headers)
        if response.status_code == 200: # if the response is valid, parse the response
            archives_data = response.json()
            # Obtain every month of play
            self.archived_months = archives_data.get("archives", [])
        
        # For every month, parse through the games found at that endpoint
        # https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM} which is already given in the response
        for month in self.archived_months:
            # print(month)
            response = requests.get(month, headers=headers)
            if response.status_code == 200:
                # print("Game successfully found!")
                games = response.json().get("games")
                for game in games:
                    try:
                        pgn = game['pgn']
                        # print(pgn)
                        self.games_pgn.append(pgn)
                    except:
                        self.unparseable_games.add(game['rules'])
                        next # Some games, such as bughouse, cannot be properly parsed and that's ok

def main():
    chess_bot = ChessComPlayer("AlanZhang")
    print("Name:", chess_bot.name)
    print("Title:", chess_bot.title)
    print("# of Archived Months:", len(chess_bot.archived_months))
    print("# of Games:", len(chess_bot.games_pgn))
    print("Unparseable Game Types:", chess_bot.unparseable_games)

main()