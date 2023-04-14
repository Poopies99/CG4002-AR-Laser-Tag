class GameState:
    def __init__(self):
        self._bullets = 6
        self._health = 100

    @property
    def bullets(self):
        return self._bullets

    @property
    def health(self):
        return self._health
    
    def update_game_state(self, updated_game_state):
        self._bullets = updated_game_state[0]
        self._health = updated_game_state[1]
