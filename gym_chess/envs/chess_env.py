import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

try:
    import chess
except ImportError as e:
    raise error.DependencyNotInstalled("{}.  (HINT: see README for python-chess installation instructions".format(e))


class ChessEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = chess.Board()

    def _step(self, action):
        """
        alternative step?:
            move = board.san(chess.Move(chess.E2, chess.E4))
            self.env.push_san(move)

        :return:
            state: numpy array
            reward: Float
            is_terminated: boolean
            info: dictionary containing any debugging information
        """
        self.env.push_san(action)

        state = self._get_array_state()
        reward = float(0)
        is_terminated = self.env.is_game_over()
        info = {}
        return state, reward, is_terminated, info

    def _reset(self):
        """
        :return: current state as numpy array
        """
        self.env.reset()
        state = self._get_array_state()
        return state

    def _render(self, mode='human', close=False):
        print(self.env)

    def _get_array_state(self):
        """
        input: String from chess.Board.board_fen().  Ex.: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'

        Each lower-case character is black piece, and upper case is white piece.

        :return: 8x8 numpy array.  Current player's pieces are positive integers, enemy pieces are negative.
        """
        state = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]])

        split_board = self.env.board_fen().split("/")
        row = 0
        for rank in split_board:
            col = 0
            for file in rank:
                if file.isdigit():
                    col += int(file)
                else:
                    piece_enum = chess.Piece.from_symbol(file).piece_type
                    if (self.env.turn and file.islower()) or (file.isupper() and not self.env.turn):
                        piece_enum *= -1

                    state[row][col] = piece_enum
                    col += 1
            row += 1
        return state
