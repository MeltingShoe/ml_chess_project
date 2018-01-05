import gym
from gym import error, spaces, utils
from gym.utils import seeding

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
        """
        self.env.push_uci(action)

        state = self.env
        reward = 0
        is_terminated = self.env.is_game_over()
        info = {}
        return state, reward, is_terminated, info

    def _reset(self):
        self.env.reset()
        return self.env

    def _render(self, mode='human', close=False):
        print(self.env)
