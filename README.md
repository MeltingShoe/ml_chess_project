# ml_chess_project

## Pre-requisites:
Below are a list of packages that are required to get the game to work, along with links for installation instructions.

### python-chess
For usage and installation instructions, see: https://pypi.python.org/pypi/python-chess

### OpenAI Gym
For installation instructions, see: https://www.youtube.com/watch?v=Io1wOuHEyW0&feature=youtu.be

## Running the game without installation:
While in the same directory as chess_env.py, you can try running the script below in Python:

```
from chess_env import ChessEnv

env = ChessEnv()
env.reset()
env.render()
env.step("b2b4")
env.render()
```

## gym_chess Installation

```
pip install -e .
```

## Running chess as a gym environment
```
import gym
import gym_chess

env = gym.make('chess-v0')
env.render()
```
