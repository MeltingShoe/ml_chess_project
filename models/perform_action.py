import torch
from torch.autograd import Variable
import operator
import numpy as np
import random
import torch.nn.functional as F
import time
from classes import utils


def PA_legal_move_values(self):
    observation_space = self.env._get_array_state()
    start_pos = self.board()
    legal_moves = observation_space[1]
    i = 0
    boards = []
    for i in range(len(legal_moves)):
        self.env.alt_step(legal_moves[i])
        board = self.board()
        boards.append(board)
        self.env.alt_reset()

    outputs = self.feed_forward(
        Variable(utils.FloatTensor(np.concatenate(boards))))
    outputs = F.softmax(outputs, dim=0).data.cpu().numpy().ravel()

    s = sum(outputs)
    # does this do anything?
    outputs / np.sum(outputs)

    move = np.random.choice(
        len(outputs),
        size=1,
        p=outputs,
    )

    outputs = self.feed_forward(
        Variable(utils.FloatTensor(np.concatenate(boards))))
    outputs = F.softmax(outputs, dim=0).data.cpu().numpy().ravel()
    outputs = outputs / np.sum(outputs)

    move_index = np.random.choice(len(outputs), size=1, p=outputs)[0]
    move = legal_moves[move_index]

    envOut = self.env._step(move)
    out = {
        'state': start_pos,
        'reward': envOut[1],
        'isTerminated': envOut[2],
        'move': move
    }
    return out
