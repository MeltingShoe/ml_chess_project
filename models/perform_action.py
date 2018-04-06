import torch
from torch.autograd import Variable
import operator
import numpy as np
import random
import torch.nn.functional as F
import time
from classes import utils
from torch.distributions import Categorical


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
    outputs = F.softmax(outputs, dim=0)

    outputs = outputs.view(-1)
    categorical = Categorical(outputs)
    move_index = categorical.sample().data.cpu().numpy()[0]

    move = legal_moves[move_index]

    envOut = self.env._step(move)
    out = {
        'state': start_pos,
        'reward': envOut[1],
        'isTerminated': envOut[2],
        'move': move
    }
    return out
