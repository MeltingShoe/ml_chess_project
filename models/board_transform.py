import numpy as np
import torch
from torch.autograd import Variable

'''
This file is for functions that transform the board representation
For example splitting the board to an 8x8x12 array
Even feeding the board through an autoencoder would be possible
'''

def noTransform(self):
    board = self.env._get_array_state()
    board = board[0]
    return board

