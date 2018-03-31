import numpy as np
import torch
from torch.autograd import Variable

'''
This file is for functions that transform the board representation
For example splitting the board to an 8x8x12 array
Even feeding the board through an autoencoder would be possible
'''

#This doesn't actually transform the board, just manipulates the array to let pytorch work
def hacky_workaround(self):
    board = self.env._get_array_state()
    board = board[0]
    # Extremely lazy and hacky because pytorch wasn't cooperating
    x = board.tolist()
    x = [[x]]
    x = torch.cuda.FloatTensor(x)
    x = torch.autograd.Variable(x)
    return x