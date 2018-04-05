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


def split_by_piece(self):
    '''
    not very elegant, but splits pieces into a 6x8x8 array
    white pieces are positive, black negative
    each slice corresponds to the env piece value -1
    '''
    board = self.env._get_array_state()
    board = board[0]
    b0 = np.zeros((6, 8, 8))
    count = 0
    while count < 64:
        floor = count // 8
        mod = count % 8
        piece_type = abs(board[floor][mod])
        if(board[floor][mod] > 0):
            piece_side = 1
        else:
            piece_side = -1
        if(piece_type != 0):
            b0[piece_type - 1][floor][mod] = piece_side
        count += 1
    return b0


def split_by_piece_and_side(self):
    '''
    board[0] = white pawns
    board[1] = white knights
    board[5] = black pawns etc
    '''
    board = self.env._get_array_state()
    board = board[0]
    b0 = np.zeros((12, 8, 8))
    count = 0
    while count < 64:
        floor = count // 8
        mod = count % 8
        piece = board[floor][mod]
        if(piece < 0):
            piece = abs(piece) + 6
        if(piece != 0):
            b0[piece - 1][floor][mod] = 1
        count += 1
    return b0


def split_by_piece_and_side_with_empty(self):
    # same as above, but board[0] = empty squares
    board = self.env._get_array_state()
    board = board[0]
    print(board)
    b0 = np.zeros((13, 8, 8))
    count = 0
    while count < 64:
        floor = count // 8
        mod = count % 8
        piece = board[floor][mod]
        if(piece < 0):
            piece = abs(piece) + 6
        b0[piece][floor][mod] = 1
        count += 1
    print(b0)
    return b0
