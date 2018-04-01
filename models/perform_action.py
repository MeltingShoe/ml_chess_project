import torch
from torch.autograd import Variable
import operator
import numpy as np
import random
import torch.nn.functional as F
import time
from classes import utils

def supervised_evaluate(self, feed_forward, dataloader):
    correct_count = 0
    total_count = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        if self.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs = Variable(inputs)

        outputs = feed_forward(inputs)
        max, argmax = torch.max(outputs.data, 1)
        predicted = argmax

        correct_count += (predicted == labels).sum()
        total_count += dataloader.batch_size

    print('Accuracy on the validation set: {0}'.format(
        100.0 * correct_count / total_count))

# TODO, naming conventions


def PA_legal_move_values(self):
    observation_space = self.env._get_array_state()
    start_pos = self.board()
    legal_moves = observation_space[1]
    outputs = utils.FloatTensor(0).zero_()
    i = 0
    while(i < len(legal_moves)):
        self.env.alt_step(legal_moves[i])
        board = self.board()
        board = utils.FloatTensor(board)
        board = Variable(board)
        '''
        I literally have no idea what .data.cpu().numpy() does or if it
        slows anything down but it doesn't work without it
        '''
        out = self.feed_forward(board).data
        outputCat = torch.cat((outputs, out), 0)
        outputs = outputCat
        self.env.alt_reset()
        i += 1
    outputs = Variable(outputs)
    outputs = F.softmax(outputs, dim=0).data.tolist()
    rng = random.random()
    count = 0
    index = 0
    while(count < rng):
        count += outputs[index][0]
        index += 1
    move_len = len(legal_moves)
    if(index >= move_len):
        move = legal_moves[move_len-1]
    else:
        move = legal_moves[index]


    envOut = self.env._step(move)
    out = {
    'state': start_pos,
    'reward': envOut[1],
    'isTerminated': envOut[2],
    'move': move
    }
    return out

