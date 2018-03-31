import torch
from torch.autograd import Variable
import operator
import numpy as np
import random

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
    evals = {}
    i = 0
    while(i < len(legal_moves)):
        self.env.alt_step(legal_moves[i])
        board = self.board()
        # this might break the app for people without cuda
        board = torch.cuda.FloatTensor(board)
        board = Variable(board)
        '''
        I literally have no idea what .data.cpu().numpy() does or if it
        slows anything down but it doesn't work without it
        '''
        out = self.feed_forward(board).data.cpu().numpy()
        evals[legal_moves[i]] = out
        self.env.alt_reset()
        i += 1

    # quickest way i could think to add rng because i wanna train while i sleep
    i = 0
    moves = []
    while(i < 3):
        if evals:
            move = max(evals.items(), key=operator.itemgetter(1))[0]
            moves.append(move)
            evals.pop(move, None)
            if i == 2:
                rng = random.randint(0,2)
        else:
            moves = legal_moves
            rng = 0
        i += 1
    
    move = moves[rng]



    envOut = self.env._step(move)
    out = {
    'state': start_pos,
    'reward': envOut[1],
    'isTerminated': envOut[2],
    'move': move
    }
    return out

