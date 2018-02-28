import torch
from torch.autograd import Variable
import operator
import numpy as np


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
    # print for debugging
    print('PA start')
    observation_space = self.env._get_array_state()
    legal_moves = observation_space[1]
    evals = {}
    i = 0
    while(i < len(legal_moves)):
        self.env.alt_step(legal_moves[i])
        board = observation_space[0]
        # Extremely lazy and hacky because pytorch was cooperating
        board = board.tolist()
        board = [[board]]
        board = torch.cuda.FloatTensor(board)
        board = Variable(board)
        out = self.feed_forward(board)
        print(i)
        evals[legal_moves[i]] = out
        self.env.alt_reset()
        i += 1
    move = max(evals.iteritems(), key=operator.itemgetter(1))[0]
    self.env._step(move)
