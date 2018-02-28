import torch
from torch.autograd import Variable
import operator


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
    board = torch.from_numpy(observation_space[0])
    legal_moves = observation_space[1]
    evals = {}
    i = 0
    while(i < len(legal_moves)):
        self.env.alt_step(legal_moves[i])
        board = observation_space[0]
        out = self.feed_forward(board)
        evals[legal_moves[i]] = out
        self.env.alt_reset()
    move = max(evals.iteritems(), key=operator.itemgetter(1))[0]
    self.env._step(move)
