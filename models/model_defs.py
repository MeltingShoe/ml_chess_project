from classes.base_model import generate_class
import models.feed_forward as ff
import models.train as tr
import models.perform_action as pa
import models.board_transform as bt
import torch.nn as nn
import torch.optim as optim

'''
I think all model classes will be defined in this file
We could have a single file for ff, train, and perform action as well
to simplify imports
'''


# feed_forward must be passed as an instance
cifar_10_params = {
    'name': 'cifar_10_model',
    'ff': ff.BasicConvNet(),
    'tr': tr.default_train,
    'pa': pa.supervised_evaluate,
    'bt': bt.noTransform,
    'learning_rate': 0.001,
    'optimizer': optim.Adam,
    'loss_function': nn.CrossEntropyLoss
}
cifar10_model = generate_class(cifar_10_params)


fc_test_params = {
	'name': 'fc_test',
    'ff': ff.ChessFC(),
    'tr': tr.default_train,
    'pa': pa.PA_legal_move_values,
    'bt': bt.noTransform,
    'learning_rate': 0.001,
    'optimizer': optim.Adam,
    'loss_function': nn.L1Loss
}
fc_test = generate_class(fc_test_params)