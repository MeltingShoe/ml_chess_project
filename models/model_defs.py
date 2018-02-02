from classes.base_model import generate_class
import models.feed_forward as ff
import models.train as tr
import models.perform_action as pa
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
    'learning_rate': 0.001,
    'optimizer': optim.Adam,
    'loss_function': nn.CrossEntropyLoss
}
cifar10_model = generate_class(
    ff.BasicConvNet(), tr.default_train, pa.supervised_evaluate, cifar_10_params)
