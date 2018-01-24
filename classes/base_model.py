import os
import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable


class base_model():
    """
    params:
        feed_forward: The feed forward method of a model
        training_method: The training loop of a model
        evaluate: Method for running a models feed_forward and parsing the output to UCI notation
        use_cuda: a boolean whether to use CUDA, only works if CUDA is installed
        resume: flag to say if we are resuming training
        filepath: file where to save the model
    """

    def __init__(self, feed_forward, training_method, evaluate, use_cuda=True, resume=False, filepath='auto'):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.filepath = filepath
        self.feed_forward = feed_forward
        self.training_method = training_method
        self.evaluate = evaluate
        if use_cuda:
            self.model.cuda()
        if resume:
            if not self.load_checkpoint():  # TODO decide a pattern for filenames
                self.start_epoch = 0
                self.initialize_weights()
        else:
            self.start_epoch = 0
            self.initialize_weights()

    def training_session(self, train_data, test_data, n_epochs):
        """function to manage the train session"""
        for epoch in range(self.start_epoch, n_epochs, 1):
            self.train(train_data, epoch)
            self.validate(test_data)

        self.save_params()
