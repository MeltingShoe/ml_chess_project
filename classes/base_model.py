import torch
import classes.utils as utils


class base_model():  # TODO class names should use CamelCase convention
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
        if self.use_cuda:
            self.feed_forward.cuda()
            self.training_method.cuda()
            self.evaluate.cuda()
        if resume:
            if not utils.load_checkpoint(self):  # TODO decide a pattern for filenames
                self.start_epoch = 0
                utils.initialize_weights(self.feed_forward.modules())
        else:
            self.start_epoch = 0
            utils.initialize_weights(self.feed_forward.modules())

    def training_session(self, train_data, test_data, n_epochs):
        """function to manage the train session"""
        for epoch in range(self.start_epoch, n_epochs, 1):
            self.training_method.train(self.feed_forward, train_data, epoch)
            self.evaluate(self.feed_forward, test_data)

        utils.save_params(self.feed_forward.state_dict(), self.filepath)
