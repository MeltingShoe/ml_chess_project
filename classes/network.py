import torch.optim as optim
import torch
from torch.autograd import Variable

class Network():
    """
    params:
        model: the model of type Model we want to use
        loss_function: a loss function like nn.MSELoss
        optimizer: an optimizer like optim.SGD
        learning rate: a float representing learning rate of the model
        use_cuda: a boolean whether to use CUDA, only works if CUDA is installed
    """
    def __init__(self, model, loss_function, optimizer, learning_rate, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.model = model
        if use_cuda:
            self.model.cuda()
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

    def training(self, dataloader, n_epochs, starting_index=0, print_batch=False):
        """function to train the network """
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for i, data in enumerate(dataloader, starting_index):
                inputs, labels = data #get inputs and labels
                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                inputs = Variable(inputs)
                labels = Variable(labels)

                self.optimizer.zero_grad() #zero parameters gradient

                outputs = self.model(inputs) #forward
                loss = self.loss_function(outputs, labels) #compute loss
                loss.backward() #backpropagation
                self.optimizer.step() #optimization

                epoch_loss += loss.data[0]

                if print_batch:
                    print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, loss.data[0]))

    def validate(self, dataloader):
        correct_count = 0
        total_count = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs = Variable(inputs)

            outputs = self.model(inputs)
            max, argmax = torch.max(outputs.data, 1)
            predicted = argmax

            correct_count += (predicted == labels).sum()
            total_count += dataloader.batch_size

        print('Accuracy on the validation set: {0}'.format(100.0 * correct_count / total_count))

    def initialize_weights(self, mean=0.0, variance=0.02, bias = -1):
        """ intialize network weights as Normal Distribution of given mean and variance """
        self.model.weight.data.normal_(mean, variance)
        if bias >= 0:
            self.model.bias.data.fill_(bias)      

    def save_params(self, path='auto'):
        """save model to file """
        if path == 'auto':
            path = 'path_to_file'
        torch.save(self.model.state_dict(), path)

    def load_params(self, path='auto'):
        """ load model from file """
        if path == 'auto':
            path = 'path_to_file'
        self.model.load_state_dict(torch.load(path))

    def save_checkpoint(self):
        """https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3"""
        #TODO
        return 0

    def load_checkpoint(self):
        #TODO
        return 0        
