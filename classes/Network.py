import torch.optim as optim
from torch.autograd import Variable
import torch.load as tLoad

from Model import Model

class Network():
    """
    params:
        model: Class of the model we want to use
        loss_function: a function like nn.MSELoss
        optimizer: optimizer like optim.SGD
        learning rate: float representing learning rate of the model
    """
    def __init__(self, model, loss_function, optimizer, learning_rate):
        self.model = model() #  here we call the model we need
        self.loss_function = loss_function
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

    def training(self, dataset_loader, n_epochs):
        """function to train the network """
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for i, data in enumerate(dataset_loader, 0):
                inputs, labels = data #get inputs and labels

                inputs = Variable(inputs)
                labels = Variable(labels)

                self.optimizer.zero_grad() #zero parameters gradient

                outputs = self.model(inputs) #forward
                loss = self.loss_function(outputs, labels) #compute loss
                loss.backward() #backpropagation
                self.optimizer.step() #optimization

                epoch_loss += loss.data[0]

                if i%2000 == 0:
                    print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, epoch_loss / 2000))
                    epoch_loss = 0.0  

    def weights_init(self, mean=0.0, variance=0.02, bias = -1):
        """ intialize network weights as Normal Distribution of given mean and variance """
        self.model.weight.data.normal_(mean, variance)
        if bias >= 0:
            self.model.bias.data.fill_(bias)      

    def save_params(self):
        """save model to file """
        path = 'path_to_file'
        self.model.save_state_dict(path)

    def load_params(self):
        """ load model from file """
        path = 'path_to_file'
        self.model.load_state_dict(tLoad.load(path))

    def save_checkpoint(self):
        """https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3"""
        #TODO
        return 0

    def load_checkpoint(self):
        #TODO
        return 0        
