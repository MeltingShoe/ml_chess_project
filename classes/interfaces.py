from abc import ABC, abstractmethod


class BaseFeedForward(ABC):

    @abstractmethod
    def feed_forward(self, data):
        pass


'''
Right now all this does is require that an instance of a subclass must override the feed_forward function
At some point we may want to consider storing the board state as a property of the subclass
so we could define setters and getters in here so the feed forward calculation could be done by getting
the value of an internal property instead of having to pass data to a method
May not be necessary but I think it would make everything cleaner but I won't bother now since I'm just working on cifar10
'''


class BaseTrain(ABC):

    @abstractmethod
    def train(self, feed_forward, training_data, epoch):
        pass


'''
Same thing, just requires that train() is overridden
As I understand it this shouldn't enforce that method having those specific params
I think the params for LR, optimizer, etc should be initialized in the BaseModel class
'''
