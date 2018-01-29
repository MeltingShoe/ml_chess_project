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
