from abc import ABC, abstractmethod

# doesn't really do anything because of how model classes are instantiated


class BaseModel(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
