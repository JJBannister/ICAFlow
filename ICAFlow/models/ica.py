from abc import ABC, abstractmethod

class ICA(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_model(self, n_dimensions):
        pass

    @abstractmethod
    def train_model(self, data):
        pass

