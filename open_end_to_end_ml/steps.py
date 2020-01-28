from abc import ABC, abstractmethod


class DataAugmentation(ABC):
    @abstractmethod
    def augment(self, data):
        pass


class DataCleaning(ABC):
    @abstractmethod
    def clean(self, data):
        pass


class FeatureAdder(ABC):
    @abstractmethod
    def add(self, data):
        pass


class ModelTrainer(ABC):
    @abstractmethod
    def create_pipeline(self):
        pass
