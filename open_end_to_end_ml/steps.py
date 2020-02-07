from abc import ABC, abstractmethod


class DataAugmentation(ABC):
    @abstractmethod
    def augment(self, data, random_state, is_train):
        pass


class DataCleaning(ABC):
    @abstractmethod
    def clean(self, data, random_state, is_train):
        pass


class FeatureAdder(ABC):
    @abstractmethod
    def add(self, data, random_state, is_train):
        pass


class ModelTrainer(ABC):
    @abstractmethod
    def create_pipeline(self, random_state):
        pass
