from abc import abstractmethod

class BasePreprocessor:
    @abstractmethod
    def process(self, example: dict) -> dict | None:
        pass

