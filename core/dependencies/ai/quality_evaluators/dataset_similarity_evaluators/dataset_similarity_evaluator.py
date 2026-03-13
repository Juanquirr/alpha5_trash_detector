from abc import abstractmethod, ABC


class DatasetSimilarityEvaluator(ABC):
    @abstractmethod
    def evaluate(self, dataset_url: str, synthetic_dataset_url: str) -> float:
        ...
