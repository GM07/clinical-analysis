from abc import ABC, abstractmethod


class Pipeline(ABC):
    """
    General pipeline class
    """

    @abstractmethod
    def __call__(self, *args, **kwds):
        """Runs the pipeline"""
        pass
