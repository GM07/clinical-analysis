from abc import ABC, abstractmethod


class Pipeline(ABC):
    """
    General pipeline class
    """

    def __init__(self, output_file_path):
        super().__init__()
        self.output_file_path = output_file_path

    @abstractmethod
    def __call__(self, *args, **kwds):
        """Runs the pipeline"""
        pass
