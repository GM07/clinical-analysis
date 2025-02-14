


from src.data.mimic import Mimic


class MimicLoader(Mimic):
    """
    Loads the Mimic dataset and formats it into a pandas DataFrame.
    """

    def __init__(self, mimic_path: str, load: bool = True):
        super().__init__(mimic_path, load)
