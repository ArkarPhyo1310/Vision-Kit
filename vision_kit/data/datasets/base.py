from vision_kit.data.datasets.datasets_wrapper import Dataset


class BaseDataset(Dataset):
    def __init__(self, input_dimension):
        super().__init__(input_dimension)

    def load_ann(self, index: int):
        pass

    def pull_items(self, index: int):
        pass
