from vision_kit.data.datasets.datasets_wrapper import Dataset


class CustomDataset(Dataset):
    def __init__(self, input_dimension):
        super().__init__(input_dimension)

    def pull_items(self, index: int):
        pass

    def load_ann(self, index: int):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
