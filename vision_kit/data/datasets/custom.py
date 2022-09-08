from vision_kit.data.datasets.base import BaseDataset


class CustomDataset(BaseDataset):
    def __init__(self, input_dimension):
        super().__init__(input_dimension)

    def pull_items(self, index: int):
        return super().pull_items(index)

    def load_ann(self, index: int):
        return super().load_ann(index)
