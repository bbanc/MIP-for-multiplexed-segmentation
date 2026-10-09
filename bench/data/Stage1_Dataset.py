import torch
from data.Generic_Dataset import GenericDataset


class Stage1Dataset(GenericDataset):
    def __init__(self, dataset_a, dataset_b):
        super().__init__()
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

    def __getitem__(self, idx):
        if torch.rand(1).item() > 0.5:
            return self.dataset_a[idx % len(self.dataset_a)]
        else:
            return self.dataset_b[idx % len(self.dataset_b)]

    def __len__(self):
        return len(self.dataset_a) + len(self.dataset_b)
