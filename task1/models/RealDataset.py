import torch
from torch.utils.data import Dataset


class RealDataset(Dataset):
    def __init__(self, features, labels):
        print(type(features))
        print(type(labels))
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
