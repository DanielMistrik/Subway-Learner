import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SubwayDataset(Dataset):
    def __init__(self, observations=None, labels=None):
        if observations is None:
            prefix = 'rollout_data/'
            self.observations = np.load(prefix + 'observations.npy', allow_pickle=True).astype(np.float32)
            self.acts = np.load(prefix + 'acts.npy', allow_pickle=True)
        else:
            self.observations = observations.astype(np.float32)
            self.acts = labels

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, idx):
        image = self.observations[idx]
        label = self.acts[idx]
        return image, label