from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from IPython import embed

from .settings import DATA_ROOT
from .utils import encode_dict

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MNIST_TRANSFORM = Compose([ToTensor(), Normalize((MNIST_MEAN,), (MNIST_STD,))])

# from cuml.manifold.umap import UMAP
from umap import UMAP
from pathlib import Path
class MNISTSummation(Dataset):
    def __init__(self, **kwargs):
        self._args = kwargs
        self.min_len = kwargs['min_len']
        self.max_len = kwargs['max_len']
        self.dataset_len = kwargs['dataset_len']
        self.train = kwargs['train']
        self.transform = kwargs['transform']

        self.mnist_dataset = MNIST(DATA_ROOT, train=self.train, transform=self.transform, download=True)
        self.mnist_loader = DataLoader(self.mnist_dataset, batch_size=len(self.mnist_dataset)//8, shuffle=True, num_workers=8, pin_memory=True)
        self.mnist_data = []
        self.mnist_labels = []
        for data, labels in self.mnist_loader:
            self.mnist_data.append(data)
            self.mnist_labels.append(labels)
        self.mnist_data = torch.cat(self.mnist_data)
        self.mnist_labels = torch.cat(self.mnist_labels)
        mnist_len = len(self.mnist_dataset)
        mnist_items_range = np.arange(0, mnist_len)

        items_len_range = np.arange(self.min_len, self.max_len + 1)
        items_len = np.random.choice(items_len_range, size=self.dataset_len, replace=True)
        self.mnist_items = []
        for i in range(self.dataset_len):
            self.mnist_items.append(np.random.choice(mnist_items_range, size=items_len[i], replace=True))

        self.embeddings_umap = self.calculate_umap_embeddings()

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor]:
        mnist_items = self.mnist_items[item]

        images = self.mnist_data[mnist_items]
        labels = self.mnist_labels[mnist_items].unsqueeze(-1)
        return images, labels

    def umap_embeddings_path(self):
        return Path(DATA_ROOT).expanduser() / f'umap.{encode_dict(self._args)}.npy'
        
    def calculate_umap_embeddings(self):
        path = self.umap_embeddings_path()
        if path.exists():
            embeddings = np.load(path)
        else:
            print('Computing UMAP embeddings..')
            imgs = []
            targets = []
            for img, target in zip(self.mnist_data, self.mnist_labels):
                imgs.append(img)
                targets.append(target)
            imgs = torch.cat(imgs)
            targets = torch.tensor(targets)
            embedder = UMAP(verbose=1)
            embeddings = embedder.fit_transform(imgs.reshape(imgs.shape[0], -1).numpy())
            np.save(path, embeddings)
        return embeddings
