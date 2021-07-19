from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from IPython import embed

from .settings import DATA_ROOT
from hashlib import md5

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MNIST_TRANSFORM = Compose([ToTensor(), Normalize((MNIST_MEAN,), (MNIST_STD,))])

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

        self.mnist = MNIST(DATA_ROOT, train=self.train, transform=self.transform, download=True)
        mnist_len = self.mnist.__len__()
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

        # the_sum = 0
        images = []
        labels = []
        for mi in mnist_items:
            img, target = self.mnist[mi]
            # the_sum += target
            images.append(img)
            labels.append(torch.LongTensor([target]))

        # return torch.stack(images, dim=0), torch.FloatTensor([the_sum])
        return torch.stack(images, dim=0), torch.stack(labels)

    def umap_embeddings_path(self):
        return Path(DATA_ROOT).expanduser() / f'umap.{md5(repr(self._args).encode("utf-8")).hexdigest()}.npy'
        
    def calculate_umap_embeddings(self):
        path = self.umap_embeddings_path()
        if path.exists():
            embeddings = np.load(path)
        else:
            print('Computing UMAP embeddings..')
            imgs = []
            targets = []
            for img, target in self.mnist:
                imgs.append(img)
                targets.append(target)
            imgs = torch.cat(imgs)
            targets = torch.tensor(targets)
            embedder = UMAP(verbose=1)
            embeddings = embedder.fit_transform(imgs.reshape(imgs.shape[0], -1))
            np.save(path, embeddings)
        return embeddings
