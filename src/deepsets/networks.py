from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable
from IPython import embed

NetIO = Union[FloatTensor, Variable]


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module, clf: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.clf = clf

    def forward(self, x: NetIO, y=None) -> NetIO:
        # # compute the representation for each data point
        # x = self.phi.forward(x)

        # w = self.clf.forward(x)
        z = self.phi.forward(x)
        w = self.clf.forward(z, y)
        z = z.reshape(2, -1, z.size(1))
        w = w.reshape(2, -1, w.size(1))

        # # sum up the representations
        # # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        # x = torch.sum(x, dim=0, keepdim=True)

        # x = torch.matmul(w.t(), z)
        x = torch.matmul(w.transpose(1, 2), z)

        # compute the output
        out = self.rho.forward(x)

        return out, w.reshape(-1, w.size(2))


class SmallMNISTCNNPhi(nn.Module):
    def __init__(self, softmax=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)
        self.softmax = softmax

    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        if self.softmax:
            x = F.softmax(self.fc2(x))
        else:
            x = F.relu(self.fc2(x))
        return x


class SmallRho(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, self.output_size)

    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ClusterClf(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, output_size)

    def forward(self, x: NetIO, y=None) -> NetIO:
        x = F.softmax(self.fc1(x), dim=1)
        return x


class OracleClf(ClusterClf):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, x: NetIO, y=None) -> NetIO:
        # x = super().forward(x, y)
        x = torch.zeros_like(x)
        x[range(len(y)), y.squeeze()] = 1
        return x
