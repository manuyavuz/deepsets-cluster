import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from IPython import embed
import datetime
import os
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import rand_score

from .datasets import MNISTSummation, MNIST_TRANSFORM
from .networks import InvariantModel, SmallMNISTCNNPhi, SmallRho, ClusterClf

os.makedirs("tsne/", exist_ok=True)


class SumOfDigits(object):
    def __init__(self, lr=1e-3, wd=5e-3, dsize=100000):
        self.lr = lr
        self.wd = wd
        # self.train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=dsize, train=True, transform=MNIST_TRANSFORM)
        self.train_db = MNISTSummation(min_len=500, max_len=500, dataset_len=dsize, train=True, transform=MNIST_TRANSFORM)
        # self.test_db = MNISTSummation(min_len=5, max_len=50, dataset_len=dsize, train=False, transform=MNIST_TRANSFORM)
        self.test_db = MNISTSummation(min_len=500, max_len=500, dataset_len=dsize, train=False, transform=MNIST_TRANSFORM)

        # self.clf = SmallMNISTCNNPhi(softmax=True)
        self.clf = ClusterClf(input_size=10)
        for param in self.clf.parameters():
            param.requires_grad = False
        self.the_phi = SmallMNISTCNNPhi()
        self.the_rho = SmallRho(input_size=10, output_size=1)

        self.model = InvariantModel(phi=self.the_phi, rho=self.the_rho, clf=self.clf)
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        # self.optimizer1 = optim.Adam(list(self.the_phi.parameters()) + list(self.the_rho.parameters()), lr=self.lr, weight_decay=self.wd)
        # self.optimizer2 = optim.Adam(self.clf.parameters(), lr=self.lr, weight_decay=self.wd)

        self.summary_writer = SummaryWriter(
            log_dir='/home/rishabhgupta/pytorch-deep-sets/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def train_1_epoch(self, epoch_num: int = 0):
        self.model.train()
        for i in tqdm(range(len(self.train_db))):
            loss, score = self.train_1_item(i)
            self.summary_writer.add_scalar('train_loss', loss, i + len(self.train_db) * epoch_num)
            self.summary_writer.add_scalar('rand_score', score, i + len(self.train_db) * epoch_num)

    def train_1_item(self, item_number: int) -> float:
        x, target = self.train_db.__getitem__(item_number)
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()

        x, target = Variable(x), Variable(target)

        self.optimizer.zero_grad()
        # self.optimizer1.zero_grad()
        # self.optimizer2.zero_grad()

        pred, w = self.model.forward(x)
        pred_labels = torch.argmax(w.data, dim=1).cpu().numpy()
        target = torch.squeeze(target).cpu().numpy()

        # the_loss = F.mse_loss(pred, target)
        # the_loss = -torch.sum(torch.cdist(pred, pred)) / 2

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        the_loss = 0
        count = 0
        for i in range(pred.shape[0]):
            for j in range(i):
                the_loss += cos(pred[i], pred[j])
                count += 1
        the_loss = the_loss / count

        the_loss.backward()
        self.optimizer.step()
        # if i % 10 == 0:
        #     self.optimizer2.step()
        # else:
        #     self.optimizer1.step()

        the_loss_tensor = the_loss.data
        if torch.cuda.is_available():
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        score = rand_score(pred_labels, target)

        return the_loss_float, score

    def evaluate(self, epoch):
        self.model.eval()
        # totals = [0] * 51
        # corrects = [0] * 51
        tsne = TSNE(n_components=2, random_state=0)

        X = np.zeros([len(self.test_db) * 10, 10])
        Y = np.zeros(len(self.test_db) * 10, dtype=int)
        A = np.zeros(len(self.test_db) * 500, dtype=int)
        B = np.zeros(len(self.test_db) * 500, dtype=int)

        for i in tqdm(range(len(self.test_db))):
            x, target = self.test_db.__getitem__(i)

            item_size = x.shape[0]

            if torch.cuda.is_available():
                x = x.cuda()

            # pred = self.model.forward(Variable(x)).data
            pred, w = self.model.forward(Variable(x))
            pred = pred.data
            pred_labels = torch.argmax(w.data, dim=1)
            A[i * 500: (i + 1) * 500] = pred_labels.cpu().numpy()

            # if torch.cuda.is_available():
            # pred = pred.cpu().numpy().flatten()
            X[i * 10: (i + 1) * 10] = pred.cpu().numpy()

            # pred = int(round(float(pred[0])))
            # target = int(round(float(target.numpy()[0])))
            B[i * 500: (i + 1) * 500] = torch.squeeze(target).cpu().numpy()
            Y[i * 10: (i + 1) * 10] = np.arange(10)

            # totals[item_size] += 1

            # if pred == target:
            #     corrects[item_size] += 1

        # totals = np.array(totals)
        # corrects = np.array(corrects)

        # print(corrects / totals)

        score = rand_score(A, B)
        self.summary_writer.add_scalar('rand_score_eval', score, epoch)

        X_2d = tsne.fit_transform(X)

        target_ids = range(10)
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
        for i, c, label in zip(target_ids, colors, target_ids):
            plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
            # plt.scatter(X[Y == i, 0], X[Y == i, 1], c=c, label=label)

        plt.legend()
        plt.savefig("tsne/test_%d.png" % epoch)
