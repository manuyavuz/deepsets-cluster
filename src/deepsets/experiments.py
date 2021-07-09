from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.autograd import Variable
from tqdm.auto import tqdm, trange
from IPython import embed
import datetime
import os
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import rand_score, adjusted_rand_score, confusion_matrix

from .datasets import MNISTSummation, MNIST_TRANSFORM
from .networks import InvariantModel, SmallMNISTCNNPhi, SmallRho, ClusterClf, OracleClf

from PIL import Image
from torchvision import transforms

from sklearn.metrics import confusion_matrix, log_loss
import seaborn as sns

os.makedirs("tsne/", exist_ok=True)
os.makedirs("cmat/", exist_ok=True)

import wandb

class SumOfDigits(object):
    def __init__(self, lr=1e-3, wd=5e-3, dsize=100000, set_size=500):
        self.lr = lr
        self.wd = wd
        self.set_size = set_size
        # self.train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=dsize, train=True, transform=MNIST_TRANSFORM)
        self.train_db = MNISTSummation(min_len=self.set_size, max_len=self.set_size, dataset_len=dsize, train=True, transform=MNIST_TRANSFORM)
        # self.test_db = MNISTSummation(min_len=5, max_len=50, dataset_len=dsize, train=False, transform=MNIST_TRANSFORM)
        self.test_db = MNISTSummation(min_len=self.set_size, max_len=self.set_size, dataset_len=dsize, train=False, transform=MNIST_TRANSFORM)

        self.clf = ClusterClf(input_size=10, output_size=10)
        # self.clf = OracleClf(input_size=10, output_size=10)
        wandb.init(sync_tensorboard=True, magic=True, tags=['classifier'])

        self.the_phi = SmallMNISTCNNPhi()
        # self.the_phi.load_state_dict(torch.load('trained_phi.pkl'))
        # for param in self.the_phi.parameters():
        #     param.requires_grad = False

        self.the_rho = SmallRho(input_size=10, output_size=10)
        # self.the_rho.load_state_dict(torch.load('trained_rho.pkl'))
        # for param in self.the_rho.parameters():
        #     param.requires_grad = False

        self.model = InvariantModel(phi=self.the_phi, rho=self.the_rho, clf=self.clf)
        wandb.watch(self.model, log_freq=10, log='all')
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        # self.optimizer1 = optim.Adam(list(self.the_phi.parameters()) + list(self.the_rho.parameters()), lr=self.lr, weight_decay=self.wd)
        # self.optimizer2 = optim.Adam(self.clf.parameters(), lr=self.lr, weight_decay=self.wd)

        self.summary_writer = SummaryWriter(
            log_dir='logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def train_1_epoch(self, epoch_num: int = 0):
        self.model.train()
        for i in trange(0, len(self.train_db), 2):
            n_train_steps = i + len(self.train_db) * epoch_num
            loss, score = self.train_1_item([i, i+1], n_train_steps)
            self.summary_writer.add_scalar('cosine_loss_train', loss, n_train_steps)
            self.summary_writer.add_scalar('rand_score_train', score, n_train_steps)
        x, target = self.train_db.__getitem__(0)
        if torch.cuda.is_available():
            x = x.cuda()

        self.summary_writer.add_graph(self.model, input_to_model=(x, target), verbose=False)

    def train_1_item(self, item_numbers: List[int], n_train_steps=None) -> float:
        x, target = self.train_db.__getitem__(item_number)
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()

        x, target = Variable(x), Variable(target)

        self.optimizer.zero_grad()
        # self.optimizer1.zero_grad()
        # self.optimizer2.zero_grad()

        pred, w = self.model.forward(x, target)
        pred_labels = torch.argmax(w.data, dim=1).cpu().numpy()

        self.summary_writer.add_scalar('cross_entropy_train', log_loss(target.squeeze().cpu().numpy(), w.cpu().numpy()), global_step=n_train_steps)

        target = torch.squeeze(target).cpu().numpy()


        # the_loss = F.mse_loss(pred, target)
        # the_loss = -torch.sum(torch.cdist(pred, pred)) / 2

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        the_loss = 0
        for i in range(pred.shape[1]):
            neg_loss = 0
            for j in range(pred.shape[1]):
                if j == i:
                    pos_loss = torch.exp(cos(pred[0, i], pred[1, j]))
                else:
                    neg_loss += torch.exp(cos(pred[0, i], pred[1, j]))
            the_loss -= torch.log(pos_loss / neg_loss)

        the_loss.backward()
        self.optimizer.step()
        # if i % 10 == 0:
        #     self.optimizer2.step()
        # else:
        #     self.optimizer1.step()
        # wandb.log({"loss": the_loss})

        the_loss_tensor = the_loss.data
        if torch.cuda.is_available():
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        # cmat = confusion_matrix(pred_labels, target)
        # plt.figure(figsize=(6, 5))
        # plt.matshow(cmat)
        # plt.savefig("cmat/train_%d.png" % item_number)

        score = rand_score(pred_labels, target)
        # score = adjusted_rand_score(pred_labels, target)

        return the_loss_float, score

    def calculate_loss(data, target):
        pass

    def evaluate(self, epoch):
        self.model.eval()
        # totals = [0] * 51
        # corrects = [0] * 51
        tsne = TSNE(n_components=2, random_state=0)

        X = np.zeros([len(self.test_db) * 10, 10])
        Y = np.zeros(len(self.test_db) * 10, dtype=int)
        A = np.zeros(len(self.test_db) * self.set_size, dtype=int)
        B = np.zeros(len(self.test_db) * self.set_size, dtype=int)

        total_predictions = []
        total_targets = []
        for i in trange(len(self.test_db)):
            x, target = self.test_db.__getitem__(i)

            item_size = x.shape[0]

            if torch.cuda.is_available():
                x = x.cuda()

            # pred = self.model.forward(Variable(x)).data
            pred, w = self.model.forward(Variable(x), Variable(target))
            pred = pred.data[0]
            pred_labels = torch.argmax(w.data, dim=1)
            A[i * self.set_size: (i + 1) * self.set_size] = pred_labels.cpu().numpy()

            # if torch.cuda.is_available():
            # pred = pred.cpu().numpy().flatten()
            X[i * 10: (i + 1) * 10] = pred.cpu().numpy()

            # pred = int(round(float(pred[0])))
            # target = int(round(float(target.numpy()[0])))
            B[i * self.set_size: (i + 1) * self.set_size] = torch.squeeze(target).cpu().numpy()
            Y[i * 10: (i + 1) * 10] = np.arange(10)

            total_predictions.append(pred_labels)
            total_targets.append(target)

            # totals[item_size] += 1

            # if pred == target:
            #     corrects[item_size] += 1

        # totals = np.array(totals)
        # corrects = np.array(corrects)

        # print(corrects / totals)

        cmat = confusion_matrix(A, B)
        plt.figure(figsize=(6, 5))
        plt.matshow(cmat)
        plt.savefig("cmat/test_%d.png" % epoch)

        score = rand_score(A, B)
        # score = adjusted_rand_score(A, B)
        self.summary_writer.add_scalar('rand_score_eval', score, epoch)

        X_2d = tsne.fit_transform(X)

        target_ids = range(10)
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
        for i, c, label in zip(target_ids, colors, target_ids):
            plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
            # plt.scatter(X[Y == i, 0], X[Y == i, 1], c=c, label=label)

        plt.legend()
        image_path = "tsne/test_%d.png" % epoch
        plt.savefig(image_path)
        self.record_image_tensorboard(image_path, 'tsne_embeddings', step=epoch)

        self.record_confusion_matrix(torch.cat(total_predictions), torch.cat(total_targets), epoch)


    def record_confusion_matrix(self, pred, target, epoch):
        labels = range(10)
        matrix = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=labels)
        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels)
        image_path = f'figures/conf_matrix_epoch{epoch}.png'
        plt.savefig(image_path)
        self.record_image_tensorboard(image_path, 'confusion_matrix')

    def record_image_tensorboard(self, image_path, label, step=None):
        img = Image.open(image_path)
        img_tensor = transforms.ToTensor()(img)
        self.summary_writer.add_image(label, img_tensor, global_step=step)
        wandb.log({label : wandb.Image(image_path)}, step=step)
