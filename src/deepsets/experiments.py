from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.autograd import Variable
from tqdm.auto import tqdm, trange
import os
from cuml.manifold.t_sne import TSNE

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, rand_score, adjusted_rand_score, confusion_matrix

from .datasets import MNISTSummation, MNIST_MEAN, MNIST_STD, MNIST_TRANSFORM
from .networks import InvariantModel, SmallMNISTCNNPhi, SmallRho, ClusterClf, OracleClf
from .utils import outdir_for_run

from PIL import Image
from torchvision import transforms

from sklearn.metrics import confusion_matrix, log_loss
import seaborn as sns

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import wandb

def tags_for_dict(dict):
    return [f'{k}:{v}' for k,v in dict.items()]

class SumOfDigits(object):
    def __init__(self, **kwargs):
        wandb.init(project='set-cluster', entity='set-cluster', sync_tensorboard=True, config=kwargs)
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.set_size = kwargs['set_size']
        self.n_sets = kwargs['n_sets']
        # self.train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=dsize, train=True, transform=MNIST_TRANSFORM)
        self.train_db = MNISTSummation(min_len=self.set_size, max_len=self.set_size, dataset_len=self.n_sets, train=True, transform=MNIST_TRANSFORM)
        # self.test_db = MNISTSummation(min_len=5, max_len=50, dataset_len=dsize, train=False, transform=MNIST_TRANSFORM)
        self.test_db = MNISTSummation(min_len=self.set_size, max_len=self.set_size, dataset_len=self.n_sets, train=False, transform=MNIST_TRANSFORM)
        self.test_loader = DataLoader(self.test_db, pin_memory=True, batch_size=len(self.test_db), shuffle=True)

        classifier_type = kwargs['classifier']
        encoder_type = kwargs['encoder']
        model_path = kwargs['model_path']
        normalize_weights = kwargs['normalize_weights']
        normalize_weights_for_predictions = kwargs['normalize_weights_for_predictions']
        self.loss_type = kwargs['loss']
        self.similarity_metric = kwargs['similarity_metric']
        self.iterative_optimization = kwargs['iterative_optimization']
        self.experiment = kwargs['exp']

        if classifier_type == 'oracle':
            classifier = OracleClf
        elif classifier_type == 'train':
            classifier = ClusterClf
        self.out_dir = outdir_for_run(kwargs)
        self.figures_dir = self.out_dir / 'figures'
        self.checkpoints_dir = self.out_dir / 'checkpoints'
        self.logs_dir = self.out_dir / 'logs'

        print(f"Writing results under {self.out_dir}")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)


        self.the_phi = SmallMNISTCNNPhi()
        if self.experiment == 'classify':
            self.clf = classifier(input_size=10, output_size=10)
        else:
            self.clf = classifier(input_size=10, output_size=10, return_softmax=False)
        self.the_rho = SmallRho(input_size=10, output_size=10)

        if encoder_type in ['pretrained', 'finetune']:
            assert model_path is not None and len(model_path) > 0, 'Model path is not provided!'
            model_path = Path(model_path)
            assert model_path.exists(), f'Model path "{model_path}" does not exist!'
            self.the_phi.load_state_dict(torch.load(model_path / 'trained_phi_latest.pt'))
            self.the_rho.load_state_dict(torch.load(model_path / 'trained_rho_latest.pt'))
        if encoder_type == 'pretrained':
            for param in self.the_phi.parameters():
                param.requires_grad = False
            for param in self.the_rho.parameters():
                param.requires_grad = False

        self.joint_training = encoder_type == 'train' and classifier_type == 'train'

        self.model = InvariantModel(phi=self.the_phi, rho=self.the_rho, clf=self.clf, normalize_weights=normalize_weights, normalize_weights_for_predictions=normalize_weights_for_predictions)
        wandb.watch(self.model, log_freq=10, log='all')
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        # self.optimizer1 = optim.Adam(list(self.the_phi.parameters()) + list(self.the_rho.parameters()), lr=self.lr, weight_decay=self.wd)
        # self.optimizer2 = optim.Adam(self.clf.parameters(), lr=self.lr, weight_decay=self.wd)

        self.summary_writer = SummaryWriter(log_dir= self.logs_dir)
        

    def update_encoder_state(self, freeze:bool):
        for param in self.the_phi.parameters():
            param.requires_grad = not freeze
        for param in self.the_rho.parameters():
            param.requires_grad = not freeze

    def update_classifier_state(self, freeze:bool):
        for param in self.clf.parameters():
            param.requires_grad = not freeze

    def decide_joint_train_policy_if_applicable(self, epoch=None):
        if not (self.joint_training and self.iterative_optimization):
            return
        if epoch % 3 == 2:
            self.update_classifier_state(freeze=False)
            self.update_encoder_state(freeze=True)
        else:
            self.update_classifier_state(freeze=True)
            self.update_encoder_state(freeze=False)

    def train_1_epoch(self, epoch_num: int = 0):
        self.model.train()
        self.decide_joint_train_policy_if_applicable(epoch=epoch_num)
        for i in trange(len(self.train_db), leave=False):
            n_train_steps = i + len(self.train_db) * epoch_num
            loss, score = self.train_1_item(i, n_train_steps)
        x, target = self.train_db.__getitem__(0)
        if torch.cuda.is_available():
            x = x.cuda()

        self.summary_writer.add_graph(self.model, input_to_model=(x, target), verbose=False)

    def train_1_item(self, item_number: int, n_train_steps=None) -> float:
        x, target = self.train_db.__getitem__(item_number)
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()

        x = Variable(x)
 
        self.optimizer.zero_grad()

        result = self.model.forward(x, target)
        pred, w, w_orig = result.out, result.w_out, result.w_orig

        pred_labels = torch.argmax(w.data, dim=1).cpu().numpy()

        self.summary_writer.add_scalar('train_cross_entropy', log_loss(target.squeeze().detach().cpu().numpy(), w.detach().cpu().numpy()), global_step=n_train_steps)
        
        target = torch.squeeze(target).cpu()

        the_loss = self.calculate_loss(target, pred, w, w_orig, pred_labels, epoch=n_train_steps, record=True)
        the_loss.backward()
        self.optimizer.step()

        the_loss_tensor = the_loss.data
        if torch.cuda.is_available():
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        score = rand_score(target, pred_labels)
        self.summary_writer.add_scalar('train_rand_score', score, n_train_steps)

        return the_loss_float, score

    def calculate_pairwise_similarities(self, pred):
        if self.similarity_metric == 'cosine':
            sim_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            n_dim = pred.shape[1]
            arange = torch.arange(n_dim)
            pairs = torch.cartesian_prod(arange, arange)
            sims = sim_fn(pred[0,pairs[:,0]], pred[1,pairs[:,1]]).reshape(n_dim, -1)
        elif self.similarity_metric == 'gaussian_kernel':
            sim_fn = lambda x, y: torch.exp(-torch.cdist(x, y))
            sims = sim_fn(pred[0], pred[1])
        else:
            raise NotImplementedError
        return sims

    def calculate_contrastive_loss(self, pred):
        sims = self.calculate_pairwise_similarities(pred)
        exp_sims = torch.exp(sims)
        divisions = exp_sims.diagonal() / (exp_sims.sum(dim=-1) - exp_sims.diagonal())
        the_loss = -torch.log(divisions).sum()
        return the_loss

    def calculate_entropy_loss(self, w, w_orig, pred_labels, epoch=None, record=False):
        to_prob_dist = lambda dist: dist / dist.shape[0]
        neg_entropy = lambda dist: (dist * dist.log()).sum()
        normalized_neg_entropy = lambda dist: - neg_entropy(dist) / neg_entropy(to_prob_dist(torch.ones_like(dist)))
        # w_entropy = entropy(w)
        pred_dist = np.unique(pred_labels, return_counts=True)[1]
        pred_labels_entropy = normalized_neg_entropy(torch.tensor(pred_dist / pred_dist.sum()))
        w_entropy = normalized_neg_entropy(w.sum(dim=0)/ w.sum())
        w_orig_entropy = normalized_neg_entropy(w_orig.sum(dim=0)/ w_orig.sum())
        if record:
            self.summary_writer.add_scalars('norm_neg_entropy', {
                'w' : w_entropy,
                'pred_labels' : pred_labels_entropy,
                'w_orig': w_orig_entropy
            }, global_step=epoch)
        return pred_labels_entropy

    def calculate_loss(self, target, pred, w, w_orig, pred_labels, epoch=None, record=False):
        losses = {}
        loss = 0
        if self.experiment == 'classify':
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(w_orig, target.squeeze().to(w_orig.device))
            losses['cross-entropy'] = loss
            losses['total'] = loss
        else:
            contrastive_loss = self.calculate_contrastive_loss(pred)
            loss += contrastive_loss
            losses['contrastive'] = contrastive_loss
            if self.loss_type == 'contrastive_entropic_reg':
                entropy_loss = self.calculate_entropy_loss(w, w_orig, pred_labels, epoch=epoch, record=record)
                loss += 10 * entropy_loss
                losses['entropy'] = entropy_loss
            losses['total'] = loss
        if record:
            self.summary_writer.add_scalars('loss', losses, epoch)
        return loss

    def calculate_metrics(self, y_true, y_pred, total_losses=None, data=None, epoch=None,):
        self.summary_writer.add_scalars('loss', {f'{data}_total_eval' : total_losses.mean()}, epoch)
        if self.experiment == 'classify':
            metric = accuracy_score
        else:
            metric = rand_score
        score = metric(y_true.cpu().numpy(), y_pred.cpu().numpy())
        self.summary_writer.add_scalars('metric', {f'{data}_{metric.__name__}_eval': score}, epoch)

    def evaluate(self, epoch, data='val'):
        self.model.eval()
        if data == 'val':
            dataset = self.test_db
        elif data == 'train':
            dataset = self.train_db

        total_predictions = []
        total_targets = []
        total_losses = []
        cluster_input_centroids = [[] for i in range(self.model.rho.output_size)]
        x_idx = []
        scores = []
        cluster_representations = []
        for i in trange(len(dataset), leave=False):
            x, target = dataset.__getitem__(i)
            x_idx.append(dataset.mnist_items[i])

            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()

            result = self.model.forward(x, target)
            pred, w, w_orig = result.out, result.w_out, result.w_orig

            pred_labels = torch.argmax(w, dim=1).unsqueeze(-1)
            the_loss = self.calculate_loss(target, pred, w, w_orig, pred_labels)
            total_losses.append(the_loss.detach().cpu())

            for c in np.arange(10):
                cluster_input_centroids[c].append(x[pred_labels == c, ::].detach().cpu().numpy())

            cluster_representations.append(pred.detach().cpu())
            total_predictions.append(pred_labels.cpu())
            total_targets.append(target)
            scores.append(rand_score(target.cpu().squeeze(), pred_labels.cpu().squeeze()))

        total_predictions = torch.cat(total_predictions)
        total_targets = torch.cat(total_targets)
        total_losses = torch.hstack(total_losses)
        scores = np.hstack(scores)
        x_idx = np.hstack(x_idx)

        self.calculate_metrics(total_targets, total_predictions, total_losses=total_losses, epoch=epoch, data=data)

        embeddings_dataset = dataset.embeddings_umap
        target_labels_dataset = dataset.mnist_labels.numpy()
        self.plot_clustering_on_umap(embeddings_dataset, target_labels_dataset, label_type='Ground Truth', data=data, epoch=epoch)

        embeddings_set_dataset = dataset.embeddings_umap[x_idx]
        target_labels_set_dataset = dataset.mnist_labels[x_idx].numpy()
        pred_labels_set_dataset = total_predictions
        self.plot_clustering_on_umap(embeddings_set_dataset, target_labels_set_dataset, label_type=f'Ground Truth Set Dataset', data=data, epoch=epoch)
        self.plot_clustering_on_umap(embeddings_set_dataset, pred_labels_set_dataset, label_type=f'Predicted Set Dataset', data=data, epoch=epoch)

        # self.record_cluster_embeddings(cluster_representations, cluster_input_centroids=cluster_input_centroids, data=data, epoch=epoch)
        self.record_confusion_matrix(total_predictions, total_targets, epoch, data=data)

        torch.save(self.the_phi.state_dict(), self.checkpoints_dir / f'trained_phi_{epoch}.pt')
        torch.save(self.the_rho.state_dict(), self.checkpoints_dir / f'trained_rho_{epoch}.pt')
        torch.save(self.the_phi.state_dict(), self.checkpoints_dir / f'trained_phi_latest.pt')
        torch.save(self.the_rho.state_dict(), self.checkpoints_dir / f'trained_rho_latest.pt')

    def plot_clustering_on_umap(self, embeddings, labels, label_type=None, data='val', epoch=None):
        fig, ax = plt.subplots(figsize=(6,5))
        plt.title(f'Clustering under UMAP space ({label_type})')
        color = labels
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=color, cmap="Spectral", s=0.1)
        plt.legend(*scatter.legend_elements())
        image_path = self.figures_dir / f'{data}_umap_clusters_test_{label_type}_{epoch}.png'
        fig.savefig(image_path)
        self.record_image_tensorboard(image_path, f'umap_clusters_{label_type}', data=data, step=epoch)
        plt.close(fig)
        
    # def record_cluster_embeddings(self, cluster_representations, cluster_input_centroids=None, data='val', epoch=None):
    #     tsne = TSNE(n_components=2, random_state=0)
    #     X_2d = tsne.fit_transform(X)

    #     target_ids = range(10)
    #     fig, ax = plt.subplots(figsize=(6, 5))
    #     colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple'
    #     for i, c, label in zip(target_ids, colors, target_ids):
    #         x_coords = X_2d[Y == i, 0]
    #         y_coords = X_2d[Y == i, 1]
    #         ax.scatter(x_coords, y_coords, c=c, label=label)

    #         x_centroid = x_coords.mean()
    #         y_centroid = y_coords.mean()
    #         centroid_image = np.vstack(cluster_input_centroids[i]).mean(axis=0)
    #         centroid_image_unnormalized = centroid_image * MNIST_STD + MNIST_MEAN
    #         ab = AnnotationBbox(OffsetImage(centroid_image_unnormalized.squeeze()), (x_centroid, y_centroid), frameon=False)
    #         ax.add_artist(ab)

    #         # plt.scatter(X[Y == i, 0], X[Y == i, 1], c=c, label=label)

    #     ax.legend()
    #     image_path = self.figures_dir / f'{data}_tsne_{epoch}.png'
    #     fig.savefig(image_path)
    #     self.record_image_tensorboard(image_path, 'tsne_embeddings', data=data, step=epoch)
    #     plt.close(fig)

    def record_confusion_matrix(self, pred, target, epoch, data='val'):
        plt.figure()
        labels = range(10)
        matrix = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=labels)
        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='Blues')
        image_path = self.figures_dir / f'{data}_conf_matrix_{epoch}.png'
        plt.savefig(image_path)
        self.record_image_tensorboard(image_path, 'confusion_matrix', data=data)
        plt.close()

    def record_image_tensorboard(self, image_path, label, data='val', step=None):
        img = Image.open(image_path)
        img_tensor = transforms.ToTensor()(img)
        self.summary_writer.add_image(label, img_tensor, global_step=step)
        wandb.log({f'{data}_{label}' : wandb.Image(str(image_path)), 'step': step})
