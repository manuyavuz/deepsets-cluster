import random
import click
import numpy as np
import torch
from tqdm.auto import tqdm, trange
from deepsets.experiments import SumOfDigits
from deepsets.settings import RANDOM_SEED

@click.command()
@click.option('--lr', envvar='LR', default=1e-3, show_default=True)
@click.option('--wd', envvar='WD', default=5e-3, show_default=True)
@click.option('--n_epochs', envvar='N_EPOCHS', default=20, show_default=True)
@click.option('--n_sets', envvar='N_SETS', default=100, show_default=True)
@click.option('--set_size', envvar='SET_SIZE', default=1000, show_default=True)
@click.option('--seed', envvar='SEED', default=RANDOM_SEED, show_default=True)
@click.option('--classifier', envvar='CLASSIFIER', default='train', type=click.Choice(['oracle', 'train']))
@click.option('--encoder', envvar='ENCODER', default='train', type=click.Choice(['pretrained', 'finetune', 'train']))
@click.option('--model-path', envvar='MODEL_PATH', default='', type=str)
@click.option('--loss', envvar='LOSS', default='contrastive', type=click.Choice(['contrastive', 'contrastive_entropic_reg']))
@click.option('--normalize-weights', envvar='NORMALIZE_WEIGHTS', default=True, type=bool)
@click.option('--normalize-weights-for-predictions', envvar='NORMALIZE_WEIGHTS_FOR_PREDICTIONS', default=False, type=bool)
def main(seed, **kwargs):
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    the_experiment = SumOfDigits(**kwargs)

    for i in trange(kwargs['n_epochs']):
        the_experiment.train_1_epoch(i)
        the_experiment.evaluate(i)

if __name__ == '__main__':
    main()
