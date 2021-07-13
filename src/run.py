import random
import click
import numpy as np
import torch
from tqdm.auto import tqdm, trange
from deepsets.experiments import SumOfDigits
from deepsets.settings import RANDOM_SEED


@click.command()
@click.option('--seed', envvar='SEED', default=RANDOM_SEED, show_default=True)
@click.option('--classifier', envvar='CLASSIFIER', default='train', type=click.Choice(['oracle', 'train']))
@click.option('--encoder', envvar='ENCODER', default='train', type=click.Choice(['pretrained', 'finetune', 'train']))
@click.option('--model-path', envvar='MODEL_PATH', default='', type=str)
@click.option('--loss', envvar='LOSS', default='contrastive', type=click.Choice(['contrastive', 'contrastive_entropic_reg']))
def main(seed, classifier, encoder, model_path, loss):
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    the_experiment = SumOfDigits(lr=1e-3, dsize=100, set_size=1000, classifier_type=classifier, encoder_type=encoder, model_path=model_path, loss_type=loss)

    # for i in range(20):
    for i in trange(20):
        the_experiment.train_1_epoch(i)
        the_experiment.evaluate(i)

if __name__ == '__main__':
    main()
