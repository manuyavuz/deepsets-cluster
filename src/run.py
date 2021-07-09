import click
import numpy as np
import torch

from deepsets.experiments import SumOfDigits
from deepsets.settings import RANDOM_SEED


@click.command()
@click.option('--random-seed', envvar='SEED', default=RANDOM_SEED)
def main(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    the_experiment = SumOfDigits(lr=1e-3, dsize=100, set_size=1000)

    # for i in range(20):
    for i in range(20):
        the_experiment.train_1_epoch(i)
        the_experiment.evaluate(i)
        # torch.save(the_experiment.the_phi.state_dict(), 'trained_phi.pkl')
        # torch.save(the_experiment.the_rho.state_dict(), 'trained_rho.pkl')


if __name__ == '__main__':
    main()
