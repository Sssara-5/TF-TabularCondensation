import click
import json
from pprint import pformat
import torch


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        return pformat(self.__dict__, compact=True)
    
def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)

@click.command()
@click.option('--dataset', default='Covertype', show_default=True, help='Dataset name.')
@click.option('--method', default='Ours', show_default=True, help='Condensation method name.')
@click.option('--eval_model',default='ft_transformer', show_default=True, help='Model type (e.g., MLP).')  
@click.option('--reduction_rate', default=0.01, type=float, show_default=True,
              help='reduction rate (eg, 1%, 5%, 10%).')
@click.option('--categorical_method', default="autoencoder",  show_default=True,
              help='pca or autoencoder')
@click.option('--num_exp', default=5, type=int, show_default=True, help='Number of experiments to run.')
@click.option('--stepwise', default='0.5', show_default=True,
              help='step decrease.')
@click.option('--gamma', default=1, type=float, show_default=True,
              help='penalty in our method (eg, 1%, 0.7%, 0.5%).')
@click.option('--epoch_eval_train', default=100, type=int, show_default=True, 
              help='Epochs to train a model with synthetic data (can be small for speed).')
@click.option('--lr_net', default=0.001, type=float, show_default=True,
              help='Learning rate for training the network in evaluate_tabular_data.')
@click.option('--device', default='0', show_default=True,
              help='select GPU.')


@click.pass_context

def cli(ctx, **kwargs):
    args = dict2obj(kwargs)

    device_id = int(args.device) 
    assert device_id < torch.cuda.device_count(), f"Invalid device ID {device_id}"
    args.device = torch.device(f"cuda:{device_id}")

    return args

def get_args():
    return cli(standalone_mode=False)

if __name__ == '__main__':
    cli()