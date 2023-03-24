import sys
import torch.utils.benchmark as benchmark

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions


def run_test_train_loop(dataset, model, opt):

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.update_learning_rate()
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.optimize_parameters()


def generate_datasets(args):
    sys.argv.extend(args)
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    return model, dataset, opt


def get_args(compiled: bool, batch_size: int = 1):
    args = [
        '--dataroot', './datasets/benchmark',
        '--name', 'Benchmark',
        '--netG', 'animegan',
        '--batch_size', str(batch_size),
        '--n_epochs', '10',
        '--n_epochs_decay', '0',
        '--model', 'cycle_gan_lpips'
    ]
    if compiled:
        args.append('--compile')
    return args


if __name__ == '__main__':
    results = []
    label = 'CycleGAN training'

    for num_threads in [1, 4, 8, 16]:
        results.append(benchmark.Timer(
            stmt='run_test_train_loop(args)',
            setup='from __main__ import run_test_train_loop',
            globals={'args': get_args(compiled=False)},
            num_threads=num_threads,
            label=label,
            description='Normal',
        ).blocked_autorange(min_run_time=1))
        results.append(benchmark.Timer(
            stmt='run_test_train_loop(args)',
            setup='from __main__ import run_test_train_loop',
            globals={'args': get_args(compiled=True)},
            num_threads=num_threads,
            label=label,
            description='Compiled',
        ).blocked_autorange(min_run_time=1))

    compare = benchmark.Compare(results)
    compare.print()
