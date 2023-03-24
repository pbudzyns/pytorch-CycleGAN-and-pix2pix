import argparse
import sys
from typing import List, Tuple

import torch.utils.benchmark as benchmark

from data import BaseDataset, create_dataset
from models import BaseModel, create_model
from options.train_options import TrainOptions


def run_train_loop(dataset: BaseDataset, model: BaseModel, opt: argparse.Namespace) -> None:
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.optimize_parameters()
        model.update_learning_rate()


def get_options(args: List[str]) -> argparse.Namespace:
    sys.argv.extend(args)  # fishy but works
    return TrainOptions().parse()


def generate_models_and_dataset(options: argparse.Namespace) -> Tuple[BaseModel, BaseDataset]:
    dataset = create_dataset(options)
    model = create_model(options)
    model.setup(options)
    return model, dataset


def get_args(
        batch_size: int = 1,
        n_epochs: int = 10,
        generator_network: str = 'animegan',
        model_name: str = 'cycle_gan_lpips',
) -> List[str]:
    return [
        '--dataroot', './datasets/benchmark',
        '--name', 'Benchmark',
        '--netG', generator_network,
        '--batch_size', str(batch_size),
        '--n_epochs', str(n_epochs),
        '--n_epochs_decay', str(n_epochs),
        '--model', model_name,
    ]


def benchmark_single_model(
        options: argparse.Namespace, description: str, threads_range: Tuple[int, ...] = (1, 4, 8, 16),
) -> List[benchmark.Measurement]:

    model, dataset = generate_models_and_dataset(options)
    results = []
    for num_threads in threads_range:
        results.append(
            benchmark.Timer(
                stmt='run_train_loop(dataset, model, opt)',
                setup='from __main__ import run_train_loop',
                globals={'dataset': dataset, 'model': model, 'opt': options},
                num_threads=num_threads,
                label='CycleGAN training',
                description=description,
            ).blocked_autorange(min_run_time=1)
        )
    return results


if __name__ == '__main__':

    options = get_options(get_args(batch_size=1))

    options.compile = False
    results_normal = benchmark_single_model(options, description='Normal', threads_range=(1, 2))

    options.compile = True
    results_compile = benchmark_single_model(options, description='Compiled', threads_range=(1, 2))

    results = results_normal + results_compile
    compare = benchmark.Compare(results)
    compare.print()
