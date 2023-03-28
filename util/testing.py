import argparse
import copy
import pathlib
import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
from PIL import Image

from data import create_dataset
from models import BaseModel
from util import html
from util.visualizer import save_images


def generate_test_images_during_training(
        model: BaseModel, opt: argparse.Namespace, epoch: Union[int, str],
) -> Tuple[str, ...]:

    test_opt = update_options(opt)
    run_test_image_generation(model, test_opt)
    result_image_paths = compose_images_to_grids(test_opt, epoch)
    return tuple(str(path) for path in result_image_paths)


def run_test_image_generation(model: BaseModel, opt: argparse.Namespace) -> None:
    dataset = create_dataset(opt)
    web_dir = os.path.join(
        opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    webpage = html.HTML(
        web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        tmp_visuals = model.visual_names
        model.visual_names = ["real_A", "fake_B", "real_B", "fake_A"]  # workaround
        visuals = model.get_current_visuals()  # get image results
        model.visual_names = tmp_visuals
        img_path = model.get_image_paths()  # get image paths
        save_images(webpage, visuals, img_path, aspect_ratio=1.0, width=256, use_wandb=False)


def compose_images_to_grids(
        opt: argparse.Namespace, epoch: Union[str, int]) -> Tuple[pathlib.Path, pathlib.Path]:
    model_results_root = pathlib.Path(opt.results_dir) / opt.name
    images_path = model_results_root / "test_latest/images"
    images = tuple(pathlib.Path(images_path).iterdir())
    image_path_A = prepare_plot(images, model_results_root, epoch, opt, "AtoB")
    image_path_B = prepare_plot(images, model_results_root, epoch, opt, "BtoA")

    return image_path_A, image_path_B


def prepare_plot(
        images: Tuple[pathlib.Path, ...],
        results_root: pathlib.Path,
        epoch: Union[str, int],
        opt: argparse.Namespace,
        direction: str,
) -> pathlib.Path:
    domain_A, domain_B = direction.split("to")
    real_images = get_sorted_images_by_suffix(images, f"real_{domain_A}")
    fake_images = get_sorted_images_by_suffix(images, f"fake_{domain_B}")
    image_path = results_root / f"{epoch}_{opt.name}_{direction}.png"
    plot_and_save_grid(real_images, fake_images, image_path)
    return image_path


def get_sorted_images_by_suffix(images: Tuple[pathlib.Path, ...], suffix: str) -> List[pathlib.Path]:
    return sorted(
        [image for image in images if suffix in image.name],
        key=lambda x: x.name.replace(f"_{suffix}", '')
    )


def plot_and_save_grid(
        real_images: List[pathlib.Path], fake_images: List[pathlib.Path], filename: pathlib.Path) -> None:
    n_cols = len(real_images)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=n_cols, figsize=(n_cols, 2))
    for i, image in enumerate(real_images):
        img = Image.open(image)
        ax1[i].imshow(img)
        ax1[i].axis('off')

    for i, image in enumerate(fake_images):
        img = Image.open(image)
        ax2[i].imshow(img)
        ax2[i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(filename, dpi=200)


def update_options(opt: argparse.Namespace) -> argparse.Namespace:
    test_opt = copy.copy(opt)
    test_opt.phase = 'test'
    test_opt.serial_batches = True
    test_opt.num_threads = 0
    test_opt.batch_size = 1
    test_opt.no_flip = True
    test_opt.results_dir = "./results/"
    test_opt.num_test = 50
    test_opt.aspect_ratio = 1.0
    test_opt.model = "test"
    test_opt.isTrain = False
    return test_opt
