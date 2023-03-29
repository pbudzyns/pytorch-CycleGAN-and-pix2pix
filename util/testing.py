import argparse
import copy
import pathlib
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

    test_opt = get_test_options(opt)
    run_test_image_generation(model, test_opt)
    result_image_paths = generate_grids_from_model_outputs(test_opt, epoch)
    return tuple(map(str, result_image_paths))


def run_test_image_generation(model: BaseModel, opt: argparse.Namespace) -> None:
    dataset = create_dataset(opt)
    web_dir = pathlib.Path(opt.results_dir) / opt.name
    webpage = html.HTML(
        web_dir,
        f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}",
    )

    for data in dataset:
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        save_images(webpage, visuals, img_path, aspect_ratio=1.0, width=256, use_wandb=False)


def generate_grids_from_model_outputs(
        opt: argparse.Namespace, epoch: Union[str, int],
) -> Tuple[pathlib.Path, ...]:
    model_results_root = pathlib.Path(opt.results_dir) / opt.name
    images_dir_path = model_results_root / "images"
    image_paths = tuple(pathlib.Path(images_dir_path).iterdir())

    result_plot_paths = tuple(
        get_plot_path(image_paths, model_results_root, epoch, opt, direction)
        for direction
        in ("AtoB", "BtoA")
    )

    return result_plot_paths


def get_plot_path(
        image_paths: Tuple[pathlib.Path, ...],
        results_root: pathlib.Path,
        epoch: Union[str, int],
        opt: argparse.Namespace,
        direction: str,
) -> pathlib.Path:
    domain_a, domain_b = direction.split("to")

    real_images = get_sorted_images_by_suffix(image_paths, f"real_{domain_a}")
    fake_images = get_sorted_images_by_suffix(image_paths,  f"fake_{domain_b}")
    reconstructed_images = get_sorted_images_by_suffix(image_paths, f"rec_{domain_a}")

    save_image_path = results_root / f"{epoch}_{opt.name}_{domain_a}.png"
    prepare_and_save_grid(real_images, fake_images, reconstructed_images, save_image_path)

    return save_image_path


def get_sorted_images_by_suffix(image_paths: Tuple[pathlib.Path, ...], suffix: str) -> List[pathlib.Path]:
    return sorted(
        [image for image in image_paths if suffix in image.name],
        key=lambda x: x.name.replace(f"_{suffix}", "")
    )


def prepare_and_save_grid(
        real_images: List[pathlib.Path],
        fake_images: List[pathlib.Path],
        reconstructed_images: List[pathlib.Path],
        filename: pathlib.Path,
) -> None:
    n_rows = 3
    n_cols = len(real_images)
    figure, (ax_real, ax_fake, ax_rec) = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols, n_rows))
    plot_axis_row(ax_real, real_images)
    plot_axis_row(ax_fake, fake_images)
    plot_axis_row(ax_rec, reconstructed_images)
    plt.subplots_adjust(wspace=0, hspace=0)
    figure.savefig(filename, dpi=200)
    plt.close()


def plot_axis_row(axes: List[plt.Axes], image_paths: List[pathlib.Path]) -> None:
    for i, image in enumerate(image_paths):
        axes[i].imshow(Image.open(image))
        axes[i].axis("off")


def get_test_options(opt: argparse.Namespace) -> argparse.Namespace:
    test_opt = copy.copy(opt)
    test_opt.phase = "test"
    test_opt.serial_batches = True
    test_opt.num_threads = 0
    test_opt.batch_size = 1
    test_opt.no_flip = True
    test_opt.results_dir = "./results/"
    test_opt.num_test = 50
    test_opt.aspect_ratio = 1.0
    test_opt.isTrain = False
    return test_opt
