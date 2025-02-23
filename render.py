#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
import torch.utils.benchmark as benchmark
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, PruneParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import sys
import yaml
from collections import defaultdict

import matplotlib.pyplot as plt

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    prune: PruneParams,
    skip_train: bool,
    skip_test: bool,
    load_vq: bool, 
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, prune)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_vq= load_vq)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(gaussians.get_xyz.shape[0])
        oc = gaussians.get_opacity
        oc = oc.detach().cpu().numpy()
        plt.hist(oc)
        plt.savefig('hist.png')

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )
        fps = measure_fps(scene, gaussians, pipeline, background, False)
        print({"FPS": fps})

def render_fn(views, gaussians, pipeline, background, use_amp):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        for view in views:
            render(view, gaussians, pipeline, background)

def measure_fps(scene, gaussians, pipeline, background, use_amp):
    with torch.no_grad():
        views = scene.getTrainCameras() + scene.getTestCameras()
        t0 = benchmark.Timer(stmt='render_fn(views, gaussians, pipeline, background, use_amp)',
                            setup='from __main__ import render_fn',
                            globals={'views': views, 'gaussians': gaussians, 'pipeline': pipeline, 
                                     'background': background, 'use_amp': use_amp},
                            )
        render_time = t0.timeit(50)
        print(len(views))
        print(render_time.median)
        fps = len(views)/render_time.median
    return fps


if __name__ == "__main__":
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    model = ModelParams(parser, config['model_params'])
    pipeline = PipelineParams(parser, config['pipe_params'])
    prune = PruneParams(parser, config['prune_params'])
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        prune.extract(args),
        args.skip_train,
        args.skip_test,
        args.load_vq
    )
