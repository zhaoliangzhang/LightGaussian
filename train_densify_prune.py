#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
import os
import json
import yaml
import torch
import torchvision.transforms.functional as tf
import subprocess as sp
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from lpipsPyTorch import lpips
import torch.utils.benchmark as benchmark
from PIL import Image
from pathlib import Path
import hashlib

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.logger_utils import prepare_output_and_logger

from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, PruneParams

# from prune_train import prepare_output_and_logger, training_report
from icecream import ic
from os import makedirs
from prune import prune_list, calculate_v_imp_score
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from collections import defaultdict
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return memory_used_values

def training(
    dataset,
    opt,
    pipe,
    prune,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    wandb_enabled = WANDB_FOUND and args.use_wandb
    gaussians = GaussianModel(dataset, prune)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.97)

    net_training_time = 0
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            gaussians.scheduler.step()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if iteration == opt.densify_until_iter+1:
            gaussians.set_trainable_mask(opt)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, (iteration>opt.densify_until_iter) and prune.use_mask)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            iter_time = iter_start.elapsed_time(iter_end)
            net_training_time += iter_time
            if prune.use_mask:
                log_mask = torch.nn.Threshold(0.5, 0)(gaussians.get_mask)
                sparsity = 1 - torch.count_nonzero(log_mask).cpu().detach().numpy()/torch.numel(log_mask)
            else:
                sparsity = None
            training_report(wandb_enabled, iteration, Ll1, loss, 
            iter_time, net_training_time, scene, sparsity)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        iteration,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration in prune.prune_iterations:
                # TODO Add prunning types
                if prune.use_mask:
                    print("prune using mask at ", iteration)
                    prune_mask = gaussians.get_mask < 0.5
                    gaussians.prune_points(prune_mask.squeeze())
                else:
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    i = prune.prune_iterations.index(iteration)
                    v_list = calculate_v_imp_score(gaussians, imp_list, prune.v_pow)
                    gaussians.prune_gaussians(
                        (prune.prune_decay**i) * prune.prune_percent, v_list
                    )



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                if iteration == checkpoint_iterations[-1]:
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy()) 

    if wandb_enabled:
        wandb.run.summary['training_time'] = net_training_time/1000


    return net_training_time/1000

def training_report(wandb_enabled, iteration, Ll1, loss, 
                    iter_time, elapsed, scene : Scene, sparsity):

    if wandb_enabled:
        wandb.log({"train_loss_patches/l1_loss": Ll1.item(), 
                   "train_loss_patches/total_loss": loss.item(), 
                   "num_points": scene.gaussians.get_xyz.shape[0],
                   "iter_time": iter_time,
                   "elapsed": elapsed,
                   }, step=iteration)
        if sparsity != None:
            wandb.log({"sparsity": sparsity,
                       }, step = iteration)

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
        gaussians = GaussianModel(dataset, prune)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_vq= load_vq)
        print("NUM POINTS:", gaussians.get_xyz.shape[0])
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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
        wandb.log({"FPS": fps})

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
        fps = len(views)/render_time.median
    return fps
        


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

                wandb.log({"SSIM": torch.tensor(ssims).mean(),
                           "PSNR": torch.tensor(psnrs).mean(),
                           "LPIPS": torch.tensor(lpipss).mean()})
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update(
                    {
                        "SSIM": torch.tensor(ssims).mean().item(),
                        "PSNR": torch.tensor(psnrs).mean().item(),
                        "LPIPS": torch.tensor(lpipss).mean().item(),
                    }
                )
                per_view_dict[scene_dir][method].update(
                    {
                        "SSIM": {
                            name: ssim
                            for ssim, name in zip(
                                torch.tensor(ssims).tolist(), image_names
                            )
                        },
                        "PSNR": {
                            name: psnr
                            for psnr, name in zip(
                                torch.tensor(psnrs).tolist(), image_names
                            )
                        },
                        "LPIPS": {
                            name: lp
                            for lp, name in zip(
                                torch.tensor(lpipss).tolist(), image_names
                            )
                        },
                    }
                )

            with open(scene_dir + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    # Set up config file
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, config['model_params'])
    op = OptimizationParams(parser, config['opt_params'])
    pp = PipelineParams(parser, config['pipe_params'])
    prunp = PruneParams(parser, config['prune_params'])
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6012)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000],
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    lp_args = lp.extract(args)
    op_args = op.extract(args)
    pp_args = pp.extract(args)
    prunp_args = prunp.extract(args)
    
    # Prepare logger
    wandb_enabled=(WANDB_FOUND and lp_args.use_wandb)
    id = hashlib.md5(lp_args.wandb_run_name.encode('utf-8')).hexdigest()
    wandb.init(
        project=lp_args.wandb_project,
        name=lp_args.wandb_run_name,
        entity=lp_args.wandb_entity,
        group=lp_args.wandb_group,
        config=args,
        sync_tensorboard=False,
        dir=lp_args.model_path,
        mode=lp_args.wandb_mode,
        id=id,
        resume=True
    )

    training_time = training(
        lp_args,
        op_args,
        pp_args,
        prunp_args,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    full_dict = {args.model_path: {}}
    full_dict[args.model_path].update({"Training time": training_time})
    with open(os.path.join(args.model_path,"results_training.json"), 'w') as fp:
        json.dump(full_dict[args.model_path], fp, indent=True)

    if not args.skip_test:
        if os.path.exists(os.path.join(args.model_path,"results.json")) and not args.retest:
            print("Testing complete at {}".format(args.model_path))
        else:
            render_sets(lp_args, op_args.iterations, pp_args, prunp_args, args.skip_train, args.skip_test, False)
    
    evaluate([lp_args.model_path])

    # All done
    print("\nTraining complete.")
