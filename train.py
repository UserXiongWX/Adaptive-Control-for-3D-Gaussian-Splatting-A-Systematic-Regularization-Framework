import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import lpips
loss_fn_alex = lpips.LPIPS(net='vgg')


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,normal_lr, finetune, load_iteration):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, normal_lr=normal_lr, finetune=finetune)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        first_iter = 0
        gaussians.restore(model_params, opt, finetune)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    first_iter += 1
    loss_fn_alex.to(gaussians.get_features.device)
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss_render = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_render

        loss_adaptive_opacity_val_for_log = 0.0
        if opt.lambda_adaptive_opacity > 0 and iteration >= opt.adaptive_opacity_start_iter:
            if iteration == opt.adaptive_opacity_start_iter:
                gaussians.set_adaptive_opacity_params(opt.smooth_region_quantile)
                print(f"[Iteration {iteration}]  Enabled adaptive opacity consistency loss")
            
            loss_adaptive_opacity = gaussians.get_adaptive_opacity_consistency_loss(visibility_filter, radii)
            if loss_adaptive_opacity.item() > 0:
                loss = loss + opt.lambda_adaptive_opacity * loss_adaptive_opacity
                loss_adaptive_opacity_val_for_log = loss_adaptive_opacity.item()

        loss_selective_normal_val_for_log = 0.0
        if opt.lambda_selective_normal > 0 and iteration >= opt.selective_normal_start_iter:
            if iteration == opt.selective_normal_start_iter:
                gaussians.set_selective_normal_params(opt.interior_region_quantile)
                print(f"[Iteration {iteration}]  Enabled selective normal smoothness loss")
            
            loss_selective_normal = gaussians.get_selective_normal_smoothness_loss(
                opt.knn_k_normal, iteration, visibility_filter, radii)
            
            if loss_selective_normal.item() > 0:
                loss = loss + opt.lambda_selective_normal * loss_selective_normal
                loss_selective_normal_val_for_log = loss_selective_normal.item()

        loss_trusted_normal_val_for_log = 0.0
        if opt.lambda_trusted_normal > 0 and iteration >= opt.trusted_normal_from_iter:
            current_normals_activated = F.normalize(gaussians._normal, p=2, dim=1)
            
            if gaussians._anchor_normal is None or gaussians._anchor_normal.shape != current_normals_activated.shape:
                print(f"[Iteration {iteration}] Initializing anchor normals for trusted normal constraint.")
                gaussians._anchor_normal = current_normals_activated.detach().clone()

            if current_normals_activated.numel() > 0 and gaussians._anchor_normal.numel() > 0:
                anchor_normals_normalized = F.normalize(gaussians._anchor_normal, p=2, dim=1)                
                dot_products_anchor = torch.sum(current_normals_activated * anchor_normals_normalized, dim=1)
                loss_trusted_normal = (1.0 - dot_products_anchor).pow(2).mean()
                loss = loss + opt.lambda_trusted_normal * loss_trusted_normal
                loss_trusted_normal_val_for_log = loss_trusted_normal.item()

                with torch.no_grad():
                    gaussians._anchor_normal.data = opt.ema_alpha_normal_anchor * anchor_normals_normalized.data + \
                                                (1.0 - opt.ema_alpha_normal_anchor) * current_normals_activated.data
                    gaussians._anchor_normal.data = F.normalize(gaussians._anchor_normal.data, p=2, dim=1)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('train_loss_parts/render_loss', loss_render.item(), iteration)
                
                if opt.lambda_adaptive_opacity > 0 and iteration >= opt.adaptive_opacity_start_iter:
                    tb_writer.add_scalar('train_loss_parts/adaptive_opacity_consistency', loss_adaptive_opacity_val_for_log, iteration)
                
                if opt.lambda_selective_normal > 0 and iteration >= opt.selective_normal_start_iter:
                    tb_writer.add_scalar('train_loss_parts/selective_normal_smoothness', loss_selective_normal_val_for_log, iteration)
                
                if opt.lambda_trusted_normal > 0 and iteration >= opt.trusted_normal_from_iter:
                    tb_writer.add_scalar('train_loss_parts/trusted_normal_constraint', loss_trusted_normal_val_for_log, iteration)
                    
                tb_writer.add_scalar('train_loss_total/total_loss_with_reg', loss.item(), iteration)

            if iteration in testing_iterations or (iteration == opt.iterations and opt.iterations not in testing_iterations):
                training_report(tb_writer, iteration, Ll1.item(), loss.item(), l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                grads = gaussians.xyz_gradient_accum / gaussians.denom
                grads[grads.isnan()] = 0.0
                np.save(os.path.join(scene.model_path, "tensor_data.npy"), grads.cpu().numpy())
                
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % 5000==0:
                if finetune:
                    gaussians.optimizer.param_groups[1]['lr'] /= 1.5
                else:
                    gaussians.optimizer.param_groups[4]['lr'] /= 1.4

            if iteration < opt.densify_until_iter and not finetune:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold, iteration)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, total_loss, l1_loss_func, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss_with_reg', total_loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_alex.forward(image, gt_image).squeeze()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test,ssim_test,lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        print(f"Number of point: {len(scene.gaussians._xyz)}")
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000,18_000, 30_000])

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--normal_lr", type=float, default = 0.003)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--load_iteration',type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    load_iteration = args.load_iteration
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.normal_lr, args.finetune, load_iteration)
    print("\nTraining complete.")