import torch
import argparse
import pandas as pd
import sys

from nerf.provider import NeRFDataset
from nerf.utils import *

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    class LoadFromFile(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            with values as f:
                # parse arguments in the file and store them in the target namespace
                parser.parse_args(f.read().split(), namespace)
    # to change
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_interval', type=int, default=5, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--test_interval', type=int, default=25,
                        help="test on the test set every interval epochs")
    parser.add_argument('--real_strength', type=float, default=0.5, help="ignore the image when 1")
    parser.add_argument('--real_cfg', type=float, default=7.5, help="image2image cfg")
    parser.add_argument('--sds_weight', type=float, default=1.0, help="init sds weight")
    parser.add_argument('--sds_decay_begin_step', type=int, default=0)
    parser.add_argument('--sds_decay_step', type=int, default=1)
    parser.add_argument('--sds_weight_end', type=float, default=1.0, help="final sds weight")

    parser.add_argument('--l2_weight', type=float, default=0.0, help="init l2 weight")
    parser.add_argument('--l2_decay_begin_step', type=int, default=0)
    parser.add_argument('--l2_decay_step', type=int, default=1)
    parser.add_argument('--l2_weight_end', type=float, default=0.0, help="final l2 weight")
    parser.add_argument('--gen_interval', type=int, default=10, help="image2image cfg")


    parser.add_argument('--anneal_step', type=int, default=-1, help="time anneal")
    parser.add_argument('--anneal_max_step', type=int, default=980, help="time anneal max step")

    # parser.add_argument('--d_batch_size', type=int, default=1, help="d batch_size")

    parser.add_argument('--real_overwrite', action='store_true', help="overwrite real image dir")
    parser.add_argument('--tet_grid_size', type=int, default=128, help="tet grid size")
    parser.add_argument('--normal_net', action='store_true',default=True, help="normal_net")

    parser.add_argument('--per_view_gt', type=int, default=5, help="per view real image")
    parser.add_argument('--save_interval', type=int, default=50, help="save interval")
    parser.add_argument('--d_reg_interval', type=int, default=16, help="discriminator regularization")
    parser.add_argument('--iters', type=int, default=30000, help="total training iters")
    parser.add_argument('--cam_drop_prob', type=float, default=0.2, help="camera drop rate")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--d_lr', type=float, default=2e-3, help="d learning rate")
    parser.add_argument('--gamma', type=float, default=1.0, help="D reg")
    parser.add_argument('--max_steps', type=int, default=384,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--file', type=open, action=LoadFromFile, help="specify a file filled with more arguments")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--p2p_text', default=None, help="text prompt")

    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--lora_dir', default=None, type=str, help="negative text prompt")

    parser.add_argument('-O', action='store_true', help="IT3D default setting")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--g_bkg_network', action='store_true', help="always render g network")
    parser.add_argument('--six_views', action='store_true', help="six_views mode: save the images of the six views")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--real_save_path', type=str, default=None)
    parser.add_argument('--nerf', type=str, default='l2', choices=['l1', 'l2', 'l3','l4','l5'],
                        help="nerf backbone level")
    parser.add_argument('--seed', default=None)

    parser.add_argument('--max_keep_ckpt', type=int, default=5000,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--image', default=None, help="image prompt")
    parser.add_argument('--image_config', default=None, help="image config csv")

    parser.add_argument('--known_view_interval', type=int, default=4,
                        help="train default view with RGB loss every & iters, only valid if --image is not None.")

    parser.add_argument('--IF', action='store_true',
                        help="experimental: use DeepFloyd IF as the guidance model for nerf stage")
    parser.add_argument('--gan', action='store_true', help="gan mode")

    parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    parser.add_argument('--guidance_scale', type=float, default=100,
                        help="diffusion model classifier-free guidance scale")
    parser.add_argument('--sdf_lr_scale', type=float, default=1.0, help="will be plused to lr")
    parser.add_argument('--deform_lr_scale', type=float, default=1.0, help="will be plused to lr")
    parser.add_argument('--no_cam_D', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=5e4, help="target face number for mesh decimation")

    parser.add_argument('--dmtet', action='store_true', help="use dmtet finetuning")
    parser.add_argument('--freeze_geo_step', type=int, default = 0, help="freeze dmtet geometry")

    parser.add_argument('--init_with', type=str, default='', help="ckpt to init dmtet")
    parser.add_argument('--lock_geo', action='store_true', help="disable dmtet to learn geometry")
    parser.add_argument('--init_color', action='store_true', help="init dmtet color")

    ## Perp-Neg options

    parser.add_argument('--perpneg', action='store_true', help="use perp_neg")
    parser.add_argument('--negative_w', type=float, default=-2,
                        help="The scale of the weights of negative prompts. A larger value will help to avoid the Janus problem, but may cause flat faces. Vary between 0 to -4, depending on the prompt")
    parser.add_argument('--front_decay_factor', type=float, default=2, help="decay factor for the front prompt")
    parser.add_argument('--side_decay_factor', type=float, default=10, help="decay factor for the side prompt")

    ### training options
    parser.add_argument('--ckpt', type=str, default='scratch',
                        help="possible options are ['latest', 'scratch', 'best', 'latest_model']")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--taichi_ray', action='store_true', help="use taichi raymarching")
    parser.add_argument('--num_steps', type=int, default=64,
                        help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32,
                        help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--latent_iter_ratio', type=float, default=-1,
                        help="training iters that only use albedo shading")
    parser.add_argument('--albedo_iter_ratio', type=float, default=0,
                        help="training iters that only use albedo shading")
    parser.add_argument('--min_ambient_ratio', type=float, default=0.1,
                        help="minimum ambient ratio to use in lambertian shading")
    parser.add_argument('--textureless_ratio', type=float, default=0.2, help="ratio of textureless shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--jitter_center', type=float, default=0.2,
                        help="amount of jitter to add to sampled camera pose's center (camera location)")
    parser.add_argument('--jitter_target', type=float, default=0.2,
                        help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
    parser.add_argument('--jitter_up', type=float, default=0.02,
                        help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0,
                        help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--grad_clip', type=float, default=-1,
                        help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_rgb', type=float, default=-1,
                        help="clip grad of rgb space grad to this limit, negative value disables it")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4,
                        help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='exp', choices=['softplus', 'exp'],
                        help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.2, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--backbone', type=str, default='grid', choices=['grid_tcnn', 'grid', 'vanilla', 'grid_taichi'],
                        help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    parser.add_argument('--w', type=int, default=512, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=512, help="render height for NeRF in training")
    parser.add_argument('--known_view_scale', type=float, default=1.5,
                        help="multiply --h/w by this for known view rendering")
    parser.add_argument('--known_view_noise_scale', type=float, default=2e-3,
                        help="random camera noise added to rays_o and rays_d")
    parser.add_argument('--dmtet_reso_scale', type=float, default=1, help="multiply --h/w by this for dmtet finetuning")
    parser.add_argument('--batch_size', type=int, default=1, help="images to render per batch using NeRF")
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.0, 3.5],
                        help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105],
                        help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180],
                        help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[10, 30], help="training camera fovy range")

    parser.add_argument('--default_radius', type=float, default=3.2, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")

    parser.add_argument('--progressive_view', action='store_true',
                        help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_view_init_ratio', type=float, default=0.2,
                        help="initial ratio of final range, used for progressive_view")

    parser.add_argument('--progressive_level', action='store_true',
                        help="progressively increase gridencoder's max_level")

    parser.add_argument('--no_random_bkg', action='store_true',
                        help=" ")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98],
                        help="stable diffusion time steps range")
    parser.add_argument('--dont_override_stuff', action='store_true', help="Don't override t_range, etc.")

    ### regularizations
    parser.add_argument('--g_loss_weight', type=float, default=1.0)
    parser.add_argument('--g_loss_decay_begin_step', type=int, default=25000)
    parser.add_argument('--g_loss_decay_step', type=int, default=2500)
    parser.add_argument('--g_loss_weight_end', type=float, default=0.0)


    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")

    parser.add_argument('--lambda_mesh_normal', type=float, default=0.5, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=0.5, help="loss scale for mesh laplacian")

    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=1000, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=500, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=10, help="loss scale for relative depth")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, default=0,
                        help="loss scale for 2D normal image smoothness")
    parser.add_argument('--lambda_3d_normal_smooth', type=float, default=0,
                        help="loss scale for 3D normal image smoothness")
    ### debugging options
    parser.add_argument('--save_guidance', action='store_true',
                        help="save images of the per-iteration NeRF renders, added noise, denoised (i.e. guidance), fully-denoised. Useful for debugging, but VERY SLOW and takes lots of memory!")
    parser.add_argument('--save_guidance_interval', type=int, default=10, help="save guidance every X step")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=20, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60,
                        help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--zero123_config', type=str,
                        default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml',
                        help="config file for zero123")
    parser.add_argument('--zero123_ckpt', type=str, default='./pretrained/zero123/105000.ckpt', help="ckpt for zero123")
    parser.add_argument('--zero123_grad_scale', type=str, default='angle',
                        help="whether to scale the gradients based on 'angle' or 'None'")

    parser.add_argument('--dataset_size_train', type=int, default=100,
                        help="Length of train dataset i.e. # of iterations per epoch")
    parser.add_argument('--dataset_size_valid', type=int, default=8,
                        help="# of frames to render in the turntable video in validation")
    parser.add_argument('--dataset_size_test', type=int, default=120,
                        help="# of frames to render in the turntable video at test time")

    parser.add_argument('--img2img_view_num', type=int, default=64,
                        help="view num for img2img dataset, should be the multiple of dataset_size_valid for visualization")

    parser.add_argument('--exp_start_iter', type=int, default=None,
                        help="start iter # for experiment, to calculate progressive_view and progressive_level")
    parser.add_argument('--exp_end_iter', type=int, default=None,
                        help="end iter # for experiment, to calculate progressive_view and progressive_level")

    # ctn
    parser.add_argument('--ctn', action='store_true', help="use ctn as img2img, default using softedge conditioning")
    parser.add_argument('--ctn_model_dir', type=str, default='ctn_models', help="dir to save ctn models")
    parser.add_argument('--depth', action='store_true', help="ctn depth conditioning")
    parser.add_argument('--normal', action='store_true', help="ctn normal conditioning")
    parser.add_argument('--ctn_sample_batch_size', type=int, default=4, help="ctn sample batch_size")

    parser.add_argument('--ctn_seed', type=int, default=1, help="seed for CTN")
    parser.add_argument('--det', type=str, default="SoftEdge_PIDI",
                        help="softedge det")  # ["SoftEdge_PIDI", "SoftEdge_PIDI_safe", "SoftEdge_HED", "SoftEdge_HED_safe", "None"]
    parser.add_argument('--image_resolution', type=int, default=512)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--guess_mode', action='store_true')

    parser.add_argument('--detect_resolution', type=int, default=512)
    parser.add_argument('--ddim_steps', type=int, default=20)
    parser.add_argument('--ctn_cfg', type=float, default=9.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--a_prompt', type=str,
                        default='3D effect, stereoscopy, plain background, solid color background')
    parser.add_argument('--n_prompt', type=str,
                        default='lowres, bad anatomy, bad hands, cropped, worst quality, strong light, bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses. blur')

    opt = parser.parse_args()

    if opt.test:
        opt.max_steps = 1024

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.perpneg = True
        opt.g_bkg_network = True


    if opt.IF:
        if 'SD' in opt.guidance:
            opt.guidance.remove('SD')
            opt.guidance.append('IF')
        opt.latent_iter_ratio = 0  # must not do as_latent

    opt.images, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
    opt.default_zero123_w = 1

    opt.exp_start_iter = opt.exp_start_iter or 0
    opt.exp_end_iter = opt.exp_end_iter or opt.iters

    # reset to None
    if len(opt.images) == 0:
        opt.images = None

    # default parameters for finetuning
    if opt.dmtet:
        opt.h = int(opt.h * opt.dmtet_reso_scale)
        opt.w = int(opt.w * opt.dmtet_reso_scale)
        opt.known_view_scale = 1

        if not opt.dont_override_stuff:  # T
            opt.t_range = [0.02, 0.50]  # ref: magic3D

        if opt.images is not None:  # F

            opt.lambda_normal = 0
            opt.lambda_depth = 0

            if opt.text is not None and not opt.dont_override_stuff:
                opt.t_range = [0.20, 0.50]

        # assume finetuning
        opt.latent_iter_ratio = 0
        opt.albedo_iter_ratio = 0
        opt.progressive_view = False
        # opt.progressive_level = False

    # record full range for progressive view expansion
    if opt.progressive_view:
        if not opt.dont_override_stuff:
            # disable as they disturb progressive view
            opt.jitter_pose = False

        opt.uniform_sphere_rate = 0
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork, GANNeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork

        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.gan:
        model = GANNeRFNetwork(opt).to(device)
    else:
        model = NeRFNetwork(opt).to(device)

    if opt.dmtet and opt.init_with != '':  # T
        if opt.init_with.endswith('.pth'):  # T
            # load pretrained weights to init dmtet
            state_dict = torch.load(opt.init_with, map_location=device)
            model.load_state_dict(state_dict['model'], strict=False)
            if opt.cuda_ray:
                model.mean_density = state_dict['mean_density']
            model.init_tet()
            if opt.init_color:
                model.init_color()
        else:
            # assume a mesh to init dmtet (experimental, not working well now!)
            import trimesh

            mesh = trimesh.load(opt.init_with, force='mesh', skip_material=True, process=False)
            model.init_tet(mesh=mesh)

    print(model)

    if opt.six_views:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace,
                          fp16=opt.fp16, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset(opt, device=device, type='six_views', H=opt.H, W=opt.W, size=6).dataloader(
            batch_size=1)
        trainer.test(test_loader, write_video=False)

        if opt.save_mesh:
            trainer.save_mesh()

    elif opt.test:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace,
                          fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            from nerf.gui import NeRFGUI

            gui = NeRFGUI(opt, trainer)
            gui.render()
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=800, W=800,
                                      size=120).dataloader(batch_size=1)
            trainer.test(test_loader)

            trainer.save_mesh()

    else:

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w,
                                   size=opt.dataset_size_train * opt.batch_size).dataloader()

        if opt.optim == 'adan':  # T
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr, opt.sdf_lr_scale, opt.deform_lr_scale),
                                           eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':  # F
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                      lambda iter: 0.1 ** min(iter / opt.iters, 1))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)  # fixed
            # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        guidance = nn.ModuleDict()

        if 'SD' in opt.guidance:
            from guidance.sd_utils import StableDiffusion

            guidance['SD'] = StableDiffusion(opt,device, opt.fp16, opt.vram_O, opt.gan, opt.sd_version, opt.hf_key,
                                             opt.t_range)

        if opt.gan:
            d_loader = NeRFDataset(opt, device=device, type='d', H=opt.h, W=opt.w, size=10000000, view_num=opt.img2img_view_num).dataloader(
                batch_size=opt.batch_size)
            trainer = GanTrainer(' '.join(sys.argv), 'df', opt, model, guidance, d_loader, device=device,
                                 workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16,
                                 lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)
        else:
            trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace,
                              optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                              use_checkpoint=opt.ckpt, scheduler_update_every_step=True)

        trainer.default_view_data = train_loader._data.get_default_view_data()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W,
                                   size=opt.dataset_size_valid).dataloader(batch_size=1)
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W,
                                  size=opt.dataset_size_test).dataloader(batch_size=1)
        if opt.gan:
            img2img_loader = NeRFDataset(opt, device=device, type='test', H=512, W=512,
                                      size=opt.img2img_view_num).dataloader(batch_size=1) # img2img at 512 resolution

            trainer.train(train_loader, valid_loader, test_loader, img2img_loader, max_epoch)
        else:
            trainer.train(train_loader, valid_loader, test_loader, max_epoch)

        if opt.save_mesh:
            trainer.save_mesh()
