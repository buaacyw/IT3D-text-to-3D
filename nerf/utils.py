import os
import glob
import tqdm
import math
import psutil
from pathlib import Path
import random
import tensorboardX
import numpy as np
from torch_utils.ops import conv2d_gradfix
import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torchmetrics import PearsonCorrCoef
from torch.utils.data import Dataset,DataLoader

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import wandb
import PIL
import shutil
class ImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = [x for x in image_path if 'png' in x]
        self.size = len(self.image_path)
        self.view_index_list = []
        for image_path in self.image_path:
            view_index = int(image_path.split('/')[-1].split('_')[0])
            self.view_index_list.append(view_index)
        self.view_num = self.view_index_list[-1]+1
        new_index_list = [x/self.view_num for x in self.view_index_list ] # [0, 1]
        self.view_index_list = new_index_list
        self.per_view_num = self.size // self.view_num
    def __len__(self):
        return len(self.image_path)

    def get_view(self,view_index):
        idx=-1
        for list_idx, idx_value in enumerate(self.view_index_list):
            if abs(idx_value - view_index.item()) < 1e-5:
                idx = list_idx
                break
        assert idx!=-1
        idx += random.randint(0,self.per_view_num-1)

        cur_path = self.image_path[idx]
        view_index = self.view_index_list[idx]
        cur_image = PIL.Image.open(cur_path)
        image = np.array(cur_image, dtype=np.float32)  # (h,w,c)
        image /= 255.0  #
        image = np.transpose(image, (2, 0, 1))  # (c, h, w)
        image = torch.from_numpy(image)  # numpy->tensor
        image = F.interpolate(image[None], (512, 512), mode='bilinear', align_corners=False)[0]
        return image


    def __getitem__(self, idx):
        cur_path = self.image_path[idx]
        view_index = self.view_index_list[idx]
        cur_image = PIL.Image.open(cur_path)
        image = np.array(cur_image, dtype=np.float32)  # (h,w,c)
        image /= 255.0  #
        image = np.transpose(image, (2, 0, 1))  # (c, h, w)
        image = torch.from_numpy(image)  # numpy->tensor
        image = F.interpolate(image[None], (512, 512), mode='bilinear', align_corners=False)[0]

        return image, view_index


def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images
def adjust_text_embeddings(embeddings, azimuth, opt):
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)



class Trainer(object):
    def __init__(self,
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=False, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = opt.max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if self.opt.images is not None:
            self.pearson = PearsonCorrCoef().to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            param = list(self.model.named_parameters())
            new_param = []
            for tup in param:
                if 'D' not in tup[0]:
                    new_param.append(tup[1])
            self.ema = ExponentialMovingAverage(new_param, decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

            # Save a copy of image_config in the experiment workspace
            if opt.image_config is not None:
                shutil.copyfile(opt.image_config, os.path.join(self.workspace, os.path.basename(opt.image_config)))

            # Save a copy of images in the experiment workspace
            if opt.images is not None:
                for image_file in opt.images:
                    shutil.copyfile(image_file, os.path.join(self.workspace, os.path.basename(image_file)))

        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] opt: {self.opt}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        # if not self.opt.test:
        wandb.login()
        wandb.init(
            project="IT3D",
            name=self.workspace,
            # entity='3d-gan',
            config={
                    'opt': self.opt,
                    },
            settings=wandb.Settings(code_dir=os.getcwd()),
        )

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:

            if 'SD' in self.guidance:
                self.embeddings['SD']['default'] = self.guidance['SD'].get_text_embeds([self.opt.text])
                self.embeddings['SD']['uncond'] = self.guidance['SD'].get_text_embeds([self.opt.negative])

                for d in ['front', 'side', 'back']:
                    self.embeddings['SD'][d] = self.guidance['SD'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'IF' in self.guidance:
                self.embeddings['IF']['default'] = self.guidance['IF'].get_text_embeds([self.opt.text])
                self.embeddings['IF']['uncond'] = self.guidance['IF'].get_text_embeds([self.opt.negative])

                for d in ['front', 'side', 'back']:
                    self.embeddings['IF'][d] = self.guidance['IF'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'clip' in self.guidance:
                self.embeddings['clip']['text'] = self.guidance['clip'].get_text_embeds(self.opt.text)

        if self.opt.images is not None:

            h = int(self.opt.known_view_scale * self.opt.h)
            w = int(self.opt.known_view_scale * self.opt.w)

            # load processed image
            for image in self.opt.images:
                assert image.endswith('_rgba.png') # the rest of this code assumes that the _rgba image has been passed.
            rgbas = [cv2.cvtColor(cv2.imread(image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for image in self.opt.images]
            rgba_hw = np.stack([cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
            rgb_hw = rgba_hw[..., :3] * rgba_hw[..., 3:] + (1 - rgba_hw[..., 3:])
            self.rgb = torch.from_numpy(rgb_hw).permute(0,3,1,2).contiguous().to(self.device)
            self.mask = torch.from_numpy(rgba_hw[..., 3] > 0.5).to(self.device)
            print(f'[INFO] dataset: load image prompt {self.opt.images} {self.rgb.shape}')

            # load depth
            depth_paths = [image.replace('_rgba.png', '_depth.png') for image in self.opt.images]
            depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]
            depth = np.stack([cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA) for depth in depths])
            self.depth = torch.from_numpy(depth.astype(np.float32) / 255).to(self.device)  # TODO: this should be mapped to FP16
            print(f'[INFO] dataset: load depth prompt {depth_paths} {self.depth.shape}')

            # load normal   # TODO: don't load if normal loss is 0
            normal_paths = [image.replace('_rgba.png', '_normal.png') for image in self.opt.images]
            normals = [cv2.imread(normal_path, cv2.IMREAD_UNCHANGED) for normal_path in normal_paths]
            normal = np.stack([cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA) for normal in normals])
            self.normal = torch.from_numpy(normal.astype(np.float32) / 255).to(self.device)
            print(f'[INFO] dataset: load normal prompt {normal_paths} {self.normal.shape}')

            # encode embeddings for zero123
            if 'zero123' in self.guidance:
                rgba_256 = np.stack([cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
                rgbs_256 = rgba_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
                rgb_256 = torch.from_numpy(rgbs_256).permute(0,3,1,2).contiguous().to(self.device)
                guidance_embeds = self.guidance['zero123'].get_img_embeds(rgb_256)
                self.embeddings['zero123']['default'] = {
                    'zero123_ws' : self.opt.zero123_ws,
                    'c_crossattn' : guidance_embeds[0],
                    'c_concat' : guidance_embeds[1],
                    'ref_polars' : self.opt.ref_polars,
                    'ref_azimuths' : self.opt.ref_azimuths,
                    'ref_radii' : self.opt.ref_radii,
                }

            if 'clip' in self.guidance:
                self.embeddings['clip']['image'] = self.guidance['clip'].get_img_embeds(self.rgb)


    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------

    def train_step(self, data, save_guidance_path:Path=None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """

        # perform RGBD loss instead of SDS if is image-conditioned
        do_rgbd_loss = self.opt.images is not None and \
            (self.global_step % self.opt.known_view_interval == 0)
        stats = {}
        # override random camera with fixed known camera
        if do_rgbd_loss: #F
            data = self.default_view_data

        # experiment iterations ratio
        # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / (self.opt.exp_end_iter - self.opt.exp_start_iter)

        # progressively relaxing view range
        if self.opt.progressive_view: #F
            r = min(1.0, self.opt.progressive_view_init_ratio + 2.0*exp_iter_ratio)
            self.opt.phi_range = [self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                    self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                    self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.opt.progressive_level: #F
            self.model.max_level = min(1.0, 0.25 + 2.0*exp_iter_ratio)

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # When ref_data has B images > opt.batch_size
        if B > self.opt.batch_size: #F
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[:self.opt.batch_size]
            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]

        if do_rgbd_loss: #F
            ambient_ratio = 1.0
            shading = 'lambertian' # use lambertian instead of albedo to get normal
            as_latent = False
            binarize = False
            bg_color = torch.rand((B * N, 3), device=rays_o.device)

            # add camera noise to avoid grid-like artifact
            if self.opt.known_view_noise_scale > 0:
                noise_scale = self.opt.known_view_noise_scale #* (1 - self.global_step / self.opt.iters)
                rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
                rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        elif exp_iter_ratio <= self.opt.latent_iter_ratio: # be latent in first 0.2 step
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None

        else:
            if exp_iter_ratio <= self.opt.albedo_iter_ratio: # F
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = self.opt.min_ambient_ratio + (1.0-self.opt.min_ambient_ratio) * random.random()
                rand = random.random()
                if rand >= (1.0 - self.opt.textureless_ratio):
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            if self.opt.bg_radius > 0 and rand > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg


        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]

        # known view loss
        if do_rgbd_loss: #F
            gt_mask = self.mask # [B, H, W]
            gt_rgb = self.rgb   # [B, 3, H, W]
            gt_normal = self.normal # [B, H, W, 3]
            gt_depth = self.depth   # [B, H, W]

            if len(gt_rgb) > self.opt.batch_size:
                gt_mask = gt_mask[choice]
                gt_rgb = gt_rgb[choice]
                gt_normal = gt_normal[choice]
                gt_depth = gt_depth[choice]

            # color loss
            gt_rgb = gt_rgb * gt_mask[:, None].float() + bg_color.reshape(B, H, W, 3).permute(0,3,1,2).contiguous() * (1 - gt_mask[:, None].float())
            loss = self.opt.lambda_rgb * F.mse_loss(pred_rgb, gt_rgb)

            # mask loss
            loss = loss + self.opt.lambda_mask * F.mse_loss(pred_mask[:, 0], gt_mask.float())

            # normal loss
            if self.opt.lambda_normal > 0 and 'normal_image' in outputs:
                valid_gt_normal = 1 - 2 * gt_normal[gt_mask] # [B, 3]
                valid_pred_normal = 2 * pred_normal[gt_mask] - 1 # [B, 3]

                lambda_normal = self.opt.lambda_normal * min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_normal * (1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean())

            # relative depth loss
            if self.opt.lambda_depth > 0:
                raise # need change
                valid_gt_depth = gt_depth[gt_mask] # [B,]
                valid_pred_depth = pred_depth[:, 0][gt_mask] # [B,]
                lambda_depth = self.opt.lambda_depth * min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_depth * (1 - self.pearson(valid_pred_depth, valid_gt_depth))

                # # scale-invariant
                # with torch.no_grad():
                #     A = torch.cat([valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1) # [B, 2]
                #     X = torch.linalg.lstsq(A, valid_pred_depth).solution # [2, 1]
                #     valid_gt_depth = A @ X # [B, 1]
                # lambda_depth = self.opt.lambda_depth #* min(1, self.global_step / self.opt.iters)
                # loss = loss + lambda_depth * F.mse_loss(valid_pred_depth, valid_gt_depth)

        # novel view loss
        else:

            loss = 0
            guidance_loss = None
            zero123_loss =None
            if 'SD' in self.guidance:
                # interpolate text_z
                azimuth = data['azimuth'] # [-180, 180]

                # ENHANCE: remove loop to handle batch size > 1
                text_z = [self.embeddings['SD']['uncond']] * azimuth.shape[0]
                if self.opt.perpneg: #F
                    text_z_comp, weights = adjust_text_embeddings(self.embeddings['SD'], azimuth, self.opt)
                    text_z.append(text_z_comp)
                else:                
                    for b in range(azimuth.shape[0]):
                        if azimuth[b] >= -90 and azimuth[b] < 90:
                            if azimuth[b] >= 0:
                                r = 1 - azimuth[b] / 90
                            else:
                                r = 1 + azimuth[b] / 90
                            start_z = self.embeddings['SD']['front']
                            end_z = self.embeddings['SD']['side']
                        else:
                            if azimuth[b] >= 0:
                                r = 1 - (azimuth[b] - 90) / 90
                            else:
                                r = 1 + (azimuth[b] + 90) / 90
                            start_z = self.embeddings['SD']['side']
                            end_z = self.embeddings['SD']['back']
                        text_z.append(r * start_z + (1 - r) * end_z)

                text_z = torch.cat(text_z, dim=0)
                if self.opt.perpneg: #F
                    loss_to_add, guidance_loss = self.guidance['SD'].train_step_perpneg(text_z, weights, pred_rgb, as_latent=as_latent, guidance_scale=self.opt.guidance_scale, grad_scale=self.opt.lambda_guidance,
                                                    save_guidance_path=save_guidance_path)
                    loss +=loss_to_add * self.opt.sds_weight
                else:
                    loss_to_add, guidance_loss = self.guidance['SD'].train_step(text_z, pred_rgb, as_latent=as_latent, guidance_scale=self.opt.guidance_scale, grad_scale=self.opt.lambda_guidance,
                                                                save_guidance_path=save_guidance_path)
                    loss += loss_to_add * self.opt.sds_weight # useful checked

            if 'IF' in self.guidance:
                # interpolate text_z
                azimuth = data['azimuth'] # [-180, 180]

                # ENHANCE: remove loop to handle batch size > 1
                text_z = [self.embeddings['IF']['uncond']] * azimuth.shape[0]
                if self.opt.perpneg:
                    text_z_comp, weights = adjust_text_embeddings(self.embeddings['IF'], azimuth, self.opt)
                    text_z.append(text_z_comp)
                else:
                    for b in range(azimuth.shape[0]):
                        if azimuth[b] >= -90 and azimuth[b] < 90:
                            if azimuth[b] >= 0:
                                r = 1 - azimuth[b] / 90
                            else:
                                r = 1 + azimuth[b] / 90
                            start_z = self.embeddings['IF']['front']
                            end_z = self.embeddings['IF']['side']
                        else:
                            if azimuth[b] >= 0:
                                r = 1 - (azimuth[b] - 90) / 90
                            else:
                                r = 1 + (azimuth[b] + 90) / 90
                            start_z = self.embeddings['IF']['side']
                            end_z = self.embeddings['IF']['back']
                        text_z.append(r * start_z + (1 - r) * end_z)

                text_z = torch.cat(text_z, dim=0)

                if self.opt.perpneg:
                    loss = loss + self.guidance['IF'].train_step_perpneg(text_z, weights, pred_rgb, guidance_scale=self.opt.guidance_scale, grad_scale=self.opt.lambda_guidance)
                else:
                    loss = loss + self.guidance['IF'].train_step(text_z, pred_rgb, guidance_scale=self.opt.guidance_scale, grad_scale=self.opt.lambda_guidance)
                    
            if 'zero123' in self.guidance:

                polar = data['polar']
                azimuth = data['azimuth']
                radius = data['radius']

                loss_to_add, zero123_loss = self.guidance['zero123'].train_step(self.embeddings['zero123']['default'], pred_rgb, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale,
                                                                  as_latent=as_latent, grad_scale=self.opt.lambda_guidance, save_guidance_path=save_guidance_path)
                loss = loss + loss_to_add * self.opt.sds_weight
            if 'clip' in self.guidance:

                # empirical, far view should apply smaller CLIP loss
                lambda_guidance = 10 * (1 - abs(azimuth) / 180) * self.opt.lambda_guidance

                loss = loss + self.guidance['clip'].train_step(self.embeddings['clip'], pred_rgb, grad_scale=lambda_guidance)

            if guidance_loss is not None:
                stats['sds'] = guidance_loss
            if zero123_loss is not None:
                stats['zero123'] = zero123_loss

        # regularizations
        if not self.opt.dmtet:
# Loss
            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity
                stats['opacity'] = loss_opacity.item()

            if self.opt.lambda_entropy > 0: # T
                # skewed entropy, favors 0 over 1
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)

                loss = loss + lambda_entropy * loss_entropy
                stats['alpha_entropy'] = loss_entropy.item()

            if self.opt.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
                # pred_vals = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # smoothed_vals = TF.gaussian_blur(pred_vals.detach(), kernel_size=9)
                # loss_smooth = F.mse_loss(pred_vals, smoothed_vals)
                # total-variation
                loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                              (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                loss = loss + self.opt.lambda_2d_normal_smooth * loss_smooth
                stats['2d_normal_smooth'] = loss_smooth.item()

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs: # T
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient
                stats['orient'] = loss_orient.item()

            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
                loss_normal_perturb = outputs['loss_normal_perturb']
                loss = loss + self.opt.lambda_3d_normal_smooth * loss_normal_perturb
                stats['3d_normal_smooth'] = loss_normal_perturb.item()

        else:

            if self.opt.lambda_mesh_normal > 0:
                loss = loss + self.opt.lambda_mesh_normal * outputs['normal_loss']
                stats['mesh_normal_smooth'] = outputs['normal_loss'].item()

                # eval to previous normal loss
            if self.opt.lambda_mesh_laplacian > 0:
                loss = loss + self.opt.lambda_mesh_laplacian * outputs['lap_loss']
                stats['mesh_laplacian'] = outputs['lap_loss'].item()

        return pred_rgb, pred_depth, loss, stats

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        bg_color = torch.tensor([0.999999] * 3).to(self.device)  # white bg

        shading = data['shading'] if 'shading' in data else 'albedo' #albedo when eval equal to lamertain with ratio 1.0
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None #None when eval

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=bg_color, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        pred_depth = self.filter_depth(pred_depth).reshape(B, H, W)
        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def filter_depth(self,pred_depth):
        pred_depth = pred_depth.reshape(-1)

        # filter first time
        if pred_depth.max()>2:
            pred_depth[pred_depth < 2] = 0

            # second filter
            foreground_index = pred_depth.nonzero()
            depth = pred_depth[foreground_index]
            thresh = torch.quantile(depth, 0.1) - 0.20  #???
            pred_depth[pred_depth < thresh] = 0

            # scale
            foreground_index = pred_depth.nonzero()
            depth = pred_depth[foreground_index]
            vmin = torch.quantile(depth, 0.00)
            vmax = torch.quantile(depth, 1.00)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            pred_depth[foreground_index] = depth

        return pred_depth

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        bg_color = torch.tensor([0.999999] * 3).to(self.device)  # white bg
        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None #None
        buffer = self.model.opt.max_steps
        self.model.opt.max_steps = 1024
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color)
        self.model.opt.max_steps = buffer

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        pred_depth = self.filter_depth(pred_depth).reshape(B, H, W)

        return pred_rgb, pred_depth, None

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            # if self.workspace is not None and self.local_rank == 0:
            #     self.save_checkpoint(full=True, best=False)

            if self.epoch == 1 or self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
            if self.epoch % self.opt.save_interval == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader)
            self.train_one_epoch(train_loader, max_epochs)


        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if True:
            if write_video:
                all_preds = []

            with torch.no_grad():

                for i, data in enumerate(loader):

                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth, _ = self.test_step(data) # out B H W C

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).clip(0, 255).astype(np.uint8)
                    pred_depth = np.stack([pred_depth]*3,axis=2)
                    wandb_save_image = np.concatenate([pred, pred_depth],axis=1)

                    if write_video:
                        all_preds.append(wandb_save_image)

                    pbar.update(loader.batch_size)

            if write_video:
                all_preds = np.stack(all_preds, axis=0)
                all_preds = np.transpose(all_preds,(0, 3, 1, 2))
                # video need: T C H W
                # imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
                # imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
                wandb.log({"video": wandb.Video(all_preds, fps=30,caption=f'video_{self.global_step:04d}.mp4')},step=self.global_step)
        if self.ema is not None:
            self.ema.restore()

        self.log(f"==> Finished Test.")


    # [GUI] train text step.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs


    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, light_d=None, ambient_ratio=1.0, shading='albedo'):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        mvp = torch.from_numpy(mvp).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        # from degree theta/phi to 3D normalized vec
        light_d = np.deg2rad(light_d)
        light_d = np.array([
            np.sin(light_d[0]) * np.sin(light_d[1]),
            np.cos(light_d[0]),
            np.sin(light_d[0]) * np.cos(light_d[1]),
        ], dtype=np.float32)
        light_d = torch.from_numpy(light_d).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'mvp': mvp,
            'H': rH,
            'W': rW,
            'light_d': light_d,
            'ambient_ratio': ambient_ratio,
            'shading': shading,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth, _ = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        outputs = {
            'image': preds[0].detach().cpu().numpy(),
            'depth': preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def train_one_epoch(self, loader, max_epochs):
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / 'guidance'
            save_guidance_folder.mkdir(parents=True, exist_ok=True)

        for data in loader:
            if self.opt.anneal_step!=-1 and self.global_step > self.opt.anneal_step:
                self.guidance['SD'].max_step=self.opt.anneal_max_step
            #self.model.cuda_ray T
            #self.model.taichi_ray F
            # update grid every 16 steps
            if (self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0: #16
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.opt.save_guidance and (self.global_step % self.opt.save_guidance_interval == 0):
                    save_guidance_path = save_guidance_folder / f'step_{self.global_step:07d}.png'
                else:
                    save_guidance_path = None
                pred_rgbs, pred_depths, loss, stats = self.train_step(data, save_guidance_path=save_guidance_path)

            # hooked grad clipping for RGB space
            if self.opt.grad_clip_rgb >= 0:
                def _hook(grad):
                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        return grad.clamp(grad_scale * -self.opt.grad_clip_rgb, grad_scale * self.opt.grad_clip_rgb)
                    else:
                        return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)
                pred_rgbs.register_hook(_hook)
                # pred_rgbs.retain_grad()

            self.scaler.scale(loss).backward()
            self.post_train_step()
            # self.model.sdf.grad = torch.nan_to_num(self.model.sdf.grad)
            # self.model.deform.grad = torch.nan_to_num(self.model.deform.grad)
            # if torch.sum(self.model.sdf.grad)==0 or torch.sum(self.model.deform.grad)==0:
            #     print("wrong grad!")
            #     self.model.sdf.grad[:]=1e-6
            #     self.model.deform.grad[:,:]=1e-6
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # if torch.sum(self.model.fix_sdf - self.model.sdf)!=0: #no nan other wise no grad
                # print("sdf change")
            # if torch.sum(self.model.deform) != 0:
                # print("deform change")

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            if self.local_rank == 0:
                for key, value in stats.items():
                    wandb.log({"{}".format(key): value}, step=self.global_step)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()



        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).clip(0, 255).astype(np.uint8)
                    pred_depth =  np.stack([pred_depth]*3,axis=2)
                    wandb_save_image = np.concatenate([pred,pred_depth],axis=1)

                    # cv2.imwrite(save_path, cv2.cvtColor(wandb_save_image, cv2.COLOR_RGB2BGR))
                    wandb_save_image = PIL.Image.fromarray(wandb_save_image, 'RGB')

                    wandb.log({'rgb_{}'.format(self.local_step): [wandb.Image(wandb_save_image, caption=f'rgb_{self.global_step:04d}.png')]}, step=self.global_step)
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale.cpu().numpy()

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return


        if self.opt.dmtet:
            del_list = []
            for key,value in checkpoint_dict['model'].items():
                if self.opt.init_color:
                    if 'encoder' in key or 'sigma_net' in key or 'bg_net' in key:
                        del_list.append(key)
                else:
                    if 'sigma_net' in key or 'bg_net' in key:
                        del_list.append(key)
            for del_name in del_list:
                checkpoint_dict['model'].pop(del_name)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict and not self.opt.dmtet:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")
                raise
        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                new_scale = torch.from_numpy(checkpoint_dict['tet_scale']).to(self.device)
                self.model.verts *= new_scale / self.model.tet_scale
                self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free)/1024**3)*1000)/1000)
        mem += mems[-1]
    return mem, mems

#GAN Trainer
class GanTrainer(Trainer):
    def __init__(self,
                 argv,  # command line args
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 guidance,  # guidance network
                 d_loader,
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=False,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        super().__init__(
            argv,  # command line args
            name,  # name of this experiment
            opt,  # extra conf
            model,  # network
            guidance,  # guidance network
            criterion=criterion,  # loss function, if None, assume inline implementation in train_step
            optimizer=optimizer,  # optimizer
            ema_decay=ema_decay,  # if use EMA, set the decay
            lr_scheduler=lr_scheduler,  # scheduler
            metrics=metrics,
            # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
            local_rank=local_rank,  # which GPU am I
            world_size=world_size,  # total num of GPUs
            device=device,  # device to use, usually setting to None is OK. (auto choose device)
            mute=mute,  # whether to mute all print
            fp16=fp16,  # amp optimize level
            workspace=workspace,  # workspace to save logs & ckpts
            best_mode=best_mode,  # the smaller/larger result, the better
            use_loss_as_metric=use_loss_as_metric,  # use loss as the first metric
            report_metric_at_train=report_metric_at_train,  # also report metrics at training
            use_checkpoint=use_checkpoint,  # which ckpt to use at init time
            use_tensorboardX=use_tensorboardX,  # whether to use tensorboard for logging
            scheduler_update_every_step=scheduler_update_every_step,  # whether to call scheduler.step() after every train step
        )
        self.per_view_gt = opt.per_view_gt
        if not hasattr(self, 'd_optimizer'):
            self.d_optimizer = torch.optim.Adam(self.model.D.parameters(), lr = opt.d_lr, betas=[0,0.99], eps=1e-8)
        if not hasattr(self, 'd_step'):
            self.d_step = 0
        self.d_loader = d_loader
        self.d_iter = iter(self.d_loader)
        self.mse = torch.nn.MSELoss()

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
            'd_step' : self.d_step,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale.cpu().numpy()

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            state['d_optimizer'] = self.d_optimizer.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:
            # T enter here
            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            raise
            # self.model.load_state_dict(checkpoint_dict)
            # self.log("[INFO] loaded model.")
            # return
        if self.opt.dmtet:
            del_list = []
            for key, value in checkpoint_dict['model'].items():
                if self.opt.init_color:
                    if 'encoder' in key or 'sigma_net' in key or 'bg_net' in key:
                        del_list.append(key)
                else:
                    if 'sigma_net' in key or 'bg_net' in key:
                        del_list.append(key)
            for del_name in del_list:
                checkpoint_dict['model'].pop(del_name)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict and not self.opt.dmtet:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")
                raise

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                new_scale = torch.from_numpy(checkpoint_dict['tet_scale']).to(self.device)
                self.model.verts *= new_scale / self.model.tet_scale
                self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        if 'd_step' in checkpoint_dict:
            self.d_step = checkpoint_dict['d_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if 'd_optimizer' in checkpoint_dict:
            try:
                self.d_optimizer = torch.optim.Adam(self.model.D.parameters(), lr=self.opt.d_lr, betas=[0, 0.99], eps=1e-8)
                self.d_optimizer.load_state_dict(checkpoint_dict['d_optimizer'])
                self.log("[INFO] loaded DDDDD optimizer.")
            except:
                self.log("[WARN] Failed to load DDDD optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    def train(self, train_loader, valid_loader, test_loader, img2img_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        self.begin_epoch = self.epoch

        self.update_sd_dataset(img2img_loader)
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            # if self.workspace is not None and self.local_rank == 0:
            #     self.save_checkpoint(full=True, best=False)
            epoch_delta = self.epoch -self.begin_epoch

            if epoch_delta % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
            if epoch_delta != 0 and epoch_delta % self.opt.save_interval == 0:
                self.save_checkpoint(full=True, best=False)
            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader)


            self.train_one_epoch(train_loader, max_epochs)

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def update_sd_dataset(self, loader):
        self.log("!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log("Begin Generating Stable Diffusion Image to Image Dataset!!")
        self.log("!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.real_dataloader = self.gen_real(loader)
        self.real_iter = iter(self.real_dataloader)
        print("Generation Over")
        # generate fake rendering

    def gen_renderings(self, loader):

        self.log(f"Gen Renderings")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        all_preds = []
        depth_list = []
        embeding_list = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, depths, _ = self.test_step(data)

                pred = preds[0].detach().clone() # H, W, 3
                depths = depths[0].detach().clone()
                depths = torch.stack([depths]*3,dim=2)

                pred = pred.permute(2, 0, 1) # 3, H, W
                depths = depths.permute(2, 0, 1) # 3, H, W

                azimuth = data['azimuth']
                for b in range(azimuth.shape[0]):
                    if azimuth[b] >= -90 and azimuth[b] < 90:
                        if azimuth[b] >= 0:
                            r = 1 - azimuth[b] / 90
                        else:
                            r = 1 + azimuth[b] / 90
                        start_z = self.embeddings['SD']['front']
                        end_z = self.embeddings['SD']['side']
                    else:
                        if azimuth[b] >= 0:
                            r = 1 - (azimuth[b] - 90) / 90
                        else:
                            r = 1 + (azimuth[b] + 90) / 90
                        start_z = self.embeddings['SD']['side']
                        end_z = self.embeddings['SD']['back']
                    embeding_list.append(r * start_z + (1 - r) * end_z)
                # pred C, H, W
                pred = F.interpolate(pred[None], (512, 512), mode='bilinear', align_corners=False)[0] # 0, 1

                all_preds.append(pred)
                pbar.update(loader.batch_size)
                depth_list.append(depths)
        self.log(f"==> Finished Gen Rendering {len(loader)}.")
        return all_preds, embeding_list, depth_list

    def gen_real(self, loader):
        view_num = self.opt.img2img_view_num
        group = self.opt.img2img_view_num // self.opt.dataset_size_valid

        save_path = os.path.join(self.opt.real_save_path)

        if not os.path.exists(save_path) or len(os.listdir(save_path))==0 or self.opt.real_overwrite:
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)
            rendering_list, embeding_list, depth_list = self.gen_renderings(loader)
            if self.opt.ctn:
                # rendering_list : list of C, H, W
                if self.opt.normal:
                    from ctn.normal_gen_dir import normal_gen
                    normal_gen(rendering_list=rendering_list, embeding_list=embeding_list, opt=self.opt)

                elif self.opt.depth:
                    from ctn.depth_gen_dir import depth_gen
                    depth_gen(rendering_list=depth_list, embeding_list=embeding_list, opt=self.opt)

                elif self.opt.p2p_text is not None:
                    from ctn.ip2p_gen_dir import p2p_gen
                    p2p_gen(rendering_list=rendering_list, embeding_list=embeding_list, opt=self.opt)
                else:
                    from ctn.softedge_gen_dir import softedge_gen
                    softedge_gen(rendering_list=rendering_list,embeding_list=embeding_list, opt=self.opt)
            else:
                pbar = tqdm.tqdm(total=view_num * self.per_view_gt, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
                with torch.no_grad():
                    for view_index in range(view_num):
                        ref_image = rendering_list[view_index]

                        images = self.guidance['SD'].pipe_img2img(prompt_embeds=embeding_list[view_index] ,output_type='np', image=ref_image, strength=self.opt.real_strength, guidance_scale=self.opt.real_cfg, num_images_per_prompt=self.per_view_gt).images
                        pil_images = numpy_to_pil(images)
                        for image_index in range(len(images)):
                            save_name = '{:0>3d}_{:0>2d}.png'.format(view_index, image_index)
                            save_name = os.path.join(save_path, save_name)
                            pil_images[image_index].save(save_name)
                            pbar.update(1)
                            # if image_index<5 and view_index % group == 0:
                            #     wandb.log({'real_{}'.format(view_index//group): [wandb.Image(images[image_index], caption=f'view_{view_index:03d}_{image_index:03d}')]}, step=image_index)
                        cat_images = numpy_to_pil(np.concatenate(images, axis = 1))[0]
                        npy_save_name = '{:0>3d}.pt'.format(view_index)
                        npy_save_name = os.path.join(save_path, npy_save_name)
                        torch.save(embeding_list[view_index].detach().cpu(), npy_save_name)

                        # if view_index % group == 0:
                        #     wandb.log({'real_{}'.format(view_index//group+1): [wandb.Image(cat_images, caption=f'view_{view_index:03d}')]})

        file_name = os.listdir(save_path)
        images_path = [os.path.join(save_path, name) for name in file_name if ('jpg' in name or 'png' in name)]
        images_path.sort()
        assert len(images_path) == view_num * self.per_view_gt, 'Generated Dataset Error: set --real_overwrite to overwrite files in `real_save_path` or set a new `real_save_path` '
        dataset = ImageDataset(images_path)
        assert dataset.view_num==self.opt.img2img_view_num

        for view_index in range(view_num):
            if view_index % group == 0:
                image_path_list = images_path[view_index * self.per_view_gt:(view_index+1) * self.per_view_gt]
                wandb_img_list = []
                for cur_img_path in image_path_list:
                    cur_img = np.asarray(PIL.Image.open(cur_img_path))
                    wandb_img_list.append(cur_img)
                    if len(wandb_img_list)>=8:
                        break
                wandb_img_to_save=PIL.Image.fromarray(np.uint8(np.concatenate(wandb_img_list,axis=1)))

                wandb.log({'real_{}'.format(view_index // group + 1): [
                    wandb.Image(wandb_img_to_save, caption=f'view_{view_index:03d}')]}, step=self.global_step)

        self.real_dataset = dataset
        dataloader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True)

        return dataloader

    def train_step(self, data, save_guidance_path: Path = None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """

        stats = {}
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / (
                    self.opt.exp_end_iter - self.opt.exp_start_iter)

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']  # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if B > self.opt.batch_size:  # F
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[:self.opt.batch_size]
            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]
        elif exp_iter_ratio <= self.opt.latent_iter_ratio:  # be latent in first 0.2 step
            # set to zero when activate dmtet
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None
        else:
            if exp_iter_ratio <= self.opt.albedo_iter_ratio:
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = self.opt.min_ambient_ratio + (1.0 - self.opt.min_ambient_ratio) * random.random()
                rand = random.random()
                if rand >= (1.0 - self.opt.textureless_ratio):
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            as_latent = False

            binarize = False

            rand = random.random()

            if (self.opt.bg_radius > 0 and rand > 0.5) or self.opt.no_random_bkg:
                bg_color = None  # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device)  # single color random bg

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color,
                                    ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W,
                                                                                                           4).permute(
                0, 3, 1, 2).contiguous()  # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        loss = 0
        # known view loss

        if self.cur_sds_weight > 0:
            guidance_loss = None
            zero123_loss = None
            if 'SD' in self.guidance:
                azimuth = data['azimuth']  # [-180, 180]

                text_z = [self.embeddings['SD']['uncond']] * azimuth.shape[0]
                if self.opt.perpneg:

                    text_z_comp, weights = adjust_text_embeddings(self.embeddings['SD'], azimuth, self.opt)
                    text_z.append(text_z_comp)
                else:
                    for b in range(azimuth.shape[0]):
                        if azimuth[b] >= -90 and azimuth[b] < 90:
                            if azimuth[b] >= 0:
                                r = 1 - azimuth[b] / 90
                            else:
                                r = 1 + azimuth[b] / 90
                            start_z = self.embeddings['SD']['front']
                            end_z = self.embeddings['SD']['side']
                        else:
                            if azimuth[b] >= 0:
                                r = 1 - (azimuth[b] - 90) / 90
                            else:
                                r = 1 + (azimuth[b] + 90) / 90
                            start_z = self.embeddings['SD']['side']
                            end_z = self.embeddings['SD']['back']
                        text_z.append(r * start_z + (1 - r) * end_z)

                text_z = torch.cat(text_z, dim=0)
                if self.opt.perpneg:  # F
                    loss_to_add, guidance_loss = self.guidance['SD'].train_step_perpneg(text_z, weights, pred_rgb,
                                                                                        as_latent=as_latent,
                                                                                        guidance_scale=self.opt.guidance_scale,
                                                                                        grad_scale=self.opt.lambda_guidance,
                                                                                        save_guidance_path=save_guidance_path)
                    loss += loss_to_add * self.cur_sds_weight
                else:
                    loss_to_add, guidance_loss = self.guidance['SD'].train_step(text_z, pred_rgb,
                                                                                as_latent=as_latent,
                                                                                guidance_scale=self.opt.guidance_scale,
                                                                                grad_scale=self.opt.lambda_guidance,
                                                                                save_guidance_path=save_guidance_path)
                    loss += loss_to_add * self.cur_sds_weight

            if guidance_loss is not None:
                stats['sds'] = guidance_loss

        if not self.opt.dmtet:
            # Loss
            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity
                stats['opacity'] = loss_opacity.item()

            if self.opt.lambda_entropy > 0:  # T
                # skewed entropy, favors 0 over 1
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)

                loss = loss + lambda_entropy * loss_entropy
                stats['alpha_entropy'] = loss_entropy.item()

            if self.opt.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:

                loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                              (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                loss = loss + self.opt.lambda_2d_normal_smooth * loss_smooth
                stats['2d_normal_smooth'] = loss_smooth.item()

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:  # T
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient
                stats['orient'] = loss_orient.item()

            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
                loss_normal_perturb = outputs['loss_normal_perturb']
                loss = loss + self.opt.lambda_3d_normal_smooth * loss_normal_perturb
                stats['3d_normal_smooth'] = loss_normal_perturb.item()

        else:
            if self.opt.lambda_mesh_normal > 0:
                loss = loss + self.opt.lambda_mesh_normal * outputs['normal_loss']
                stats['mesh_normal_smooth'] = outputs['normal_loss'].item()

                # eval to previous normal loss
            if self.opt.lambda_mesh_laplacian > 0:
                loss = loss + self.opt.lambda_mesh_laplacian * outputs['lap_loss']
                stats['mesh_laplacian'] = outputs['lap_loss'].item()

        return pred_rgb, pred_depth, loss, stats

    def train_one_epoch(self, loader, max_epochs):
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
        self.train_loader_len = len(loader)
        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / 'guidance'
            save_guidance_folder.mkdir(parents=True, exist_ok=True)
        for data in loader:
            self.update_status()

            stats = {}
            # train G
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss, reg_stats = self.train_step(data, save_guidance_path=None)
                stats.update(reg_stats)
            # hooked grad clipping for RGB space
            if self.opt.grad_clip_rgb >= 0: # F
                def _hook(grad):
                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        return grad.clamp(grad_scale * -self.opt.grad_clip_rgb, grad_scale * self.opt.grad_clip_rgb)
                    else:
                        return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)
                pred_rgbs.register_hook(_hook)

            self.scaler.scale(loss).backward()
            del loss
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # discrimination loss
            if self.cur_g_loss_weight > 0:
                self.model.D.requires_grad_(False)
                self.optimizer.zero_grad()
                g_loss, g_loss_dict = self.g_loss()
                stats.update(g_loss_dict)
                g_loss.backward()
                del g_loss
                self.optimizer.step()

                self.model.D.requires_grad_(True)

            # l2 loss
            if self.cur_l2_weight > 0:
                self.optimizer.zero_grad()

                l2_loss, l2_loss_dict = self.l2_loss()
                stats.update(l2_loss_dict)

                l2_loss.backward()
                del l2_loss
                self.optimizer.step()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            # train D
            if self.cur_g_loss_weight > 0:
                try:
                    real_data, view_index = next(self.real_iter)
                except Exception as e:
                    self.real_iter = iter(self.real_dataloader)
                    real_data, view_index = next(self.real_iter)
                if self.global_step % self.opt.d_reg_interval == 0:
                    d_loss_dict = self.d_update(real_data, view_index, reg=True)
                else:
                    d_loss_dict = self.d_update(real_data, view_index, reg=False)
                stats.update(d_loss_dict)

            if self.local_rank == 0:
                for key, value in stats.items():
                    wandb.log({"{}".format(key): value}, step=self.global_step)
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.")

    def update_status(self):
        if self.global_step > self.opt.g_loss_decay_begin_step:
            decay_step = self.global_step - self.opt.g_loss_decay_begin_step
            self.cur_g_loss_weight = self.opt.g_loss_weight + min(decay_step / self.opt.g_loss_decay_step, 1) * (
                        self.opt.g_loss_weight_end - self.opt.g_loss_weight)
        else:
            self.cur_g_loss_weight = self.opt.g_loss_weight

        if self.global_step > self.opt.sds_decay_begin_step:
            decay_step = self.global_step - self.opt.sds_decay_begin_step
            self.cur_sds_weight = self.opt.sds_weight + min(decay_step / self.opt.sds_decay_step, 1) * (
                        self.opt.sds_weight_end - self.opt.sds_weight)
        else:
            self.cur_sds_weight = self.opt.sds_weight

        if self.global_step > self.opt.l2_decay_begin_step:
            decay_step = self.global_step - self.opt.l2_decay_begin_step
            self.cur_l2_weight = self.opt.l2_weight + min(decay_step / self.opt.l2_decay_step, 1) * (
                        self.opt.l2_weight_end - self.opt.l2_weight)
        else:
            self.cur_l2_weight = self.opt.l2_weight

        if self.opt.anneal_step!=-1 and self.global_step > self.opt.anneal_step:
            self.guidance['SD'].max_step=self.opt.anneal_max_step
        if (self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0: #16
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()
        self.local_step += 1
        self.global_step += 1

    def l2_loss(self):

        try:
            fake_data = next(self.d_iter)
        except Exception as e:
            self.d_iter = iter(self.d_loader)
            fake_data = next(self.d_iter)
        rays_o = fake_data['rays_o']  # [B, N, 3]
        rays_d = fake_data['rays_d']  # [B, N, 3]
        mvp = fake_data['mvp']  # [B, 4, 4]
        view_index = fake_data['view_index']

        B, N = rays_o.shape[:2]
        H, W = fake_data['H'], fake_data['W']
        ambient_ratio = 1.0
        shading = 'lambertian'

        rand = random.random()
        if (self.opt.bg_radius > 0 and rand > 0.5) or self.opt.no_random_bkg or self.opt.g_bkg_network:
            bg_color = None  # use bg_net
        else:
            bg_color = torch.rand(3).to(self.device)  # single color random bg

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged = False, perturb = True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading)
        loss = 0
        loss_dict = {}

        fake_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        fake_rgb = F.interpolate(fake_rgb, (512, 512), mode='bilinear', align_corners=False)

        real_data = self.real_dataset.get_view(view_index)
        real_data = real_data.to(self.device)

        l2_loss = self.mse(fake_rgb, real_data).mean()

        loss_dict['l2_loss'] = l2_loss.item()
        loss += l2_loss * self.cur_l2_weight
        return loss, loss_dict

    def g_loss(self):
        try:
            fake_data = next(self.d_iter)
        except Exception as e:
            self.d_iter = iter(self.d_loader)
            fake_data = next(self.d_iter)
        rays_o = fake_data['rays_o'] # [B, N, 3]
        rays_d = fake_data['rays_d'] # [B, N, 3]
        mvp = fake_data['mvp'] # [B, 4, 4]
        view_index = fake_data['view_index']
        view_input = torch.tensor([view_index,0.0],device=self.device)

        B, N = rays_o.shape[:2]
        H, W = fake_data['H'], fake_data['W']
        ambient_ratio = 1.0
        # render a
        shading = 'lambertian'

        #binarize = False  only cuda_ray, default false, forget it
        rand = random.random()
        if (self.opt.bg_radius > 0 and rand > 0.5) or self.opt.no_random_bkg or self.opt.g_bkg_network:
            bg_color = None  # use bg_net
        else:
            # bg_color = torch.tensor([0.999999]*3).to(self.device)  # white bg
            bg_color = torch.rand(3).to(self.device)  # single color random bg

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged = False, perturb = True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading)
            # perturb false related to Nerf, should be false when train D, True when train G
            # staged not related
        # show_image(outputs['image'].reshape(B, H, W, 3))
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
        loss = 0
        loss_dict = {}

        fake_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        fake_rgb = F.interpolate(fake_rgb, (512, 512), mode='bilinear', align_corners=False)

        fake_logits = self.query_D(fake_rgb, view_input)

        loss_fake = (torch.nn.functional.softplus(-fake_logits)).mean()
        loss_dict['loss_fake'] = loss_fake.item()
        loss_dict['loss_fake_weighted'] = loss_fake.item() * self.cur_g_loss_weight

        loss += loss_fake * self.cur_g_loss_weight
        return loss, loss_dict

    def d_update(self, real_data, real_view_index, reg):
        self.d_optimizer.zero_grad()
        loss_dict={}
        if reg:
            loss_reg = self.d_reg_loss(real_data, real_view_index)
            loss_dict['d_reg_loss'] = loss_reg.item()
            loss = loss_reg
        else:
            try:
                fake_data = next(self.d_iter)
            except Exception as e:
                self.d_iter = iter(self.d_loader)
                fake_data = next(self.d_iter)
            loss_fake, loss_real = self.d_main_loss(fake_data, real_data, real_view_index)
            loss_dict['d_fake_loss'] = loss_fake.item()
            loss_dict['d_real_loss'] = loss_real.item()
            loss = loss_fake + loss_real
        loss.backward()
        self.d_optimizer.step()
        return loss_dict

    def d_main_loss(self, fake_data, real_image, real_view_index):
        rays_o = fake_data['rays_o'] # [B, N, 3]
        rays_d = fake_data['rays_d'] # [B, N, 3]
        mvp = fake_data['mvp'] # [B, 4, 4]
        view_index = fake_data['view_index']
        view_input = torch.tensor([view_index,0.0],device=self.device)
        B, N = rays_o.shape[:2]
        H, W = fake_data['H'], fake_data['W']
        ambient_ratio = 1.0
        # render a
        shading = 'lambertian'

        binarize = False
        bg_color = None
        with torch.no_grad():
            outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged = True, perturb = False, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
            # perturb false related to Nerf, should be false when train D, True when train G
        fake_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        fake_rgb = F.interpolate(fake_rgb, (512, 512), mode='bilinear', align_corners=False)
        fake_logits = self.query_D(fake_rgb, view_input)
        loss_fake = (torch.nn.functional.softplus(fake_logits)).mean()

        # show_image(outputs['image'].reshape(B, H, W, 3))
        real_image = real_image.to(self.device)
        real_view_input = torch.tensor([real_view_index,0.0],device=self.device)
        real_logits = self.query_D(real_image, real_view_input)
        loss_real = (torch.nn.functional.softplus(-real_logits)).mean()

        return loss_fake, loss_real

    def query_D(self, img, view_input):
        if self.opt.no_cam_D:
            fake_logits = self.model.D(img, None) # fake rgb 0 to 1
        else:
            if random.random()<self.opt.cam_drop_prob:
                view_input[0] = 0.0
                view_input[1] = 1.0
            fake_logits = self.model.D(img, view_input[None]) # fake rgb 0 to 1
        return fake_logits

    def d_reg_loss(self, real_image, real_view_index):

        real_image = real_image.to(self.device)
        real_image = real_image.detach().requires_grad_(True)
        real_view_input = torch.tensor([real_view_index,0.0], device=self.device)
        real_logits = self.query_D(real_image, real_view_input)

        with conv2d_gradfix.no_weight_gradients():
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_image],
                                           create_graph=True, only_inputs=True)
            r1_grads_image = r1_grads[0]
        r1_penalty = r1_grads_image.square().sum([1, 2, 3]) # loss lower so the
        loss_reg = r1_penalty * (self.opt.gamma / 2)
        loss_reg = loss_reg.mean() * self.opt.d_reg_interval
        return loss_reg


def show_image(preds):
    # outputs['image'].reshape(B, H, W, 3)
    import matplotlib.pyplot as plt
    pred = preds[0].detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    wandb_save_image = PIL.Image.fromarray(pred, 'RGB')
    plt.imshow(wandb_save_image)
    plt.show()