import sys
sys.path.insert(0,'./ctn')
import os

import PIL
from tqdm import tqdm

import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

model_name = 'control_v11e_sd15_ip2p'

model = create_model(f'/home/yiwen/sd_models/{model_name}.yaml').cpu()

model.load_state_dict(load_state_dict('/home/yiwen/sd_models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'/home/yiwen/sd_models/{model_name}.pth', location='cuda'), strict=False)

model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):

    with torch.no_grad():
        input_image = HWC3(input_image)

        detected_map = input_image.copy()

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

def p2p_gen(rendering_list, embeding_list, opt):
    save_dir = opt.real_save_path
    prompt = opt.p2p_text
    assert prompt is not None
    num_samples = opt.per_view_gt
    seed = opt.ctn_seed
    image_resolution = opt.image_resolution
    strength = opt.strength
    guess_mode = opt.guess_mode
    ddim_steps = opt.ddim_steps
    scale = opt.ctn_cfg
    eta = opt.eta
    a_prompt = opt.a_prompt
    n_prompt = opt.n_prompt

    print('save_dir:', save_dir)
    iterator = tqdm(rendering_list, desc='CTN Normal', total=len(rendering_list))

    max_samples = 8
    part_num = num_samples // max_samples

    part_list=[max_samples] * part_num
    if num_samples % max_samples !=0:
        part_list.append(num_samples%max_samples)

    for view_index, input_image in enumerate(iterator):
        input_image = input_image.permute(1,2,0) # C, H, W -> H, W, C
        input_image = input_image.detach().cpu().numpy()
        input_image = (input_image * 255).astype(np.uint8) # 0, 255
        res=[]
        for cur_samples_num in part_list:
            cur_res = process(input_image, prompt, a_prompt, n_prompt, cur_samples_num, image_resolution,
                          ddim_steps, guess_mode, strength, scale, seed, eta)
            cur_res = cur_res[1:] # remove softedge
            res.extend(cur_res)
        assert len(res)==num_samples
        for res_index, cur_image in enumerate(res):
            cur_image = PIL.Image.fromarray(cur_image, 'RGB')
            cur_image.save(os.path.join(save_dir,"{:0>4d}_{:0>3d}.png".format(view_index, res_index)))
        npy_save_name = '{:0>3d}.pt'.format(view_index)
        npy_save_name = os.path.join(save_dir, npy_save_name)
        torch.save(embeding_list[view_index].detach().cpu(), npy_save_name)

