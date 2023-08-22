# IT3D

Official implementation of **IT3D: Improved Text-to-3D Generation with Explicit View Synthesis**.



https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/e1d05536-67bd-4ef7-8440-8ed39e5bd8f8



https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/ed27802f-f2fe-4b10-af33-61e90f74fac8


https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/437d7242-b8c7-4ee7-a92c-1f3627a3d9dc



# Install

```bash
git clone https://github.com/buaacyw/IT3D.git
cd it3d
conda create -n it3d python==3.8
conda activate it3d
pip install -r requirements.txt
bash scripts/install_ext.sh
```
### Wandb Login

You need to register a wandb account if you don't have one.
```bash
wandb login
```
### Download image-to-image models (Optional)
For image-to-image pipeline, We have implemented [Stadiffusion Image2Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) and [ControlNetv1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly). 

In our experiment, Controlnet always provides better results. If you want to use Controlnet as image-to-image pipeline, you need to download models from [here](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) following instructions from [ControlNetv1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly).

For example, if you want to use Controlnet conditioned on softedge, you need to download `control_v11p_sd15_softedge.yaml` and `control_v11p_sd15_softedge.pth` and put them in the folder `ctn_models`.
Additionally, You need to download Stable Diffusion 1.5 model [`v1-5-pruned.ckpt`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and put it in the folder `ctn_models`.

### Tested environments
* Ubuntu 22 with torch 2.0.1 & CUDA 11.7 on a A6000.

### Tips for OOM
All our demos (coarse models and fine models) are trained in 512 resolution.
In 512 resolution, it takes around 30G for training a coarse model (vanilla Stable Dreamfusion)  and 35G for refining it with IT3D
You may lower the memory consumption by:
* Lower training resolution by setting `--h` and `--w`. While this will significantly reduce memory usage, it will also lead to a substantial decrease in performance. Our default setting is `--h 512 --w 512`.
* Use lightweight NeRF by setting `--nerf l1`. Our default setting is `--nerf l2`.
* Lower sampling steps per ray by setting `--max_steps`. Our default setting is `--max_steps 384`
* If you OOM during Controlnet data generation, lower `--ctn_sample_batch_size`.

### Tips for Performance
* Change prompt and seed by setting `--text` and `--seed`. Sadly, training a coarse model free from the Janus problem often requires multiple attempts.
* Rendering NeRF as latent feature at the early stage of coarse model training by setting `--latent_iter_ratio 0.1`.

### Download coarse model checkpoints
We release our [coarse model checkpoints](https://drive.google.com/file/d/1juXz2qVLipriaEoUxZjQwOGahXm6IyZd/view). 
Unzip into folder `ckpts`. All these checkpoints are trained in our default coarse model setting.

# Usage
Check scripts in folder `scripts`.

```bash

## Refine a coarse NeRF
# --no_cam_D: camera free discriminator, camera pose won't be input to discriminator
# --g_loss_decay_begin_step: when to decay the weight of discrimination loss
# --real_save_path: path to generated dataset

# Jasmine
python main.py -O --text "a bunch of white jasmine" --workspace jas_ctn --ckpt ckpts/jas_df_ep0200.pth --no_cam_D --gan --ctn --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/jas_ctn

# Iron Man
python main.py -O --text "a 3D model of an iron man, highly detailed, full body" --workspace iron_ctn --ckpt ckpts/iron_man_df_ep0400.pth --no_cam_D --gan --ctn --g_loss_decay_begin_step 45000 --real_save_path generated_dataset/iron_ctn

# Darth Vader
python main.py -O --text "Full-body 3D model of Darth Vader, highly detailed" --workspace darth_ctn --ckpt ckpts/darth_df_ep0200.pth --no_cam_D --gan --ctn --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/darth_ctn

# Hulk
python main.py -O --text "3D model of hulk, highly detailed" --workspace hulk_ctn --ckpt ckpts/hulk_df_ep0200.pth --no_cam_D --gan --ctn  --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/hulk_ctn

# Edit to red Hulk, change --text
python main.py -O --text "a red hulk, red skin, highly detailed" --workspace hulk_red_ctn --ckpt ckpts/hulk_df_ep0200.pth --no_cam_D --gan --ctn  --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/hulk_ctn

## Generate Dataset and DMTET Mesh
# generate dataset
python main.py -O --text "a bunch of blue rose, highly detailed" --workspace rose_blue_ctn --ckpt ckpts/rose_df_ep0200.pth  --gan --ctn --no_cam_D --iters 0 --real_save_path generated_dataset/rose_blue_ctn 
# DMTET Mesh
python main.py -O --text "a bunch of blue rose, highly detailed" --workspace rose_blue_ctn_dm  --gan --ctn --no_cam_D  --g_loss_decay_begin_step 5000 --g_loss_decay_step 5000  --init_with ckpts/rose_df_ep0200.pth --dmtet --init_color --real_save_path generated_dataset/rose_blue_ctn


## Train your own coarse NeRF
python main.py -O --text "a bunch of white jasmine" --workspace jas
# Refine it
python main.py -O --text "a bunch of white jasmine" --workspace jas_ctn --ckpt jas/checkpoints/df_ep0200.pth --no_cam_D --gan --ctn --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/jas_ctn
```
Possible hyperparameters you need to change:
* --real_overwrite: open it to overwrite the real dataset directory
* --per_view_gt: how many images will be generated for each camera view. Default: 5
* --img2img_view_num: how many camera views for img2img generation. Default: 64.
* --depth: depth-conditioned Controlnet
* --noraml: normal-conditioned Controlnet
* --strength: strength of Controlnet conditioning
* --init_color: whether init the color of DMTET. Sometimes you have to open this option to avoide this [bug](https://github.com/ashawkey/stable-dreamfusion/issues/278).
  
# Acknowledgement

Our code is based on these wonderful repos:

* [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
    ```
    @misc{stable-dreamfusion,
        Author = {Jiaxiang Tang},
        Year = {2022},
        Note = {https://github.com/ashawkey/stable-dreamfusion},
        Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
    }
    ```

* [EG3D](https://github.com/NVlabs/eg3d)
    ```
    @inproceedings{Chan2022,
      author = {Eric R. Chan and Connor Z. Lin and Matthew A. Chan and Koki Nagano and Boxiao Pan and Shalini De Mello and Orazio         Gallo and Leonidas Guibas and Jonathan Tremblay and Sameh Khamis and Tero Karras and Gordon Wetzstein},
      title = {Efficient Geometry-aware {3D} Generative Adversarial Networks},
      booktitle = {CVPR},
      year = {2022}
    }
    ```

* [Controlnet](https://github.com/lllyasviel/ControlNet)
    ```
    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Maneesh Agrawala},
      year={2023},
      eprint={2302.05543},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    ```

# Citation

If you find this work useful, a citation will be appreciated via:
```

```
