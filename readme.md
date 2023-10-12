# Official repo for IT3D

**IT3D: Improved Text-to-3D Generation with Explicit View Synthesis**.

[Yiwen Chen](https://buaacyw.github.io/), [Chi Zhang](https://icoz69.github.io/), Xiaofeng Yang, [Zhongang Cai](https://caizhongang.github.io/), [Gang Yu](https://www.skicyyu.org/), [Lei Yang](https://scholar.google.com.hk/citations?user=jZH2IPYAAAAJ&hl=en), [Guosheng Lin](https://guosheng.github.io/)

[Arxiv](https://arxiv.org/abs/2308.11473)

## Abstract

Recent strides in Text-to-3D techniques have been propelled by distilling knowledge from powerful large text-to-image diffusion models (LDMs). Nonetheless, existing Text-to-3D approaches often grapple with challenges such as over-saturation, inadequate detailing, and unrealistic outputs. This study presents a novel strategy that leverages explicitly synthesized multi-view images to address these issues. Our approach involves the utilization of image-to-image pipelines, empowered by LDMs, to generate posed high-quality images based on the renderings of coarse 3D models. Although the generated images mostly alleviate the aforementioned issues, challenges such as view inconsistency and significant content variance persist due to the inherent generative nature of large diffusion models, posing extensive difficulties in leveraging these images effectively. To overcome this hurdle, we advocate integrating a discriminator alongside a novel Diffusion-GAN dual training strategy to guide the training of 3D models. For the incorporated discriminator, the synthesized multi-view images are considered real data, while the renderings of the optimized 3D models function as fake data. We conduct a comprehensive set of experiments that demonstrate the effectiveness of our method over baseline approaches.






https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/5256c1f8-bcc5-4141-b071-dee1f8b7d3d3






# Demos
<details close>
  <summary>More Videos</summary>

  ### Left: Coarse Model (Baseline). Right: Refined Model (Ours). File Name: Prompt

  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/5dc67170-c515-4287-b38a-62d15e11c73e

  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/c55486cf-cb97-4542-9f80-f93632ae6387


  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/ca1f1af2-d9e8-4967-8c58-d7990451df03

  
  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/ed27802f-f2fe-4b10-af33-61e90f74fac8

  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/83a89e36-5cc3-4fd8-82d9-da48a6e41e38
  
  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/3d30ea64-1211-4b79-94c9-475089dc811c

  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/e1d05536-67bd-4ef7-8440-8ed39e5bd8f8
  
  
  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/4294ae3d-9248-4fa8-88c0-8a119ea8932d
  
  
  
  
  
  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/90e7dcd1-7c87-43a1-8367-38ca55998231
  
  
  
  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/fcc9a4a5-3a40-4468-94fe-5da4f6c67f9c
  
  
  
  
  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/c053a4e5-763a-4b94-a104-0961d40cfdb2
  
  
  https://github.com/buaacyw/IT3D-text-to-3D/assets/52091468/5dd56968-da1b-42a4-8684-303b6abfd8ea
  



</details>




# Install

```bash
git clone https://github.com/buaacyw/IT3D-text-to-3D.git
cd IT3D-text-to-3D
conda create -n it3d python==3.8
conda activate it3d
pip install -r requirements.txt
pip install ./raymarching
pip install ./shencoder
pip install ./freqencoder
pip install ./gridencoder
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
In 512 resolution, it takes around 30G for training a coarse model (vanilla Stable Dreamfusion)  and 35G for refining it with IT3D.
You may lower the memory consumption by:
* Lower training resolution by setting `--h` and `--w`. While this will significantly reduce memory usage, it will also lead to a substantial decrease in performance. It takes about 10G for IT3D in 64 resolution.
* Use lightweight NeRF by setting `--nerf l1`. Our default setting is `--nerf l2`.
* Lower sampling steps per ray by setting `--max_steps`. Our default setting is `--max_steps 384`
* If you OOM during Controlnet data generation, lower `--ctn_sample_batch_size`.

### Tips for Performance
* Change prompt and seed by setting `--text` and `--seed`. Sadly, training a coarse model free from the Janus problem often requires multiple attempts.
* Rendering NeRF as latent feature at the early stage of coarse model training by setting `--latent_iter_ratio 0.1`.
* Change the discrimination loss `--g_loss_weight`. You need to lower `--g_loss_weight` when the generated dataset is too various. You may enlarge `--g_loss_weight` for high quality dataset.
* Tune the GAN longer will increase quality. Change `--g_loss_decay_begin_step` and `--g_loss_decay_step`. In our default setting, we tune the GAN for 7500 steps and then discard it.

### Download coarse model checkpoints
We release our [coarse model checkpoints](https://drive.google.com/file/d/1juXz2qVLipriaEoUxZjQwOGahXm6IyZd/view). 
Unzip into folder `ckpts`. All these checkpoints are trained in our default coarse model setting.

# Usage
On our A6000, it takes 6 minutes to generate a dataset of 640 images using SD-I2I, and 25 minutes using Controlnet, respectively.

```bash

## Refine a coarse NeRF
# --no_cam_D: camera free discriminator, camera pose won't be input to discriminator
# --g_loss_decay_begin_step: when to decay the weight of discrimination loss
# --real_save_path: path to generated dataset

# Jasmine
python main.py -O --text "a bunch of white jasmine" --workspace jas_ctn --ckpt ckpts/jas_df_ep0200.pth --no_cam_D --gan --ctn --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/jas_ctn

# Use stable diffusion img2img pipeline instead of Controlnet
python main.py -O --text "a bunch of white jasmine" --workspace jas_sd --ckpt ckpts/jas_df_ep0200.pth --no_cam_D --gan  --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/jas_sd

# Iron Man
python main.py -O --text "a 3D model of an iron man, highly detailed, full body" --workspace iron_ctn --ckpt ckpts/iron_man_df_ep0400.pth --no_cam_D --gan --ctn --g_loss_decay_begin_step 45000 --real_save_path generated_dataset/iron_ctn

# Darth Vader
python main.py -O --text "Full-body 3D model of Darth Vader, highly detailed" --workspace darth_ctn --ckpt ckpts/darth_df_ep0200.pth --no_cam_D --gan --ctn --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/darth_ctn

# Hulk
python main.py -O --text "3D model of hulk, highly detailed" --workspace hulk_ctn --ckpt ckpts/hulk_df_ep0200.pth --no_cam_D --gan --ctn  --g_loss_decay_begin_step 25000 --real_save_path generated_dataset/hulk_ctn

# Ablation Experiment in Paper
# Note: our default setting is sds loss + decayed gan loss. gan loss weight will be decayed to zero after 7500 steps (depending on g_loss_decay_begin_step)
# only l2 loss
python main.py -O --text "3D model of hulk, highly detailed" --workspace hulk_ctn_l2 --ckpt ckpts/hulk_df_ep0200.pth --no_cam_D --gan --ctn --l2_weight 100.0 --l2_decay_begin_step 25000 --l2_decay_step 2500 --l2_weight_end 0.0 --sds_weight_end 0.0 --g_loss_decay_begin_step 0 --real_save_path generated_dataset/hulk_ctn

# l2 loss + sds loss
python main.py -O --text "3D model of hulk, highly detailed" --workspace hulk_ctn_l2_sds --ckpt ckpts/hulk_df_ep0200.pth --no_cam_D --gan --ctn --l2_weight 100.0 --l2_decay_begin_step 25000 --l2_decay_step 2500 --l2_weight_end 0.0  --g_loss_decay_begin_step 0 --real_save_path generated_dataset/hulk_ctn

# only GAN
python main.py -O --text "3D model of hulk, highly detailed" --workspace hulk_ctn_only_gan --ckpt ckpts/hulk_df_ep0200.pth --no_cam_D --gan --ctn --sds_weight_end 0.0 --real_save_path generated_dataset/hulk_ctn

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
* --gan: Incorporating discriminator (IT3D)
* --ctn: Using ControlNet condition on softedge. If false, StableDiffusion Image-to-Image Pipeline will be used. SD I2I is much faster but with lower quality.
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
  @misc{chen2023it3d,
        title={IT3D: Improved Text-to-3D Generation with Explicit View Synthesis}, 
        author={Yiwen Chen and Chi Zhang and Xiaofeng Yang and Zhongang Cai and Gang Yu and Lei Yang and Guosheng Lin},
        year={2023},
        eprint={2308.11473},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```
