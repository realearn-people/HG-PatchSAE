# HG-PatchSAE

Official code for **"Heatmap-Guided Spatial Weighting Improves Class-Discriminative SAE Feature Vector Extraction in Vision Transformers"**
Chavakan Yimmark (JAIST) and Teeradaj Racharak (Tohoku University).

> **Built on PatchSAE.** This repository is a modified version of [PatchSAE](https://github.com/dynamical-inference/patchsae) by Hyesu Lim, Jinho Choi, Jaegul Choo and Steffen Schneider ([*Sparse autoencoders reveal selective remapping of visual concepts during adaptation*](https://openreview.net/forum?id=imT03YXlG2), ICLR 2025). We use their pre-trained sparse autoencoder, their hooked ViT and their Top-K masking protocol. The PatchSAE code is kept in `src/`, `tasks/` and `configs/`, the git history starts from their release, and every change we made to their files is listed in [Changes to PatchSAE](#changes-to-patchsae). If you use this code, please cite PatchSAE as well (see [Citation](#citation)).

PatchSAE gives one sparse feature vector per ViT patch. A single image-level vector is usually obtained by averaging all patches, which lets background patches dilute the class-discriminative features. **HG-PatchSAE** is parameter-free: it uses a post-hoc heatmap (LayerCAM, Grad-SAM, LeGrad, …) as soft pooling weights over the patch-level SAE latents.

```
z_p = ReLU(W_enc x_p + b_enc)         PatchSAE latent of patch p            (Eq. 1)
w_p = heatmap averaged over patch p   14 x 14 grid, L1-normalised           (Eq. 2)
f   = sum_p w_p z_p                   image-level HG-PatchSAE vector        (Eq. 3)
```

Class prototypes average `f` over 130 training images per class (Eq. 4). We evaluate them with **Top-K Feature Faithfulness**, `FF(K) = ON_K − OFF_K`. This metric intervenes on CLIP ViT-B/16 through the SAE and measures whether a prototype's top-K features are sufficient (ON) and necessary (OFF) for zero-shot classification.

## Main results

ImageNet-1k validation, CLIP ViT-B/16, PatchSAE with F = 49,152 features.

| Method | ON₁₀₀ | OFF₁₀₀ | FF₁₀₀ | Jaccard@50 | NCC Top-1 | NCC Top-5 | EPG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform (PatchSAE mean) | 29.67 | 12.37 | +17.31 | 0.0932 | 55.70 | 82.23 | 0.318 |
| Random | 29.71 | 12.35 | +17.36 | 0.0917 | 55.45 | 82.11 | 0.318 |
| Raw Attention | 24.32 | 25.99 | −1.67 | 0.0428 | 19.46 | 36.52 | 0.138 |
| GradCAM | 29.59 | 12.96 | +16.62 | 0.0727 | 46.32 | 72.00 | 0.321 |
| Attention Rollout | 32.80 | 11.15 | +21.65 | 0.0906 | 56.29 | 82.38 | 0.471 |
| LeGrad | 30.58 | **9.49** | +21.09 | **0.1364** | 63.69 | 88.02 | 0.469 |
| Transformer Attribution | 35.33 | 10.07 | +25.27 | 0.1217 | 65.63 | 89.15 | 0.424 |
| Grad-SAM | 37.79 | 10.42 | +27.37 | 0.1228 | 65.62 | 89.55 | 0.504 |
| **LayerCAM** | **38.29** | 9.95 | **+28.34** | 0.1333 | **66.27** | **90.00** | **0.505** |

The unmodified ViT reaches 63.96%. Across the seven heatmap methods and Random, EPG and FF₁₀₀ correlate at Pearson r = 0.949 and Spearman ρ = 0.905.

## Repository layout

```
hgpatchsae/                  HG-PatchSAE (new)
  model.py                     HGPatchSAE: heatmap extractors + heatmap-weighted SAE pooling
experiments/                 paper experiments (new unless noted)
  build_prototypes.py          class prototypes from 130 train images/class      Sec. 4.4
  feature_faithfulness.py      Top-K ON/OFF masking → FF                          Table 1, Fig. 3
                               (adapted from PatchSAE's classification_with_top_k_masking.py)
  jaccard.py                   intra-class Jaccard@50                             Table 1
  ncc.py                       nearest-centroid classification                    Table 1
  epg.py                       Energy-based Pointing Game                         Table 2
  paper_tables.py              prints Table 1 (FF) and Table 2 with the EPG–FF correlation
  plot_topk_sweep.py           Fig. 3
  feature_galleries.py         top-activating images of SAE features (Fig. 1-style)
  ff_significance.py           bootstrap CIs and paired ΔFF vs. uniform (supplementary)
  plot_epg_heatmaps.py         per-method heatmaps with the GT box and EPG (supplementary)
scripts/                     01–04: end-to-end commands for every table and figure
src/, tasks/, configs/       PatchSAE (Lim et al., ICLR 2025), with the changes listed below
```

## Setup

```bash
conda env create -f environment.yml      # Python 3.12, PyTorch 2.2.2, CUDA 12.1
conda activate hgpatchsae
```

**PatchSAE checkpoint and feature statistics.** These come from the [PatchSAE release](https://github.com/dynamical-inference/patchsae). Extract them into the repo root so that `data/sae_weight/base/out.pt` and `out/feature_data/` exist:

```bash
gdown --id 1reuDjXsiMkntf1JJPLC5a3CcWuJ6Ji3Z && unzip data.zip   # data/ (~513 MB)
gdown --id 1NJzF8PriKz_mopBY4l8_44R0FVi2uw2g && unzip out.zip    # out/  (~35 MB)
```

**ImageNet.** Images are streamed from the Hugging Face Hub (`evanarlian/imagenet_1k_resized_256`) on first use. EPG additionally needs the original ILSVRC 2012 validation images and bounding-box XMLs (from the Kaggle *ImageNet Object Localization Challenge*). Put them under `data/imagenet/ILSVRC/{Data,Annotations}/CLS-LOC/val`, or set `IMAGE_DIR` / `BBOX_DIR`.

## Reproducing the paper

Run all commands from the repo root. Results go to `outputs/`.

```bash
bash scripts/01_build_prototypes.sh     # prototypes, 130 train images/class   (Sec. 4.4)
bash scripts/02_feature_faithfulness.sh # ON/OFF masking, Fig. 3, FF columns   (Table 1)
bash scripts/03_jaccard_ncc.sh          # Jaccard@50 and NCC                   (Table 1)
bash scripts/04_epg.sh                  # EPG and EPG–FF correlation           (Table 2)
```

Step 1 is the most expensive: it needs one forward pass and one shared backward pass per image for all methods. You can split it across GPUs by class range, as described in the docstring of `experiments/build_prototypes.py`. All later steps reuse its prototypes. Once the masking and EPG results exist, `python experiments/paper_tables.py` reprints the tables without a GPU.

Prototypes are written to `outputs/imagenet_130/prototypes/<method>/cls_hgpatchsae_<tag>.npy`. Each method directory also contains `cls_patchsae_<tag>.npy`, the uniform-pooling (original PatchSAE) prototype, which is identical in every directory.

### Using HG-PatchSAE on your own images

```python
from hgpatchsae import HGPatchSAE        # run from the repo root

hg = HGPatchSAE()                       # loads CLIP ViT-B/16 + PatchSAE (ImageNet class names)
heatmap = hg.get_heatmap("cat.jpg", "a photo of a tabby cat", method="layercam")   # [1, 224, 224]
feats = hg.run_all_methods("cat.jpg", true_class="tabby cat", methods=["layercam"])
f = feats["layercam"]                   # HG-PatchSAE vector, shape [49152]
uniform = feats["uniform"]              # PatchSAE uniform-average baseline
```

Methods evaluated in the paper: `layercam`, `grad_sam`, `transformer_attribution`, `legrad`, `gradcam`, `attention_rollout`, `raw_attention` (plus `random`). `grad_rollout`, `eigencam` and `reciprocam` are also implemented but were not evaluated. You can add your own method with `hg.register_heatmap_method(name, fn)`, where `fn(image, text_prompt)` returns an `[H, W]` array.

### Implementation details

- SAE latents below τ = 0.2 are zeroed before pooling (`activation_mode="thresholded_real"`).
- PatchSAE's noisy-feature filter is kept: features whose mean activation on ImageNet-Sketch exceeds 0.1 are zeroed.
- Each heatmap is min-max scaled on the 14×14 grid, then bilinearly upsampled to 224×224. For Attention Rollout and Raw Attention, patches below 0.5 × max are zeroed first. GradCAM and LayerCAM use transformer block −4.
- `w_p` is the mean of the *positive* heatmap pixels inside patch p (`HGPatchSAE.heatmap_to_vit_token_indices`). Patches with no positive pixels get weight 0.
- Pooled vectors are L1-normalised. This gives the same direction as normalising `w` in Eq. 2.
- EPG uses 500 random validation images (seed 42) whose bounding boxes cover at most 50% of the image.

## Changes to PatchSAE

Files under `src/`, `tasks/` and `configs/` come from PatchSAE ([commit `6b468b3`](https://github.com/dynamical-inference/patchsae/commit/6b468b3)). Every change is marked with an `HG-PatchSAE:` comment in the code. `git diff 6b468b3 -- src tasks configs` shows them all.

| File | Change | Why |
|---|---|---|
| `src/models/utils.py` | The CLIP processor resizes to 224×224 instead of center-cropping. | Setting used for all experiments. |
| `src/sae_training/sparse_autoencoder.py` | `forward_clamp` applies `clamp_feat_dim` as an element-wise mask instead of indexing. | Indexing dropped features for partial masks. Identical for the all-True mask used in Top-K masking. |
| `src/demo/core.py` | The noisy-feature filter reads the `imagenet-sketch` statistics. | Only those statistics are loaded (see below). |
| `tasks/utils.py` | `--dataset_name imagenet` loads the **validation** split (`imagenet-train` for training). Class names come from the dataset's label names. Caltech-101 is no longer loaded in `load_datasets()`. | All evaluations run on the validation split. |
| `tasks/classification_with_top_k_masking.py` | Moved to `experiments/feature_faithfulness.py`. It takes the prototype path directly and builds masks on the GPU. | Top-K Feature Faithfulness (Sec. 4.5). |

Removed because HG-PatchSAE does not use them: the Gradio demo (`src/demo/app.py`, `src/demo/utils.py`), `demo.ipynb`, `analysis/`, `assets/`, and the SAE training scripts in `scripts/`. SAE training itself (`tasks/train_sae_vit.py`) is unchanged and documented in [tasks/README.md](tasks/README.md). For the demo and the analysis notebooks, see the [PatchSAE repository](https://github.com/dynamical-inference/patchsae).

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{yimmark2026hgpatchsae,
  title  = {Heatmap-Guided Spatial Weighting Improves Class-Discriminative {SAE} Feature Vector Extraction in Vision Transformers},
  author = {Yimmark, Chavakan and Racharak, Teeradaj},
  year   = {2026}
}
```

and PatchSAE, which this code builds on:

```bibtex
@inproceedings{lim2025patchsae,
  title     = {Sparse autoencoders reveal selective remapping of visual concepts during adaptation},
  author    = {Hyesu Lim and Jinho Choi and Jaegul Choo and Steffen Schneider},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=imT03YXlG2}
}
```

## License and acknowledgements

Released under the MIT License, the same license as PatchSAE. [LICENSE](LICENSE) keeps the original PatchSAE copyright notice alongside ours, and [NOTICE](NOTICE) lists the third-party code included in PatchSAE.

We thank the PatchSAE authors for releasing their code and trained SAE. The heatmaps use [LeGrad](https://github.com/WalBouss/LeGrad) and [OpenCLIP](https://github.com/mlfoundations/open_clip). The Transformer Attribution heatmap follows [Chefer et al. (2021)](https://github.com/hila-chefer/Transformer-Explainability). We thank Gregory Schwartzman for insightful discussions. This work was supported by JSPS KAKENHI Grant Number JP26K03014 and used computing resources of the Japan Advanced Institute of Science and Technology.
