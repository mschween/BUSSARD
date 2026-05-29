# BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection

[![arXiv](https://img.shields.io/badge/arXiv-2603.16645-b31b1b.svg)](https://arxiv.org/abs/2603.16645)

<p align="center">
  <img src="imgs/Model-Pipeline.png" alt="BUSSARD Pipeline" width="80%">
</p>

BUSSARD detects anomalous object relationships in indoor scenes by combining scene graph generation with normalizing flows. It embeds object-relation-object triplets from scene graphs using word embeddings and an autoencoder, then uses a RealNVP flow to score each triplet by its likelihood under a learned normal distribution. Anomalous relationships, like 'plate-on-chair', receive low likelihood and are flagged as outliers.
 

## Abstract

We propose Bijective Universal Scene-Specific Anomalous Relationship Detection (BUSSARD), a normalizing flow-based model for detecting anomalous relations in scene graphs, generated from images. Our work follows a multimodal approach, embedding object and relationship tokens from scene graphs with a language model to leverage semantic knowledge from the real world. A normalizing flow model is used to learn bijective transformations that map object-relation-object triplets from scene graphs to a simple base distribution (typically Gaussian), allowing anomaly detection through likelihood estimation. We evaluate our approach on the SARD dataset containing office and dining room scenes. Our method achieves around 10% better AUROC results compared to the current state-of-the-art model, while simultaneously being five times faster. Through ablation studies, we demonstrate superior robustness and universality, particularly regarding the use of synonyms, with our model maintaining stable performance while the baseline shows 17.5% deviation. This work demonstrates the strong potential of learning-based methods for relationship anomaly detection in scene graphs.

## Setup

### Environment

Create the virtual environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

### Dataset Preparation

**SARD dataset:**  
Copy or link the SARD dataset to `data/SARD_IndoorDataset`. See the [original codebase](https://github.com/marow17623/SARD) for details.

**MIT-67 dataset (optional, for evaluation):**  
Copy or link the MIT-67 dataset to `data/mit-67`. See the [original website](https://web.mit.edu/torralba/www/indoor.html) for details, then run the following command, to create a balanced subset of MIT-67:

```bash
python sample_mit-67.py
```

**GloVe word vectors:**  
Download the latest pretrained GloVe vectors from [their website](https://nlp.stanford.edu/projects/glove/) and extract the contents into `data/`.

### Generate Scene Graphs

Set up the [SARD](https://github.com/marow17623/SARD) repository following its instructions, then run the following for both `dining_room` and `office` (replace xx):

```bash
python sgg_egtr.py --scene_name xx --dataset_dir data/SARD_IndoorDataset

# Optional: for MIT-67 evaluation
python sgg_egtr.py --scene_name xx --dataset_dir data/balanced_mit-67 --output_dir output_egtr/balanced_indoorCVPR_09
```

Copy the generated `output_egtr/` folder into this repository.

### Prepare ConceptNet

Download the needed [ConceptNet](https://conceptnet.io/) data by running the following command:

```bash
python build_conceptnet_db.py
```

## Training

### 1. Pretrain Autoencoder

```bash
python autoencoder.py
```

> If using MIT-67, enable the corresponding section at the end of the file.  
> Pretrained weights will be saved in `output/autoencoder_weights/`.


### 2. Train Normalizing Flow

Run experiments for both scenes with multiple seeds:

```bash
python run_experiments.py
```

Or run a single scene with a single seed:

```bash
python run_flow.py
```

> Check `config` for data and scene graph paths.

## Citation

If you use this work, please cite:

```bibtex
@misc{schween2026bussard,
  title         = {{BUSSARD}: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection},
  author        = {Schween, Melissa and Kruse, Mathis and Rosenhahn, Bodo},
  year          = {2026},
  eprint        = {2603.16645},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2603.16645}
}
```
 
Once published at CVPR, please use:
 
```bibtex
@inproceedings{schween2026bussard,
  title     = {{BUSSARD}: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection},
  author    = {Schween, Melissa and Kruse, Mathis and Rosenhahn, Bodo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```