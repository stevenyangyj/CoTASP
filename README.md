# CoTASP
Code for "[**Co**ntinual **T**ask **A**llocation in Meta-Policy Network via **S**parse **P**rompting](https://arxiv.org/abs/2305.18444)", presented in ICML 2023.

## Key Dependencies
```console
python==3.7.13
- jax==0.3.17
- jaxlib==0.3.15+cuda11.cudnn82
- flax==0.6.4
- optax==0.1.4
- scikit-learn==1.0.2
- tensorflow-probability==0.18.0
- sentence-transformers==2.2.2
```
Refer to [this repo](https://github.com/awarelab/continual_world) for the installation of Continual World.

## Quick Start
```python
python train_cotasp.py
```

## Reproducibility
Tracked experiments on CW20 via [Weights & Biases](https://api.wandb.ai/links/yang-yj/5kbiuz7h).

## Citing CoTASP
If you use the code in CoTASP, please kindly cite our paper using following BibTeX entry.
```
@inproceedings{
yang2023cotasp,
title={Continual Task Allocation in Meta-Policy Network via Sparse Prompting},
author={Yijun Yang and Tianyi Zhou and Jing Jiang and Guodong Long and Yuhui Shi},
booktitle={International Conference on Machine Learning},
year={2023}
}
```

## Acknowledgement
We appreciate the open source of the following projects:

[Continual World](https://github.com/awarelab/continual_world), [Meta World](https://github.com/Farama-Foundation/Metaworld), and [JaxRL](https://github.com/ikostrikov/jaxrl)