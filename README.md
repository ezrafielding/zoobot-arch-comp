# A Comparison of Deep Learning Architectures for Optical Galaxy Morphology Classification
[![GitHub](https://img.shields.io/github/license/ezrafielding/zoobot-arch-comp)](https://github.com/ezrafielding/zoobot-arch-comp/blob/main/LICENSE) [![DOI](https://img.shields.io/badge/DOI-10.1109%2FICECET52533.2021.9698414-blue)](https://doi.org/10.1109/ICECET52533.2021.9698414) [![arXiv](https://img.shields.io/badge/arXiv-2111.04353-b31b1b.svg)](https://arxiv.org/abs/2111.04353)

## Abstract
The classification of galaxy morphology plays a crucial role in understanding galaxy formation and evolution. Traditionally, this process is done manually. The emergence of deep learning techniques has given room for the automation of this process. As such, this paper offers a comparison of deep learning architectures to determine which is best suited for optical galaxy morphology classification. Adapting the model training method proposed by Walmsley et al in 2021, the Zoobot Python library is used to train models to predict Galaxy Zoo DECaLS decision tree responses, made by volunteers, using EfficientNet B0, DenseNet121 and ResNet50 as core model architectures. The predicted results are then used to generate accuracy metrics per decision tree question to determine architecture performance. DenseNet121 was found to produce the best results, in terms of accuracy, with a reasonable training time. In future, further testing with more deep learning architectures could prove beneficial.

## Zoobot Library
The lastest version of the Zoobot library used in this work can be found here: https://github.com/mwalmsley/zoobot

## Citation
E. Fielding, C. N. Nyirenda and M. Vaccari, "A Comparison of Deep Learning Architectures for Optical Galaxy Morphology Classification," 2021 International Conference on Electrical, Computer and Energy Technologies (ICECET), 2021, pp. 1-5, doi: 10.1109/ICECET52533.2021.9698414.
