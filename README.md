# UniTS-Sensory-Time-Series-Classification
Datasets and Pytorch code for [UniTS: Short-Time Fourier Inspired Neural Networks for Sensory Time Series Classification](https://dl.acm.org/doi/10.1145/3485730.3485942), in The 19th ACM Conference on Embedded Networked Sensor Systems (SenSys 2021).

## Prerequisite

> PyTorch >= 1.8.0
>
> numpy
>
> scikit-learn
>
> pytorch-complex

## How to run

1. Run main.py for UniTS and the following baseline models ([ResNet](https://arxiv.org/abs/1611.06455), [MaCNN](https://dl.acm.org/doi/10.1145/3161174), [RFNet-base](https://dl.acm.org/doi/10.1145/3384419.3430735), [THAT](https://ojs.aaai.org/index.php/AAAI/article/view/16103), [LaxCat](https://arxiv.org/abs/2011.11631))
2. Run complex_main.py for model [STFNets](https://arxiv.org/abs/1902.07849).

## Datasets

### Processed:

https://drive.google.com/file/d/1aPb-iy6ic-bcg-azXVQ2_2uegn0oQc_j/view?usp=sharing

### Raw:

Motion:

https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition#:~:text=Data%20Set%20Information%3A-,The%20OPPORTUNITY%20Dataset%20for%20Human%20Activity%20Recognition%20from%20Wearable%2C%20Object,%2C%20feature%20extraction%2C%20etc).

Seizure:

https://physionet.org/content/chbmit/1.0.0/

WiFi:

https://github.com/ermongroup/Wifi_Activity_Recognition

KETI:

https://github.com/Shuheng-Li/Relational-Inference/tree/master/KETI_oneweek

## Notes

Code will use cuda by default. You may tune model hyperparameter for better results.

## Citations

Please cite the following paper if you use this repository in your research work:

```
@inproceedings{10.1145/3485730.3485942,
author = {Li, Shuheng and Chowdhury, Ranak Roy and Shang, Jingbo and Gupta, Rajesh K. and Hong, Dezhi},
title = {UniTS: Short-Time Fourier Inspired Neural Networks for Sensory Time Series Classification},
year = {2021},
isbn = {9781450390972},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3485730.3485942},
doi = {10.1145/3485730.3485942}
}
```

Contact **Shuheng Li** [✉️](mailto:shl060@ucsd.edu) for questions, comments and reporting bugs.

