I only modified the VOC dataset script, and the ADE script was modified by referring to VOC. The sequential, overlap, and disjoint modes are specified through the "setting" parameter. If the GPU memory is sufficient, you can increase the batch size. trainacil first predicts and then upsamples, while trainacil2 upsamples first and then predicts (approximately increasing the computation and memory usage by 200 times).

I'm currently running trainacil, and in the alignment step, I lose around 2 mIoU. If you have any questions, feel free to ask me. Also, the code is poorly written, and you can refactor it based on the original BARM repository or other sources. Please don't criticize too harshly!
---Jiaxu Li


<!-- # BARM - Official Pytorch Implementation (ECCV 2024) -->
<div align="center">
<h1>Official Pytorch Implementation of BARM: 

Background Adaptation with Residual Modeling for Exemplar-Free
Class-Incremental Semantic Segmentation </h1>
Anqi Zhang, Guangyu Gao <br />
School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China</sub><br />

Accepted in ECCV 2024 <br />

[![Paper](https://img.shields.io/badge/arxiv-2407.09838-aqua
)](https://arxiv.org/abs/2407.09838)

<img src = "figures/framework.png" width="100%" height="100%">
</div>

## Preparation

### Requirements

- CUDA>=11.8
- torch>=2.0.0
- torchvision>=0.15.0
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib


### Datasets

We use the Pascal VOC 2012 and ADE20K datasets for evaluation following the previous methods. You can download the datasets from the following links:

Download Pascal VOC 2012 dataset:
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
Download Additional Segmentation Class Annotations:
```bash
wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
```

Download ADE20K dataset:
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```


```
data_root/
   ├── VOC2012/
   │   ├── Annotations/
   │   ├── ImageSet/
   │   ├── JPEGImages/
   │   ├── SegmentationClassAug/
   │   └── saliency_map/
   └── ADEChallengeData2016
       ├── annotations
       │   ├── training
       │   └── validation
       └── images
           ├── training
           └── validation
```
MODEL=deeplabv3_resnet101
DATA_ROOT=/data/yt/BARM2/data_root/cityscapes
DATASET=cityscapes_domain
TASK=11-5
EPOCH=20
BATCH=2
LOSS=bce_loss
LR=0.001
THRESH=0.7
SUBPATH=BARM
CURR=1
METHOD=acil
SETTING=overlap

CUDA_VISIBLE_DEVICES=0 \
python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp\
    --curr_step ${CURR} --subpath ${SUBPATH} --initial --method ${METHOD} --setting ${SETTING}
## Getting Started

### Class-Incremental Segmentation Segmentation on VOC 2012

Run our scripts `run_init.sh` and `run.sh` for class-incremental segmentation on VOC 2012 dataset, or follow the instructions below.



```bash
MODEL=deeplabv3_resnet101
DATA_ROOT=/data/yt/BARM/data_root/VOC2012
DATASET=voc
TASK=10-1
EPOCH=50
BATCH=16
LOSS=bce_loss
LR=0.001
THRESH=0.7
SUBPATH=BARM
CURR=0
METHOD=acil
SETTING=overlap
CUDA_VISIBLE_DEVICES=3 \
python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp \
    --curr_step ${CURR} --subpath ${SUBPATH} --initial  --method ${METHOD} --setting ${SETTING}
```

Incremental steps:


```bash
MODEL=deeplabv3_resnet101
DATA_ROOT=/data/yt/BARM/data_root/VOC2012
DATASET=voc
TASK=10-1
EPOCH=20
BATCH=2
LOSS=bce_loss
LR=0.001
THRESH=0.7
SUBPATH=BARM
CURR=1
METHOD=acil
SETTING=sequential

CUDA_VISIBLE_DEVICES=1 \
python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp\
    --curr_step ${CURR} --subpath ${SUBPATH}  --method ${METHOD} --setting ${SETTING}
```


### Class-Incremental Segmentation Segmentation on ADE20K

Run our scripts `run_init.sh` and `run.sh` for class-incremental segmentation on ADE20K dataset, or follow the instructions below.

Initial step: 
```bash
MODEL=deeplabv3_resnet101
DATA_ROOT=/data/yt/BARM/data_root/ADEChallengeData2016
DATASET=ade
TASK=100-5
EPOCH=60
BATCH=8
LOSS=bce_loss
LR=0.01
THRESH=0.7
SUBPATH=BARM
CURR=0
METHOD=acil
SETTING=overlap

CUDA_VISIBLE_DEVICES=0 \
python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp \
    --curr_step ${CURR} --subpath ${SUBPATH} --initial --method ${METHOD} --setting ${SETTING}
```

Incremental steps:
```bash
MODEL=deeplabv3_resnet101
DATA_ROOT= /data/yt/BARM/data_root/ADEChallengeData2016
DATASET=ade
TASK=100-5
EPOCH=100
BATCH=4
LOSS=bce_loss
LR=0.001
THRESH=0.7
SUBPATH=BARM
CURR=1
METHOD=acil
SETTING=overlap
CUDA_VISIBLE_DEVICES=3 \

python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp \
    --curr_step ${CURR} --subpath ${SUBPATH} --method ${METHOD} --setting ${SETTING}
```


## Experiment Results

### Quantitative Results

The following table shows the mIoU results of different methods on the Pascal VOC 2012 and ADE20K dataset. 
Our trained weights are available at [ModelScope](https://www.modelscope.cn/models/UltraDoughnut/BARM) and [123Pan](https://www.123pan.com/s/VOFbVv-grcsH.html)(registration required). 

|  Method  | VOC 10-1 (11 tasks) | VOC 15-1 (6 tasks) | VOC 5-3 (6 tasks) | VOC 19-1 (2 tasks) |
|:--------:|:-------------------:|:------------------:|:-----------------:|:------------------:|
|   MiB    |        12.7         |        29.3        |       46.7        |        69.2        |
|   PLOP   |        30.5         |        54.6        |       18.7        |        73.5        |
|   DKD    |        60.4         |        69.7        |       58.1        |        76.0        |
| **BARM** |      **62.1**       |      **70.0**      |     **61.1**      |      **76.4**      |

|   Method   | ADE 100-5 (11 tasks) | ADE 100-10 (6 tasks) | ADE 100-50 (2 tasks) | ADE 50-50 (3 tasks) |
|:----------:|:--------------------:|:--------------------:|:--------------------:|:-------------------:|
|    MiB     |         26.0         |         29.2         |         32.8         |        29.3         |
|    PLOP    |         28.8         |         31.6         |         32.9         |        30.4         |
|    DKD     |          -           |         34.3         |         36.0         |        33.9         |
| **BARM** |       **34.1**       |       **35.2**       |       **35.7**       |      **33.7**       |


### Qualitative Results

<img src = "figures/barm_effect.png" width="100%" height="100%">
<img src = "figures/qual.png" width="100%" height="100%">

## Citation
```
@inproceedings{zhang2024background,
  title={Background Adaptation with Residual Modeling for Exemplar-Free Class-Incremental Semantic Segmentation},
  author={Zhang, Anqi and Gao, Guangyu},
  journal={ECCV},
  year={2024}
}
```

## Acknowledgement

Our implementation is based on these repositories: [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch), [SSUL](https://github.com/clovaai/SSUL).
Thanks for their great work!


