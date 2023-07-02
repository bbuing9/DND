# DND: What Makes Better Augmentation Strategies? Augment Difficult but Not too Different (ICLR 22)

This repository contains code for the paper
**"What Makes Better Augmentation Strategies? Augment Difficult but Not too Different"** 
by [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), Dongyeop Kang, Sungsoo Ahn, and Jinwoo Shin.

<p align="center" >
    <img src=assets/iclr22_main_figure.jpg width="70%">
</p>

## Dependencies
The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using `Python 3.7`. 

## Scripts
Please check out `run.sh` for the scripts to run the baseline algorithms and ours (DND). One can download the pre-constructed augmentations from the [google drive](https://drive.google.com/file/d/13oaDkg4QDUthceQ3HfEpr-scqYfeplHX/view?usp=sharing).

### Training procedure of DND 

Train a network with baseline augmentation, e.g., Cutoff 
```
python train.py --train_type 0000_aug_cutoff --cutoff 0.3 --dataset review --seed 1234
```
Train a network with the proposed DND 
```
python train.py --train_type 0000_dnd --reweight --policy_update 5 --policy_lr 1e-3 --lambda_sim 0.5 --lambda_recon 0.05 --dataset review --seed 1234
```
