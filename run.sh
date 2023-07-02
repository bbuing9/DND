DATASET=review
SEED="3456"
GPU=7
EPOCHS=15
RATIO=1.0
BATCH_SIZE=8
BACKBONE=roberta

for seed in $SEED
do
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_base --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_aug_eda --lambda_aug 1.0 --lambda_kl 0.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_aug_backtrans --lambda_aug 1.0 --lambda_kl 0.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_aug_r3f --lambda_aug 0.0 --lambda_kl 1.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_aug_adv0.1 --step_size 0.1 --lambda_aug 1.0 --lambda_kl 0.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_aug_coda0.1 --step_size 0.1 --lambda_aug 1.0 --lambda_kl 0.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_aug_contextual --lambda_aug 1.0 --lambda_kl 0.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_aug_cutoff --cutoff 0.3 --lambda_aug 1.0 --lambda_kl 0.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_mixup --mixup_alpha 1 --lambda_aug 1.0 --lambda_kl 0.0 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
 
  CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0xxx_dnd --reweight --policy_update 5 --policy_lr 1e-3 --lambda_aug 1.0 --lambda_sim 0.5 --lambda_recon 0.05 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
done
