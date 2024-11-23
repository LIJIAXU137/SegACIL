MODEL=deeplabv3_resnet101
DATA_ROOT=/data/yt/BARM2/data_root/cityscapes
DATASET=cityscapes_domain
TASK=1-1
EPOCH=50
BATCH=6
LOSS=bce_loss
LR=0.01
THRESH=0.7
SUBPATH=BARM
CURR=1
METHOD=acil
SETTING=overlap

CUDA_VISIBLE_DEVICES=1 \
python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp\
    --curr_step ${CURR} --subpath ${SUBPATH}  --method ${METHOD} --setting ${SETTING}