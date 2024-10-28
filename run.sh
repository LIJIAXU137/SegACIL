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
SETTING=disjoint

CUDA_VISIBLE_DEVICES=2 \
python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp\
    --curr_step ${CURR} --subpath ${SUBPATH}  --method ${METHOD} --setting ${SETTING}