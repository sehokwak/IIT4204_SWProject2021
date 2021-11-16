# NEED TO SET
GPU=0
DATASET_ROOT=dataset
SAVE_ROOT=log
SESSION=voc12_cls
MAIN_ROOT=IIT4204_SWProject2021


# Default setting
DATASET=voc12
IMG_ROOT=${DATASET_ROOT}/image_10
BACKBONE=resnet50_cls


# 1. train classification network
CUDA_VISIBLE_DEVICES=${GPU} python3 ${MAIN_ROOT}/main.py \
    --dataset ${DATASET} \
    --train_list metadata/${DATASET}/train_10.txt \
    --session ${SESSION} \
    --network network.${BACKBONE} \
    --data_root ${IMG_ROOT} \
    --crop_size 448 \
    --max_iters 20000 \
    --batch_size 8 \
    --save_root ${SAVE_ROOT}


## 2. inference CAM
INFER_DATA=train_10 # train / train_aug
TRAINED_WEIGHT=${SAVE_ROOT}/${SESSION}/checkpoint_cls.pth
GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python3 ${MAIN_ROOT}/infer.py \
    --dataset ${DATASET} \
    --infer_list metadata/${DATASET}/${INFER_DATA}.txt \
    --img_root ${IMG_ROOT} \
    --network network.${BACKBONE} \
    --weights ${TRAINED_WEIGHT} \
    --thr 0.20 \
    --n_gpus 1 \
    --n_processes_per_gpu 1 \
    --cam_png ${SAVE_ROOT}/${SESSION}/result/cam_png
#
## 3. evaluate CAM
GT_ROOT=${DATASET_ROOT}/mask_10/

CUDA_VISIBLE_DEVICES=${GPU} python3 ${MAIN_ROOT}/evaluate_png.py \
    --dataset ${DATASET} \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --gt_dir ${GT_ROOT} \
    --save_path ${SAVE_ROOT}/${SESSION}/result/${INFER_DATA}.txt \
    --pred_dir ${SAVE_ROOT}/${SESSION}/result/cam_png