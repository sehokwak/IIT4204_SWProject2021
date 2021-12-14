# NEED TO SET
GPU=0
DATASET_ROOT=dataset
SESSION=10classes
HOME_ROOT="drive/Othercomputers/MacBook/IIT4204_SWProject2021"
SAVE_ROOT="${HOME_ROOT}"/log
NUM_CLASSES=10


# Default setting
DATASET=voc12
IMG_ROOT=${DATASET_ROOT}/image_${NUM_CLASSES}
BACKBONE=resnet50_cls


# 1. train classification network
CUDA_VISIBLE_DEVICES=${GPU} python3 "$HOME_ROOT"/main.py \
    --dataset ${DATASET} \
    --train_list "$HOME_ROOT"/metadata/${DATASET}/train_${NUM_CLASSES}.txt \
    --session ${SESSION} \
    --network network.${BACKBONE} \
    --data_root ${IMG_ROOT} \
    --crop_size 448 \
    --max_iters 10000 \
    --batch_size 8 \
    --save_root ${SAVE_ROOT}


## 2. inference CAM
INFER_DATA=train_${NUM_CLASSES} # train / train_aug
TRAINED_WEIGHT=${SAVE_ROOT}/${SESSION}/${NUM_CLASSES}_final_cls.pth
GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python3 "$HOME_ROOT"/infer.py \
    --dataset ${DATASET} \
    --infer_list "$HOME_ROOT"/metadata/${DATASET}/${INFER_DATA}.txt \
    --img_root ${IMG_ROOT} \
    --network network.${BACKBONE} \
    --weights ${TRAINED_WEIGHT} \
    --thr 0.2 \
    --n_gpus 1 \
    --n_processes_per_gpu 1 \
    --cam_png ${SAVE_ROOT}/${SESSION}/result/cam_png


## 3. evaluate CAM
GT_ROOT=${DATASET_ROOT}/mask_${NUM_CLASSES}/

CUDA_VISIBLE_DEVICES=${GPU} python3 "$HOME_ROOT"/evaluate_png.py \
    --dataset ${DATASET} \
    --datalist "$HOME_ROOT"/metadata/${DATASET}/${INFER_DATA}.txt \
    --gt_dir ${GT_ROOT} \
    --save_path ${SAVE_ROOT}/${SESSION}/result/${INFER_DATA}.txt \
    --pred_dir ${SAVE_ROOT}/${SESSION}/result/cam_png