#! /in/bash

SCENE=lego
CONFIG_NAME=lego_dynamic
BATCH_SIZE=1
# sentitive to gpu memory, lower it if gpu mem is small
PATCH_SIZE=52
# related to receptive field
PATCH_STRIDE=4
RGB_W=0
STYLE_W=1e8
CONTENT_W=1
DENSITY_W=1e6
EXPNAME=${CONFIG_NAME}_dynamic_P${PATCH_SIZE}_PS${PATCH_STRIDE}_RGB${RGB_W}_S${STYLE_W}_C${CONTENT_W}_D${DENSITY_W}_updateAllwithteacherbatch
mkdir -p logs/$CONFIG_NAME
mkdir -p logs/$CONFIG_NAME/checkpoints

# Number of GPUs available in the machine
CUDA_VISIBLE_DEVICES=1,2 python run_nerf.py \
 --num_devices 2 \
 --batch_size ${BATCH_SIZE} \
 --config configs/${CONFIG_NAME}.txt  \
 --data_path datasets/dnerf_synthetic/${SCENE}   \
 --expname $EXPNAME    \
 --with_teach  \
 --d_weight $DENSITY_W  \
 --content_weight $CONTENT_W  \
 --rgb_weight ${RGB_W}   \
 --style_weight  ${STYLE_W}  \
 --patch_size ${PATCH_SIZE}  \
 --N_iters 200000  \
 --render_video \
 --loss_terms coarse fine style_v_all density   \
 --fix_param False False \
 --i_testset 10000  \
 --i_weights 10000 \
 --i_video 10000  \
 --patch_stride ${PATCH_STRIDE} \
 --stl_idx 0 \
 --is_dynamic \
 --with_mask \
 --style_path datasets/single_styles/the_scream.jpg \
 --ckpt_path ckpts/${CONFIG_NAME}_P40_00150000.ckpt 2>&1 | tee -a logs/${EXPNAME}/${EXPNAME}.txt \
#  --teach_ckpt_path ckpts/${CONFIG_NAME}_00075000.ckpt \
#  --eval
