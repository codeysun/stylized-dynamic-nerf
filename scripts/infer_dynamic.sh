CUDA_VISIBLE_DEVICES=1  python run_nerf.py  \
    --num_devices 1\
    --batch_size 1 \
    --config configs/lego_dynamic.txt  \
    --data_path datasets/dnerf_synthetic/lego   \
    --expname  lego_dynamic_thescream    \
    --ckpt_path ckpts/lego_dynamic_thescream.ckpt    \
    --style_path datasets/single_styles/the_scream.jpg \
    --render_video \
    --eval