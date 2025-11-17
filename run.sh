#!/bin/bash
DATA_DIR="/data/xiongwenxuan/Datasets/tandt_db/tandt/truck"
OUTPUT_DIR="${DATA_DIR}/result/3DHGS_250514_claude4_30000"

CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    -s "$DATA_DIR" \
    -m "$OUTPUT_DIR" \
    --iterations 30000 \
    --save_iterations 7000 15000 22000 30000 \
    --test_iterations 7000 15000 22000 30000 \
    \
    --lambda_adaptive_opacity 0.0008 \
    --adaptive_opacity_start_iter 6000 \
    --smooth_region_quantile 0.68 \
    \
    --lambda_selective_normal 0.0006 \
    --selective_normal_start_iter 10000 \
    --interior_region_quantile 0.72 \
    --knn_k_normal 5 \
    --knn_recompute_interval 600 \
    --position_lr_max_steps 30000 \
    --lambda_trusted_normal 0.001 \
    --trusted_normal_from_iter 18000 \
    --ema_alpha_normal_anchor 0.99 > "${OUTPUT_DIR}/250514output_claude4_30000.log" 2>&1 &

echo "30K Training finished - Results in: $OUTPUT_DIR"
