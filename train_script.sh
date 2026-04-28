python main_train.py \
--image_dir /data/serverC/Projects/R2Gen-Mamba/datasets/UAPD/JPEGImages \
--ann_path /data/serverC/Projects/R2Gen-Mamba/datasets/UAPD/annotations-split.json \
--dataset_name UAPD \
--max_seq_length 100 \
--threshold 3 \
--batch_size 32 \
--epochs 100 \
--save_dir results/UAPD \
--step_size 10 \
--gamma 0.8 \
--d_vf 512 \
--visual_mode vmamba_swin \
--vis_patch_size 4 \
--vis_embed_dim 128 \
--cross_scan_fuse concat \
--seed 9223 \

