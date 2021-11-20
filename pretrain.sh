export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export CUBLAS_WORKSPACE_CONFIG=:16:8

GPUS_PER_NODE=4
BATCH_SIZE_PER_GPU=2

# python -m torch.distributed.launch --nproc_per_node="$GPUS_PER_NODE" --use_env main.py --dataset_config configs/pretrain.json --ema --backbone timm_tf_efficientnet_b5_ns --lr_backbone 5e-5 --batch_size "$BATCH_SIZE_PER_GPU" --output-dir ./results

python -m torch.distributed.launch --nproc_per_node="$GPUS_PER_NODE" --use_env main.py --dataset_config configs/pretrain.json --batch_size "$BATCH_SIZE_PER_GPU" --output-dir ./results --backbone swin_S --backbone_checkpoints pretrained_models/swin_small_patch4_window7_224.pth --lr_backbone 5e-5 --ema