# Only label data (Baseline)
CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py --save_path "results/baseline"

# Ada-Net
CUDA_VISIBLE_DEVICES=0 python3 train_adanet.py --save_path "results/Ada-Net"

# Meta-learning (Ours)
CUDA_VISIBLE_DEVICES=0 python3 train_meta_learning.py --save_path "results/Meta-learning"
