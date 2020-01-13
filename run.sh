# Only label data (Baseline)
python3 train_baseline.py --parallel --gpus 0 1 --num_workers 8 --save_path "results/baseline"

# Ada-Net
python3 train_adanet.py --parallel --gpus 0 1 --num_workers 8 --save_path "results/Ada-Net"

# Meta-learning (Ours)
python3 train_meta_learning.py --parallel --gpus 0 1 --num_workers 8 --save_path "results/Meta-learning"
