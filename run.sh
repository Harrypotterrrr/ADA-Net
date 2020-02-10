# Only label data (Baseline)
# CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py --save_path "results/baseline"
#
# Ada-Net
# CUDA_VISIBLE_DEVICES=0 python3 train_adanet.py --save_path "results/ada-Net"
#
# Meta-learning (Ours)
CUDA_VISIBLE_DEVICES=0 python3 train_meta_learning.py --save_path "results/meta-learning" \
                                                      --num_label "4000" \
                                                      --epsilon "1e-2" \
                                                      --multiplier "1." \
                                                      --print_freq "1"
