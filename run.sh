# Only label data (Baseline)
# CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py --save_path "results/baseline"
#
# Ada-Net
# CUDA_VISIBLE_DEVICES=0 python3 train_adanet.py --save_path "results/ada-Net"
#
# Meta-learning (Ours)
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile150-ilr0.1-inile150-type0-label3-auto-weight" \
        --use-label \
        --auto-weight \
        --unlabel-weight "3." \
        --total-steps "200000" \
        --num-label "4000" \
        --epsilon "1e-2" \
        --lr "0.1" \
        --milestones "[150000]" \
        --inner-lr "0.1" \
        --inner-milestones "[150000]" \
        --type "0" \
        --aug "zca" \
        --print-freq "20";
