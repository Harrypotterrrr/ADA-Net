### monster
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-only-alpha1-kl" \
        --additional "None" \
        --mix-up \
        --alpha "1." \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "kl" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-reweight-mixup-only-alpha1-kl" \
        --additional "None" \
        --mix-up \
        --mix-up-reweight \
        --alpha "1." \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "kl" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='2' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-only-alpha1-mse" \
        --additional "None" \
        --mix-up \
        --alpha "1." \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='3' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-only-alpha1-kl-fix-inner" \
        --additional "None" \
        --mix-up \
        --alpha "1." \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --fix-inner \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "kl" \
        --print-freq "100";

### 4x2080ti
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-unlabel-auto-weight1-alpha1-kl" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "1." \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "kl" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-unlabel-auto-weight10-alpha1-kl" \
        --additional "unlabel" \
        --weight "10." \
        --auto-weight \
        --mix-up \
        --alpha "1." \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "kl" \
        --print-freq "100";

### desktop
CUDA_VISIBLE_DEVICES='0' python3 train_meta_simple.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-only-alpha1-simple-multiplier1" \
        --additional "None" \
        --mix-up \
        --alpha "1." \
        --multiplier "1." \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='1' python3 train_meta_simple.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-only-alpha1-simple-multiplier0.1" \
        --additional "None" \
        --mix-up \
        --alpha "1." \
        --multiplier ".1" \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --print-freq "100";
