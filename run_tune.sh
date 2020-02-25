######## 02/21 ########
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
# 89.92

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
# 89.01

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
# 90.26

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
# 88.29

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
# 91.57

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse" \
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
        --consistency "mse" \
        --print-freq "100";
# 92.64

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
# 89.89

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
# 88.95

######## 02/22 ########
### monster
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-label-auto-weight1-alpha1-kl" \
        --additional "label" \
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
# 90.35

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-label-auto-weight1-alpha1-mse" \
        --additional "label" \
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
        --consistency "mse" \
        --print-freq "100";
# 90.66

CUDA_VISIBLE_DEVICES='2' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-both-auto-weight1-alpha1-kl" \
        --additional "both" \
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
# 91.28

CUDA_VISIBLE_DEVICES='3' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-both-auto-weight1-alpha1-mse" \
        --additional "both" \
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
        --consistency "mse" \
        --print-freq "100";
# 91.25

### desktop
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse-zca" \
        --additional "unlabel" \
        --zca \
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
        --consistency "mse" \
        --print-freq "100";
# killed

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-unlabel-auto-weight1-alpha0.1-mse" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "0.1" \
        --num-label "4000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";
# 90.45

### 4x2080ti
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse-ent0.1" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --ent-min \
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
# 92.22

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "svhn" \
        --save-path "results/svhn-labels1000-mile30-35-mixup-unlabel-auto-weight1-alpha0.1-mse" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "0.1" \
        --num-label "1000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";
# 95.71

######## 02/23 ########
### monster
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar10" \
        -a "shakeshake" \
        --save-path "results/cifar10-labels4000-shakeshake-mile30-35-mixup-unlabel-auto-weight1-alpha1-kl" \
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
        -a "shakeshake" \
        --save-path "results/cifar10-labels4000-shakeshake-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse" \
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
        --consistency "mse" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='2' python3 train_meta.py \
        --dataset "cifar100" \
        -a "convlarge" \
        --save-path "results/cifar100-labels10000-mile30-35-mixup-unlabel-auto-weight1-alpha1-kl" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "1." \
        --num-label "10000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "kl" \
        --print-freq "100";
# 67.09

CUDA_VISIBLE_DEVICES='3' python3 train_meta.py \
        --dataset "cifar100" \
        -a "convlarge" \
        --save-path "results/cifar100-labels10000-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "1." \
        --num-label "10000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";
# 69.53

### desktop
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar100" \
        -a "shakeshake" \
        --save-path "results/cifar100-labels10000-shakeshake-mile30-35-mixup-unlabel-auto-weight1-alpha1-kl" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "1." \
        --num-label "10000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "kl" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "cifar100" \
        -a "shakeshake" \
        --save-path "results/cifar100-labels10000-shakeshake-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "1." \
        --num-label "10000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";

## 4x2080ti
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "svhn" \
        --save-path "results/svhn-labels1000-mile30-35-mixup-unlabel-auto-weight3-alpha1-mse" \
        --additional "unlabel" \
        --weight "3." \
        --auto-weight \
        --mix-up \
        --alpha "1" \
        --num-label "1000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100"

CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "svhn" \
        --save-path "results/svhn-labels1000-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "1" \
        --num-label "1000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100"

######## 02/24 ########
### monster
CUDA_VISIBLE_DEVICES='2' python3 train_meta.py \
        --dataset "svhn" \
        -a "convlarge" \
        --save-path "results/svhn-labels1000-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse-again" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "0.1" \
        --num-label "1000" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";

CUDA_VISIBLE_DEVICES='3' python3 train_meta.py \
        --dataset "cifar10" \
        -a "convlarge" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup-unlabel-auto-weight1-alpha1-mse-again" \
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
        --consistency "mse" \
        --seed "94578" \
        --print-freq "100";

######## 02/25 ########
### 4x2080ti
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "svhn" \
        -a "convlarge" \
        --save-path "results/svhn-labels1000-mile50-70-90-mixup-unlabel-auto-weight1-alpha0.1-mse" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "0.1" \
        --num-label "1000" \
        --total-steps "1000000" \
        --milestones "[500000, 700000, 900000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";
        
CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "svhn" \
        -a "convlarge" \
        --save-path "results/svhn-labels1000-mixup-unlabel-auto-weight1-alpha0.1-mse-swa-ci4-lrmin-1e-3" \
        --swa \
        --cycle-interval "40000" \
        --lr-min "1e-3" \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "0.1" \
        --num-label "1000" \
        --total-steps "1000000" \
        --milestones "[500000, 700000, 900000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";
        
CUDA_VISIBLE_DEVICES='2' python3 train_meta.py \
        --dataset "svhn" \
        -a "convlarge" \
        --save-path "results/svhn-labels1000-mixup-unlabel-auto-weight1-alpha0.1-mse-swa-ci4-lrmin-1e-3-log" \
        --swa \
        --cycle-interval "40000" \
        --lr-min "1e-3" \
        --log \
        --additional "unlabel" \
        --weight "1." \
        --auto-weight \
        --mix-up \
        --alpha "0.1" \
        --num-label "1000" \
        --total-steps "1000000" \
        --milestones "[500000, 700000, 900000]" \
        --lr "0.1" \
        --warmup "4000" \
        --const-steps "0" \
        --consistency "mse" \
        --print-freq "100";
        

