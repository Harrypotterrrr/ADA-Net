CUDA_VISIBLE_DEVICES='0' python3 train_meta_swa.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-lr0.1-fi84-ci12-num33-warmup4-const0-mixup1" \
        --mix-up \
        --alpha "1." \
        --num-label "4000" \
        --epsilon "1e-2" \
        --lr "0.1" \
        --full-interval "84000" \
        --cycle-interval "12000" \
        --num-cycles "33" \
        --fastswa-freq "1200" \
        --warmup "4000" \
        --const-steps "0" \
        --print-freq "20";

CUDA_VISIBLE_DEVICES='1' python3 train_meta_swa.py \
        --dataset "cifar10" \
        --save-path "results/cifar10-labels4000-lr0.1-fi84-ci12-num33-warmup4-const160-minup1" \
        --mix-up \
        --alpha "1." \
        --num-label "4000" \
        --epsilon "1e-2" \
        --lr "0.1" \
        --full-interval "84000" \
        --cycle-interval "18000" \
        --num-cycles "15" \
        --fastswa-freq "1200" \
        --warmup "4000" \
        --const-steps "160000" \
        --print-freq "20";

CUDA_VISIBLE_DEVICES='2' python3 train_meta_swa.py \
        --dataset "svhn" \
        --save-path "results/svhn-labels4000-lr0.1-fi84-ci12-num33-warmup4-const0-mixup0.1" \
        --mix-up \
        --alpha ".1" \
        --num-label "4000" \
        --epsilon "1e-2" \
        --lr "0.1" \
        --full-interval "84000" \
        --cycle-interval "12000" \
        --num-cycles "33" \
        --fastswa-freq "1200" \
        --warmup "4000" \
        --const-steps "0" \
        --print-freq "20";

CUDA_VISIBLE_DEVICES='3' python3 train_meta_swa.py \
        --dataset "svhn" \
        --save-path "results/svhn-labels4000-lr0.1-fi84-ci18-num15-warmup4-const160-mixup0.1" \
        --mix-up \
        --alpha ".1" \
        --num-label "4000" \
        --epsilon "1e-2" \
        --lr "0.1" \
        --full-interval "84000" \
        --cycle-interval "18000" \
        --num-cycles "15" \
        --fastswa-freq "1200" \
        --warmup "4000" \
        --const-steps "160000" \
        --print-freq "20";

