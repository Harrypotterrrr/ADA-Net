CUDA_VISIBLE_DEVICES="1" python3 plot-features.py \
        --dataset "cifar10" \
        --checkpoint-path "results/cifar10-labels4000-baseline-mile30-35-wd1e-4-balanced/model_best.pth" \
        --index-path "results/cifar10-labels4000-baseline-mile30-35-wd1e-4-balanced/label_indices.txt" \
        --save-path "results/cifar10-labels4000-baseline-mile30-35-wd1e-4-balanced/visualization" \
        --aligned-path "results/cifar10-labels4000-baseline-mile30-35-wd1e-4-balanced/visualization/aligned.npy" \
        --num-point '5000';
 
CUDA_VISIBLE_DEVICES="1" python3 plot-features.py \
        --dataset "cifar10" \
        --checkpoint-path "results/cifar10-labels4000-mile30-35-mixup1-wd1e-4-balanced/model_best.pth" \
        --index-path "results/cifar10-labels4000-mile30-35-mixup1-wd1e-4-balanced/label_indices.txt" \
        --save-path "results/cifar10-labels4000-mile30-35-mixup1-wd1e-4-balanced/visualization" \
        --aligned-path "results/cifar10-labels4000-mile30-35-mixup1-wd1e-4-balanced/visualization/aligned.npy" \
        --num-point '5000';
