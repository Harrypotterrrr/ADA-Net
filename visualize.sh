CUDA_VISIBLE_DEVICES="0" python3 plot-features.py \
        --dataset "svhn" \
        --checkpoint-path "results/svhn-labels1000-mile30-35-mixup0.1-wd5e-5-balanced-run2/model_best.pth" \
        --index-path "results/svhn-labels1000-mile30-35-mixup0.1-wd5e-5-balanced-run2/label_indices.txt" \
        --save-path "results/svhn-labels1000-mile30-35-mixup0.1-wd5e-5-balanced-run2/visualization"
