CUDA_VISIBLE_DEVICES='1' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.1-inile32-48-type0-label1" \
                                                        --use-label \
                                                        --unlabel-weight "1." \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.1" \
                                                        --inner-milestones "[32000, 48000]" \
                                                        --type "0" \
                                                        --print-freq "20";

CUDA_VISIBLE_DEVICES='1' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.01-inile32-48-type0-label1" \
                                                        --use-label \
                                                        --unlabel-weight "1." \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.01" \
                                                        --inner-milestones "[32000, 48000]" \
                                                        --type "0" \
                                                        --print-freq "20";

CUDA_VISIBLE_DEVICES='1' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.001-inile32-48-type0-label1" \
                                                        --use-label \
                                                        --unlabel-weight "1." \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.001" \
                                                        --inner-milestones "[32000, 48000]" \
                                                        --type "0" \
                                                        --print-freq "20";

