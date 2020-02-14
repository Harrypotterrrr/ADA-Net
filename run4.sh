CUDA_VISIBLE_DEVICES='3' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.01-inile32-48-type1-label1" \
                                                        --use-label \
                                                        --unlabel-weight "1." \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.01" \
                                                        --inner-milestones "[32000, 48000]" \
                                                        --type "1" \
                                                        --print-freq "20";

CUDA_VISIBLE_DEVICES='3' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.01-inile32-48-type2-label1" \
                                                        --use-label \
                                                        --unlabel-weight "1." \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.01" \
                                                        --inner-milestones "[32000, 48000]" \
                                                        --type "2" \
                                                        --print-freq "20";

CUDA_VISIBLE_DEVICES='3' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.01-inile32-48-type3-label1" \
                                                        --use-label \
                                                        --unlabel-weight "1." \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.01" \
                                                        --inner-milestones "[32000, 48000]" \
                                                        --type "3" \
                                                        --print-freq "20";

