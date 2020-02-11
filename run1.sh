CUDA_VISIBLE_DEVICES='0' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.1-inile32-48-type0" \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.1" \
                                                        --inner-milestones "[32000, 48000]" \
                                                        --type "0" \
                                                        --print-freq "20";

CUDA_VISIBLE_DEVICES='0' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.1-inile24-40-56-type0" \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.1" \
                                                        --inner-milestones "[24000, 40000, 56000]" \
                                                        --type "0" \
                                                        --print-freq "20";


CUDA_VISIBLE_DEVICES='0' python3 train_meta_learning.py --save-path "results/principled-meta-learning-labels4000-epsilon1e-2-lr0.1-mile32-48-ilr0.1-igm0.316-inile16-32-40-48-56-type0" \
                                                        --total-steps "64000" \
                                                        --num-label "4000" \
                                                        --epsilon "1e-2" \
                                                        --lr "0.1" \
                                                        --milestones "[32000, 48000]" \
                                                        --inner-lr "0.1" \
                                                        --inner-gamma "0.316" \
                                                        --inner-milestones "[16000, 32000, 40000, 48000, 56000]" \
                                                        --type "0" \
                                                        --print-freq "20";
