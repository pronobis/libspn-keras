#!/bin/sh

./spn-model -v train dense mnist "~/Data/mnist" \
            --num-epochs 100 \
            --batch-size 100 \
            --shuffle \
            --seed 100 \
            --image-format binary \
            --ratio 2 \
            --crop 2 \
            --mnist-subset=train\
            --dense-input-dist raw \
            --dense-decomps 1 \
            --dense-subsets 2 \
            --dense-mixtures 4 \
            --dense-input-mixtures 2 \
            --weight-init random \
            --value-inference marginal \
            --init-accum 20 \
            --smoothing-val 0.0 \
            --smoothing-decay 0.2 \
            --smoothing-min 0.0 \
            --stop-condition 0.05
