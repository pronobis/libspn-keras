#!/bin/sh

ratio=4
crop=1
image_format="binary"

./spn-model -v \
            build discrete_dense \
            --weight-init=random \
            --dense-input-dist=raw \
            --dense-decomps=1 \
            --dense-subsets=2 \
            --dense-mixtures=4 \
            --dense-input-mixtures=2 \
            train em mnist "~/Data/mnist" \
            --num-epochs=100 \
            --batch-size=100 \
            --allow_smaller_batch=true \
            --shuffle=true \
            --seed=100 \
            --image-format=$image_format \
            --ratio=$ratio \
            --crop=$crop \
            --mnist-subset=train\
            --value-inference=marginal \
            --init-accum=20 \
            --smoothing-val=0.0 \
            --smoothing-decay=0.2 \
            --smoothing-min=0.0 \
            --stop-condition=0.05 \
            test mnist "~/Data/mnist" \
            --batch-size=100 \
            --image-format=$image_format \
            --ratio=$ratio \
            --crop=$crop \
            --mnist-subset=test\
