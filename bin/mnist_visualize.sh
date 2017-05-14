#!/bin/sh

mnist_dir="~/Data/mnist"

mkdir -p ./mnist
./spn-data -v read mnist $mnist_dir \
           --ratio 1 \
           --crop 4 \
           --image-format binary \
           visualize
