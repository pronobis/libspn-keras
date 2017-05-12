#!/bin/sh

mnist_dir="~/Data/mnist"

mkdir -p ./mnist
./spn-data -v read mnist $mnist_dir write image "./mnist/%n_%l.png"
