#!/bin/sh

mkdir -p ./mnist
./spn-data -v -c mnist_bin_small.yaml read write image "./mnist/%n_%l.png"
