#!/bin/sh

mkdir -p ./celeba
./spn-data -v -c celeba_bin_small.yaml read write image "./celeba/%n_%l.png"
