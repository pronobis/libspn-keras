#!/bin/sh

./spn-model -v -c mnist_bin_small.yaml -c model_discrete_dense.yaml build train test
