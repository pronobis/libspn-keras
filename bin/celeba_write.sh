#!/bin/sh

if [ "$#" -ne 1 ]
then
    echo "Usage: $0 <data_conf_yaml>"
    exit 1
fi
data_conf_yaml="$1"

mkdir -p ./celeba
./spn-data -v -c "$data_conf_yaml" read write image "./celeba/%n_%l.png"
