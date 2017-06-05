#!/bin/sh

if [ "$#" -ne 2 ]
then
    echo "Usage: $0 <data_conf_yaml> <out_dir>"
    exit 1
fi
data_conf_yaml="$1"
out_dir="$2"

mkdir -p "$out_dir"
./spn-data -v -c "$data_conf_yaml" read write image "$out_dir/%n_%l.png"
