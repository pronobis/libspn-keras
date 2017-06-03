#!/bin/sh

if [ "$#" -ne 2 ]
then
    echo "Usage: $0 <data_conf_yaml> <model_conf_yaml>"
    exit 1
fi
data_conf_yaml="$1"
model_conf_yaml="$2"

./spn-model -v -c "$data_conf_yaml" -c "$model_conf_yaml" build train save model.spn --pretty=true test
