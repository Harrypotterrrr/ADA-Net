#!/bin/bash

DATASET_PATH=$2

if [[ $1 == "local" ]]; then
  var="python main.py --parallel True --train False --gpus 0 1 --num_workers 8 --pretrained_model 9600
  "
  echo $var
  exec $var

elif [[ $1 == "vllab4" ]]; then
  var="python main.py --parallel True --train False --gpus 0 1 2 3 --num_workers 8 --pretrained_model 9600
  "
  echo $var
  exec $var

fi
