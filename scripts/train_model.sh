#!/bin/bash

DATASET_PATH=$2

if [[ $1 == "local" ]]; then
  var="python main.py --parallel True --gpus 0 1 --num_workers 8 --use_tensorboard True
  "
  echo $var
  exec $var

elif [[ $1 == "vllab4" ]]; then
  var="python main.py --parallel True --gpus 0 1 2 3 --num_workers 8 --use_tensorboard True
  "
  echo $var
  exec $var

fi
