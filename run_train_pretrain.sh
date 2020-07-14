#!/bin/bash

if [ -n "$1" ]; then
  ID=$1
else
  LAST_EPOCH=$(find save_weights/ -name contran-*.model |
      sed "s/save\_weights\/contran\-\([0-9][0-9]*\)\.model/\1/" |
      sort -n |
      tail -n 1)
  echo "Starting training at epoch $LAST_EPOCH"
  ID=$LAST_EPOCH
fi

export CUDA_VISIBLE_DEVICES=0
python3 main_run.py $ID
