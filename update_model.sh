#!/bin/bash

MODEL_SOURCE=./purchase-prediction/output/model
MODEL_DEST=./webapp/model

if test -d "$MODEL_DEST"; then
  rm -rf webapp/model
fi

if test -d "$MODEL_SOURCE"; then
  cp -r purchase-prediction/output/model webapp
else
  echo "Model source: $MODEL_SOURCE does not exist."
fi