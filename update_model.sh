#!/bin/bash

echo "Which model to update? (A/B)"
read -r model

if [ "$model" = "A" ]; then
  MODEL_SOURCE=./purchase-prediction/output/modelA
  MODEL_DEST=./webapp/model/modelA
else
  if [ "$model" = "B" ]; then
    MODEL_SOURCE=./purchase-prediction/output/modelB
    MODEL_DEST=./webapp/model/modelB
  else
    exit 0
  fi
fi


if test -d "$MODEL_DEST"; then
  rm -rf $MODEL_DEST
fi

if test -d "$MODEL_SOURCE"; then
  cp -r $MODEL_SOURCE webapp/model
else
  echo "Model source: $MODEL_SOURCE does not exist."
fi