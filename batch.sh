#!/bin/bash

### iterates training script over command line arguments given in file.txt (file given as first argument).

file=$1
if [[ $# -eq 0 ]] ; then
    echo 'no file argument given'
    exit 0
fi
IFS=$'\n'
m=$(cat $file | wc -l) # number of arguments in file.txt
N=$(nvidia-smi -L | wc -l) # number of cuda devices
i=0
for ((i=0; i<$m; i++)); do
  line=$(sed -n "$((i+1))p" $file)
  if [[ "$line" != +(*"&"*|*"#"*) ]]; then
    sed -n "$((i+1))p" $file | xargs python train.py
  fi
done
wait
