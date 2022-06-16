#!/bin/bash
num_epochs=1
use_cb=True
dataset_id=0
# run_name=
# echo hi && echo hey echo ho
python ./main.py  -e $num_epochs $use_cb $dataset_id
# tensorboard ./dir/$run_name