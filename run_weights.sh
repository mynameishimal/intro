#!/bin/bash
num_epoch=40
use_cb=True
dataset_id=0
num_kernels=32
num_hidden=64
num_kern_array=32
# num_kern_array=( 128 )

num_hidd_nodes_array=128
# run_name=
# echo hi && echo hey echo ho
# python ./findingWeights.py  $num_epochs $use_cb $dataset_id
# tensorboard ./dir/$run_name
# python ./findingWeights.py -e $num_epoch -d $dataset_id -k $num_kernels -l $num_hidden  
# python ./findingWeights.py $num_epochs $use_cb $dataset_id $num_kernels $num_hidden

python ./findingWeights.py -e $num_epoch \
 -d $dataset_id -k $num_kern_array -l $num_hidden


