#!/bin/bash
PYTHON_SCRIPT="Main.py"
server="My"

#mode can be 'train' or 'test', if you want to train the model, set it to 'train', if you want to test the model, set it to 'test'
mode="train"
mode="test"
module_name='resnet'
img_size=64;test_img_size=64;databasesize=64
weight_decay=0
# regis_lr=0.001
regis_lr=0.0001

#Regularity weight of registration loss, svf_reg_w2 for TLRN and svf_reg_w for voxelmorph. They are set the same for fair comparison
svf_reg_w=0.03;svf_reg_w2=0.03  

#training expochs
max_epochs=20000 

#resmode: select to use which model, 'voxelmorph' or 'TLRN'
#dataset: select to use which dataset, 'lemniscate' or 'cine_slice_img'
#series_len: the length of the series
#reslevel: the number of the resnet blocks in the model

#Here are four examples you can run
#example 1
resmode='TLRN';dataset="lemniscate";series_len=12;reslevel=1;device=0
#example 2
# resmode='voxelmorph';dataset="lemniscate";series_len=12;reslevel=1;device=0
#example 3
# resmode='TLRN';dataset="cine_slice_img";series_len=7;reslevel=3;device=0
#example 4
# resmode='voxelmorph';dataset="cine_slice_img";series_len=7;reslevel=3;device=0

CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --mode $mode --module_name $module_name --server $server --max_epochs $max_epochs \
--resmode $resmode --weight_decay $weight_decay --reslevel $reslevel --series_len $series_len --dataset $dataset \
--svf_reg_w $svf_reg_w --regis_lr $regis_lr --img_size $img_size --test_img_size $test_img_size --databasesize $databasesize \
--svf_reg_w2 $svf_reg_w2

