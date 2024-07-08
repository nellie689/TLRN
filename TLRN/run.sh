#!/bin/bash
# 设置要调用的 Python 脚本路径和文件名
PYTHON_SCRIPT="Main.py"

# 在这里设置传递给 Python 脚本的参数（可选）
server="My"
mode="train"
# mode="test"
module_name='resnet'

weight_decay=0
# regis_lr=0.001
regis_lr=0.0001
svf_reg_w=0.01;svf_reg_w2=0.01
# svf_reg_w=0.03;svf_reg_w2=0.03

max_epochs=20000 

resmode='TLRN';dataset="lemniscate12002_nopretrain_cat";series_len=12;reslevel=1;device=0
# resmode='voxelmorph';dataset="lemniscate12002_nopretrain_cat";series_len=12;reslevel=1;device=0

# resmode='TLRN';dataset="cine_slice_img_reversed_cat";series_len=7;reslevel=3;device=0
# resmode='voxelmorph';dataset="cine_slice_img_reversed_cat";series_len=7;reslevel=3;device=0

img_size=64;test_img_size=64
databasesize=64

CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --mode $mode --module_name $module_name --server $server --max_epochs $max_epochs \
--resmode $resmode --weight_decay $weight_decay --reslevel $reslevel --series_len $series_len --dataset $dataset \
--svf_reg_w $svf_reg_w --regis_lr $regis_lr --img_size $img_size --test_img_size $test_img_size --databasesize $databasesize \
--svf_reg_w2 $svf_reg_w2

