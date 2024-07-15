# -*- coding: utf-8 -*-
import os
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from TrainScript import trainer
from TesterScript import tester
from networks.core_model_resnet import Net2DResNet   #skip-connect
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
""" ~~~~~~~~~    basic setting ~~~~~~~~~~~~"""
parser.add_argument('--mode', type=str, default='test', help='train or test') 
parser.add_argument('--server', type=str, default='My', help='server name')
parser.add_argument('--debug', type=bool, default=False, help='--')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

""" ~~~~~~~~~    about training phase ~~~~~~~~~~~~"""
parser.add_argument('--loss_type', type=str, default='MSE', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--regis_lr', type=float,  default=0.0005, help='lr of unet')
parser.add_argument('--regis_stp', type=float,  default=100, help='step of unet')
parser.add_argument('--regis_gamma', type=float,  default=0.5, help='gamma of unet')
parser.add_argument('--weight_decay', type=float, default=0, help='--')
parser.add_argument('--svf_reg_w2', type=float, default='0', help='--')           ### noskip + noavg
parser.add_argument('--svf_simi_w', type=float, default=1.0, help='--')
parser.add_argument('--svf_reg_w', type=float, default=0.01, help='--')
parser.add_argument('--pred_num_steps', type=int, default=99999, help='5')

""" ~~~~~~~~~    network setting ~~~~~~~~~~~~"""
parser.add_argument('--series_len', type=int, default=3, help='3')
parser.add_argument('--one_residue_block', type=bool, default=False, help='3')  
parser.add_argument('--reslevel', type=int, default=1, help='--')

""" ~~~~~~~~~    about testing phase ~~~~~~~~~~~~"""
parser.add_argument('--visdir', type=str, default='1', help='visdir') 
parser.add_argument('--testlen', type=int, default=20, help='experiment_name')
parser.add_argument('--num_steps', type=int, default=7, help='5')

""" ~~~~~~~~~    import arguments~~~~~~~~~~~~"""
parser.add_argument('--module_name', type=str, default='resnet', help='the name of module')
parser.add_argument('--resmode', type=str, default='TLRN', help='TLRN or voxemorph')
parser.add_argument('--dataset', type=str, default='dense_addinfull', help='experiment_name')
parser.add_argument('--test_type', type=str, default='hollowquickdraw', help='pretrain')
parser.add_argument('--img_size', type=int, default=128, help='input img_size')
parser.add_argument('--test_img_size', type=int, default=128, help='input img_size')
parser.add_argument('--databasesize', type=int, default=128, help='input img_size')
args = parser.parse_args()

################################################################
# configs
################################################################
imagesize = args.img_size
batch_size = args.batch_size
TSteps = args.num_steps
if (args.pred_num_steps == 99999):
    args.pred_num_steps = args.num_steps
if (args.test_img_size == 99999):
    args.test_img_size = args.img_size
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


if __name__ == "__main__":
    HOME = "WORK_ROOT_DIRECTORY"
    args.HOME = HOME
    
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    # torch.autograd.set_detect_anomaly(True)


    dataset_name = args.dataset
    module_name = args.module_name


    args.nb_features=[[16, 32, 32], [32, 32, 32, 32, 16, 16]]

    #DATA PATH
    dataset_config = {
        "lemniscate": "DataPath for lemniscate", #HOME+ '/datasets/lemniscate_example_series.mat' ,
        "cine_slice_img": "DataPath for cine",#HOME+ '/datasets/revserse_cine64_12slice.mat' ,
        "reversed_cine_slice_mask": "DataPath for cine mask",#HOME+ '/datasets/revserse_cine64_12slice_mask.mat' ,
    }

    
    if "cine_slice_img_reversed" in dataset_name:
        args.train_dense = dataset_config['cine_slice_img']
        args.test_dense = dataset_config['cine_slice_img']
        args.pathmask = dataset_config['reversed_cine_slice_mask']
    elif "lemniscate" in dataset_name:
        args.train_dense = dataset_config['lemniscate']
        args.test_dense = dataset_config['lemniscate']
        args.pathmask = None
   
    

    args.exp = dataset_name + str(args.img_size)
    #set the dirctory to save the model
    snapshot_path =  HOME+"/TLRN/models/{}/".format(args.exp)    
    snapshot_path += args.module_name
    snapshot_path = snapshot_path + '_' + str(args.loss_type) if args.loss_type == 'NCC' else snapshot_path
    snapshot_path = snapshot_path + '_Tsteps' + str(args.num_steps)
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs)
    snapshot_path = snapshot_path + '_regis_lr' + str(args.regis_lr)
    snapshot_path = snapshot_path + "_weight_decay_" + str(args.weight_decay)
    snapshot_path = snapshot_path + "_resmode" + str(args.resmode)
    snapshot_path = snapshot_path + "_one_residue_block" if args.one_residue_block else snapshot_path
    snapshot_path = snapshot_path + "_reslevel_" + str(args.reslevel)
    snapshot_path = snapshot_path + "_series_len_" + str(args.series_len) if args.series_len !=0 else snapshot_path
    snapshot_path = snapshot_path + "_svf_simi_w_" + str(args.svf_simi_w) if args.svf_simi_w !=1.0 else snapshot_path
    snapshot_path = snapshot_path + "_svf_reg_w_" + str(args.svf_reg_w) if args.svf_reg_w !=1.0 else snapshot_path
    snapshot_path = snapshot_path + "_svf_reg_w2_" + str(args.svf_reg_w2) if args.svf_reg_w2 !=1.0 else snapshot_path

    args.snapshot_path = snapshot_path
   
    print("snapshot_path   ", snapshot_path)
    if args.mode == "train" and (not os.path.exists(snapshot_path)):
        os.makedirs(snapshot_path)

    ######## make log directory
    writerdir = snapshot_path.split('/'); writerdir.insert(-1,'log'); writerdir='/'.join(writerdir)
    writer = SummaryWriter(writerdir)
    args.writer = writer
    
    ##MyVs lagomorh lddmm 
    trainer_dic = {'resnet': trainer}
    tester_dic = {'resnet': tester}
    netdic = {'resnet': Net2DResNet(args).cuda()}

    net = netdic[module_name]
    if args.mode =="train":
        #start to train
        trainer_dic[module_name](args, net, snapshot_path)

    elif args.mode =="test":
        #start to test
        save_mode_path_TLRN = "./models/registration_TLRN.pth"
        save_mode_path_VM = "./models/registration_voxelmorph.pth"  
        if args.resmode == "voxelmorph":
            net.load_state_dict(torch.load(save_mode_path_VM,map_location='cpu')['registration'])
        elif args.resmode == "TLRN":
            net.load_state_dict(torch.load(save_mode_path_TLRN,map_location='cpu')['registration'])
    
        # set the dircetory to save the visualization results
        args.visdir = f"{args.test_type}/{args.test_img_size}"
        args.visjpg = snapshot_path+"/view_res/"+args.visdir
        if not os.path.exists(args.visjpg):
            os.makedirs(args.visjpg)
        if not os.path.exists(args.visjpg+"/temp"):
            os.makedirs(args.visjpg+"/temp")
            
        tester_dic[module_name](args, net)