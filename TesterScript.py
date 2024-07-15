import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import Hausdorff_distance, dice_coefficient, Mgridplot, drawImage, Cjacobian_determinant4
import SimpleITK as sitk
from PIL import Image,ImageDraw,ImageFont
import matplotlib.font_manager as fm
from scipy.ndimage import zoom
from datasets.datasetDec12 import DENSE


def tester(args, model):
    db_test = DENSE(path="./datasets/lemniscate_example_series.mat", split="test", testlen=70, cur_img_size=args.img_size, resmode=args.resmode, series_len=args.series_len, basesize=args.databasesize, pathmask=args.pathmask)
    testloader = DataLoader(db_test, batch_size=7, shuffle=False)
    model.eval()

   
    count = 0
    for i_batch, sampled_batch in enumerate(testloader):
        if args.resmode == "voxelmorph":
            src, tar = sampled_batch['src'], sampled_batch['tar']  #[5, 1, 16, 64, 64]
            b,b,w,h = src.shape
            src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[16, 1, 64, 64]
            if "mask_src" in sampled_batch.keys():
                mask_src, mask_tar = sampled_batch['mask_src'], sampled_batch['mask_tar']
                mask_src = mask_src.reshape(-1,w,h).unsqueeze(1).cuda()
                mask_tar = mask_tar.reshape(-1,w,h).unsqueeze(1).cuda()
                input_mask = torch.cat((mask_src, mask_tar), dim=1)
            else:
                input_mask = None

            input = torch.cat((src,tar),dim=1)

            #get the deformed images, velocity and deformation field 
            Sdef, v, u_seq, Sdef_mask_series, ui_seq = model(input, resmode = args.resmode, masks=input_mask)   # v:[b, 3, 64, 64]
            

            #reshape to visualize
            Sdef_series = Sdef.reshape(-1,(args.series_len-1),1,w,w)
            v_series = v.reshape(-1,(args.series_len-1),2,w,w)
            u_seq = [u.permute(0,3,1,2) for u in u_seq]
            u_series = [u.reshape(-1,(args.series_len-1),2,w,w) for u in u_seq]
            ui_seq = [ui.permute(0,3,1,2) for ui in ui_seq]
            ui_series = [ui.reshape(-1,(args.series_len-1),2,w,w) for ui in ui_seq]
            tar = tar.reshape(-1,(args.series_len-1),1,w,w)
            src = src.reshape(-1,(args.series_len-1),1,w,w)
            if Sdef_mask_series is not None:
                Sdef_mask_series = Sdef_mask_series.reshape(-1,(args.series_len-1),1,w,w)
                mask_src = mask_src.reshape(-1,(args.series_len-1),1,w,w)
                mask_tar = mask_tar.reshape(-1,(args.series_len-1),1,w,w)

            
            #get the deformed images, velocity and deformation field in the sequence one by one
            for tag in range(src.shape[0]):  #batch size
                for i in range(Sdef_series.shape[1]):   
                    phiinv = u_series[-1][tag,i:i+1] 
                    phiinv = phiinv.permute(0,2,3,1)  
                    phiinv = phiinv.detach().cpu().numpy()
                    phiinv = phiinv[0]

                    v_ = v_series[tag,i:i+1]   
                    v_ = v_.permute(0,2,3,1)  
                    u_ = u_series[-1][tag,i:i+1]                        
                    ui_ = ui_series[-1][tag,i:i+1]


                    Sdef_ = Sdef_series[tag][i].detach().cpu().numpy()   
                    tar_ = tar[tag,i].cpu().numpy()  
                    diff_ = np.abs(Sdef_ - tar_)
                    v_ = torch.cat((v_, torch.zeros_like(v_)[...,0:1]), dim=-1)  
                    v_ = v_.detach().cpu().numpy()


                    if Sdef_mask_series is not None:
                        Sdef_mask = Sdef_mask_series[tag][i]   
                        tar_mask = mask_tar[tag,i]  #

                        Sdef_mask = Sdef_mask.detach().cpu().numpy()
                        tar_mask = tar_mask.detach().cpu().numpy()
                        
                   
                    
        

        elif args.resmode == "TLRN":
            slices = sampled_batch['series']
            b,b,c,w,h = slices.shape
            slices_series = slices.reshape(-1,c,w,h).cuda()
            if "masks" in sampled_batch.keys():
                masks = sampled_batch['masks']
                masks_series = masks.reshape(-1,c,w,h).cuda()
            else:
                masks_series = None


            Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = model(slices_series, resmode = "TLRN", masks=masks_series)

            #get the deformed images, velocity and deformation field in the sequence one by one
            for tag in range(slices_series.shape[0]): #batch size
                for i in range(len(Sdef_series)):   #sequence length
                    Sdef_ = Sdef_series[i][tag].detach().cpu().numpy()   #1,64,64   deformed image
                    tar_ = slices_series[tag,i+1:i+2,:,:].cpu().numpy()  #1,64,64   target image
                    phiinv = u_series[i][tag].detach().cpu().numpy()   #[64, 64, 2] transformation field
                    
                    if Sdef_mask_series is not None:
                        Sdef_mask = Sdef_mask_series[i][tag]   #1,64,64
                        tar_mask = masks_series[tag,i+1:i+2,:,:]  #1,64,64

                    diff_ = np.abs(Sdef_ - tar_)   #difference between deformed image and target image
                    v_ = v_series[i][tag:tag+1]   #[1, 64, 64, 2] velocity field
                    v_ = v_.permute(0,2,3,1)  #1,64,64,2 velocity field

                    v_ = torch.cat((v_, torch.zeros_like(v_)[...,0:1]), dim=-1)  #1,64,64,3 velocity field
                    v_ = v_.detach().cpu().numpy() #1,64,64,3 velocity field

                    
    
                    

