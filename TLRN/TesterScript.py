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
    db_test = DENSE(path=args.test_dense, split="test", testlen=200, cur_img_size=args.img_size, resmode=args.resmode, series_len=args.series_len, basesize=args.databasesize, pathmask=args.pathmask)
    testloader = DataLoader(db_test, batch_size=7, shuffle=False)
    imagesize = args.test_img_size
    scale=5; 
    gap = imagesize*scale + 4
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),10*scale)
    model.eval()
    result_visul = args.visjpg

    diff_list = []
    dice_list = []
    hausdorff_list = []
    det_list = []
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
            Sdef, v, u_seq, Sdef_mask_series, ui_seq = model(input, resmode = args.resmode, masks=input_mask)   # v:[16, 3, 64, 64]
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

            
            num_tags = src.shape[0] if src.shape[0]<= 10 else 10
            if args.debug:
                res = Image.new('RGB', (gap*5,  gap*7*num_tags),'white') #宽  高
            else:
                res = Image.new('RGB', (gap*Sdef_series.shape[1],  gap*7*num_tags),'white') #宽  高
            draw = ImageDraw.Draw(res)
            font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),10*scale)


            number = 0
            for tag in range(num_tags):  #一共：num_tags 组数据
                count += 1
                baseH = tag*gap*7                

                number += 1; diff_list_inner=[]; dice_inner=[]
                hausdorff_inner=[]; det_inner=[]


                for i in range(Sdef_series.shape[1]):   #10个
                    phiinv = u_series[-1][tag,i:i+1] #[1, 2, 64, 64]
                    phiinv = phiinv.permute(0,2,3,1)  #1,64,64,2  
                    phiinv = phiinv.detach().cpu().numpy()
                    phiinv = phiinv[0]
                    negative_count, total_count, percent = Cjacobian_determinant4(phiinv)
                    det_inner.append(percent)
                    # continue

                    v_ = v_series[tag,i:i+1]   #[1, 2, 64, 64]
                    v_ = v_.permute(0,2,3,1)  #1,64,64,2
                    u_ = u_series[-1][tag,i:i+1]                         #1,2,64,64
                    ui_ = ui_series[-1][tag,i:i+1]


                    Sdef_ = Sdef_series[tag][i].detach().cpu().numpy()   #1,64,64
                    tar_ = tar[tag,i].cpu().numpy()  #1,64,64
                   

                    if Sdef_mask_series is not None:
                        Sdef_mask = Sdef_mask_series[tag][i]   #1,64,64
                        tar_mask = mask_tar[tag,i]  #1,64,64
                        dice = dice_coefficient(Sdef_mask, tar_mask, return_mean=True)
                        hausdorff = Hausdorff_distance(Sdef_mask.detach().cpu(), tar_mask.detach().cpu())

                        ## save as sitk image
                        Sdef_mask = Sdef_mask.detach().cpu().numpy()
                        tar_mask = tar_mask.detach().cpu().numpy()
                        sitk.WriteImage(sitk.GetImageFromArray(Sdef_mask), result_visul+f"/temp/{count}_Sdef_mask_T{i}.nii.gz")
                        sitk.WriteImage(sitk.GetImageFromArray(Sdef_), result_visul+f"/temp/{count}_Sdef_T{i}.nii.gz")
                        sitk.WriteImage(sitk.GetImageFromArray(tar_), result_visul+f"/temp/{count}_tar_T{i}.nii.gz")
                        sitk.WriteImage(sitk.GetImageFromArray(tar_mask), result_visul+f"/temp/{count}_tar_mask_T{i}.nii.gz")
                    else:
                        dice = 0
                        hausdorff = 0
                    dice_inner.append(dice)
                    hausdorff_inner.append(hausdorff)

                    v_ = torch.cat((v_, torch.zeros_like(v_)[...,0:1]), dim=-1)  #1,64,64,3
                    v_ = v_.detach().cpu().numpy()
                    diff_ = np.abs(Sdef_ - tar_)
                    diff_list_inner.append(np.mean(diff_))

                    # continue
                    diff_ = np.abs(Sdef_ - tar_); diff_[0,0,0]=1;diff_[0,0,1]=0

                    pSdef = result_visul+f"/temp/{count}_Tar_T{i}.png"
                    dSdef = (tar_[0]*255).astype(np.uint8)
                    Isdef = Image.fromarray(dSdef, mode='L')
                    Isdef.save(pSdef)

                    tar_ = zoom(tar_[0], (scale, scale), order=3)*255; tarImg = Image.fromarray(tar_, mode='F'); res.paste(tarImg, box=(gap*i,baseH+gap))        
                    pSdef = result_visul+f"/temp/{count}_Sdef_T{i}.png"
                    dSdef = (Sdef_[0]*255).astype(np.uint8)
                    Isdef = Image.fromarray(dSdef, mode='L')
                    Isdef.save(pSdef)


                    Sdef_ = zoom(Sdef_[0], (scale, scale), order=3)*255; 
                    res.paste(Image.fromarray(Sdef_, mode='F'), box=(i*gap,baseH+2*gap))
                    


                    diff_ = zoom(diff_[0], (scale, scale), order=3)*255; res.paste(Image.fromarray(diff_, mode='F'), box=(i*gap,baseH+3*gap))
                    draw.text((i*gap,baseH+3*gap),"tar-def"+str(count),font=font,fill=(255,255,255))


                    v_ = zoom(v_[0], (scale, scale,1), order=3)  #v:1,64,64,3
                    vpre = (v_-np.min(v_))/(np.max(v_)-np.min(v_))*255
                    a1=vpre[...,0];a2=vpre[...,1];a3=vpre[...,2]
                    r = Image.fromarray(a1).convert('L');g = Image.fromarray(a2).convert('L');b = Image.fromarray(a3).convert('L')
                    vpre = Image.merge('RGB',(r,g,b))
                    res.paste(vpre, box=(i*gap,baseH+5*gap))   #宽   高

                    u_ = u_.detach().cpu().numpy()
                    pGrids = result_visul+f"/temp/{count}_Grids_T{i}.png"
                    Mgridplot(u_, pGrids, int(imagesize/4), int(imagesize/4), False,  dpi=imagesize, scale=scale) 
                    PDphi_t = sitk.GetArrayFromImage(sitk.ReadImage(pGrids))
                    image = drawImage(PDphi_t)
                    res.paste(image, box=(i*gap,baseH+6*gap))   #宽   高
                    ui_ = ui_.detach().cpu().numpy()
                    pGrids = result_visul+f"/temp/{count}_iGrids_T{i}.png"
                    Mgridplot(ui_, pGrids, int(imagesize/4), int(imagesize/4), False,  dpi=imagesize, scale=scale) 
                    PDphi_t = sitk.GetArrayFromImage(sitk.ReadImage(pGrids))
                    image = drawImage(PDphi_t)
                    res.paste(image, box=(i*gap,baseH+4*gap))   #宽   高

                diff_list.append(diff_list_inner); dice_list.append(dice_inner)
                det_list.append(det_inner)
                hausdorff_list.append(hausdorff_inner)
               


        elif args.resmode == "TLRN":
            slices = sampled_batch['series']    #[2, 2, 9, 64, 64] [b, 2, 9, 64, 64]
            b,b,c,w,h = slices.shape
            slices_series = slices.reshape(-1,c,w,h).cuda()  #[2, 2, 9, 64, 64]  ->  [4, 9, 64, 64]]
            if "masks" in sampled_batch.keys():
                masks = sampled_batch['masks']
                masks_series = masks.reshape(-1,c,w,h).cuda()  #[2, 2, 9, 64, 64]  ->  [4, 9, 64, 64]
            else:
                masks_series = None


            Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = model(slices_series, resmode = "TLRN", masks=masks_series)

            num_tags = slices_series.shape[0]
            if args.debug:
                res = Image.new('RGB', (gap*5,  gap*7*num_tags),'white') #宽  高
            else:
                res = Image.new('RGB', (gap*len(Sdef_series),  gap*7*num_tags),'white') #宽  高
            draw = ImageDraw.Draw(res)

            number = 0
           

            
            for tag in range(num_tags):
                count += 1
                baseH = tag*gap*7

                number += 1; diff_list_inner=[]; dice_inner=[];hausdorff_inner=[];det_inner=[]
                
                for i in range(len(Sdef_series)):   #7个
                    Sdef_ = Sdef_series[i][tag].detach().cpu().numpy()   #1,64,64
                    tar_ = slices_series[tag,i+1:i+2,:,:].cpu().numpy()  #1,64,64
                    phiinv = u_series[i][tag].detach().cpu().numpy()   #[64, 64, 2]
                    negative_count, total_count, percent = Cjacobian_determinant4(phiinv)
                    det_inner.append(percent)
                    continue
                    
                    
                    if Sdef_mask_series is not None:
                        Sdef_mask = Sdef_mask_series[i][tag]   #1,64,64
                        tar_mask = masks_series[tag,i+1:i+2,:,:]  #1,64,64
                        dice = dice_coefficient(Sdef_mask, tar_mask, return_mean=True)
                        hausdorff = Hausdorff_distance(Sdef_mask.detach().cpu(), tar_mask.detach().cpu())
                    else:
                        dice = 0
                        hausdorff = 0
                    dice_inner.append(dice)
                    hausdorff_inner.append(hausdorff)



                    v_ = v_series[i][tag:tag+1]   #[1, 64, 64, 2]
                    # print(v_.shape)  #torch.Size([1, 2, 64, 64])
                    v_ = v_.permute(0,2,3,1)  #1,64,64,2

                    debug_v_npy = v_.detach().cpu().numpy()
                    # np.save(result_visul+f"/temp/v_resnet_{debug_idx}.npy", debug_v_npy)

                    
                                            #1,64,64,2
                    v_ = torch.cat((v_, torch.zeros_like(v_)[...,0:1]), dim=-1)  #1,64,64,3
                    v_ = v_.detach().cpu().numpy()

                    # print(v_.shape, "~~~~~@@@@@@@@@~~~~~~") #(1, 64, 64, 3)
                    # assert 4>888
                    # sitk.WriteImage(sitk.GetImageFromArray(v_), result_visul+f"/temp/v_resnet_{debug_idx}.nii.gz")

                    
                    diff_ = np.abs(Sdef_ - tar_)
                    diff_list_inner.append(np.mean(diff_))


                    tar_ = zoom(tar_[0], (scale, scale), order=3)*255; tarImg = Image.fromarray(tar_, mode='F'); res.paste(tarImg, box=(gap*i,baseH+gap))
                    # draw.text((gap*i,baseH+gap),"tar",font=font,fill=(255,255,255))

                    pSdef = result_visul+f"/temp/{count}_Sdef_T{i}.png"
                    dSdef = (Sdef_[0]*255).astype(np.uint8)
                    Isdef = Image.fromarray(dSdef, mode='L')
                    Isdef.save(pSdef)


                    Sdef_ = zoom(Sdef_[0], (scale, scale), order=3)*255; res.paste(Image.fromarray(Sdef_, mode='F'), box=(i*gap,baseH+2*gap))
                    # draw.text((i*gap,baseH+2*gap),"def",font=font,fill=(255,255,255))
                    diff_ = zoom(diff_[0], (scale, scale), order=3)*255; res.paste(Image.fromarray(diff_, mode='F'), box=(i*gap,baseH+3*gap))
                    draw.text((i*gap,baseH+3*gap),"tar-def"+str(count),font=font,fill=(255,255,255))

                    v_ = zoom(v_[0], (scale, scale,1), order=3)  #v:1,64,64,3
                    vpre = (v_-np.min(v_))/(np.max(v_)-np.min(v_))*255
                    a1=vpre[...,0];a2=vpre[...,1];a3=vpre[...,2]
                    r = Image.fromarray(a1).convert('L');g = Image.fromarray(a2).convert('L');b = Image.fromarray(a3).convert('L')
                    vpre = Image.merge('RGB',(r,g,b))
                    res.paste(vpre, box=(i*gap,baseH+5*gap))   #宽   高
                    # draw.text((i*gap,baseH+5*gap),"velocity",font=font,fill=(255,255,255))


                    u_ = u_series[i][tag:tag+1]                          #1,2,64,64
                    u_ = u_.permute(0,3,1,2)                #1,64,64,2
                    u_ = u_.detach().cpu().numpy()
                    # print(u_.shape)  #(1, 2, 64, 64)
                    pGrids = result_visul+f"/temp/{count}_Grids_T{i}.png"
                    Mgridplot(u_, pGrids, int(imagesize/4), int(imagesize/4), False,  dpi=imagesize, scale=scale) 
                    PDphi_t = sitk.GetArrayFromImage(sitk.ReadImage(pGrids))
                    image = drawImage(PDphi_t)
                    res.paste(image, box=(i*gap,baseH+6*gap))   #宽   高
                    
                    ui_ = ui_series[i][tag:tag+1]                          #1,2,64,64
                    ui_ = ui_.permute(0,3,1,2)                #1,64,64,2
                    ui_ = ui_.detach().cpu().numpy()
                    pGrids = result_visul+f"/temp/{count}_iGrids_T{i}.png"
                    Mgridplot(ui_, pGrids, int(imagesize/4), int(imagesize/4), False,  dpi=imagesize, scale=scale) 
                    PDphi_t = sitk.GetArrayFromImage(sitk.ReadImage(pGrids))
                    image = drawImage(PDphi_t)
                    res.paste(image, box=(i*gap,baseH+4*gap))   #宽   高

                    
                diff_list.append(diff_list_inner); dice_list.append(dice_inner)
                hausdorff_list.append(hausdorff_inner), det_list.append(det_inner)



    #open a file 
    file1 = open(result_visul+"/diff_1.txt", "w")
    file3 = open(result_visul+"/dice.txt", "w")
    file4 = open(result_visul+"/hausdorff.txt", "w")
    file5 = open(result_visul+"/det.txt", "w")
    # write diff_list, dice_list, hausdorff_list, and det_list in to file through loop
    print(len(diff_list), len(dice_list), len(hausdorff_list), len(det_list))
    for i in range(len(diff_list)):
        print(diff_list[i], dice_list[i], hausdorff_list[i], det_list[i])
        file1.write(str(diff_list[i]) + "\n")
        file3.write(str(dice_list[i]) + "\n")
        file4.write(str(hausdorff_list[i]) + "\n")
        file5.write(str(det_list[i]) + "\n")
    print(result_visul+"/det.txt has been written")


