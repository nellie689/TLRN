import numpy as np
import torch
import torch.nn as nn
from PIL import Image,ImageDraw,ImageFont
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode=None):
        if mode is None:
            mode = self.mode
        # new locations
        new_locs = self.grid + flow   #self.grid:  identity
        new_locs_unnormalize = self.grid + flow
        shape = flow.shape[2:]
        #  new_locs  :  torch.Size([1, 3, 64, 64, 64])
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 1) #[1, 64, 64, 64,3]
            new_locs_unnormalize = new_locs_unnormalize[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 4, 1)
            new_locs_unnormalize = new_locs_unnormalize[..., [2, 1, 0]]

        warped = F.grid_sample(src, new_locs, mode=mode)
        # print(new_locs.shape)   #[b, 64, 64, 64, 3]
        # print(warped.shape)     #[6, 3, 64, 64, 64]
        # return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

       
        return (warped, new_locs_unnormalize)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, registration=False):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)
        self.registration = registration
    def forward(self, vec):  ###速度场->形变场
        dispList = []

        vec = vec * self.scale
        dispList.append(vec)

        for _ in range(self.nsteps):
            scratch,_ = self.transformer(vec, vec)
            vec = vec + scratch
            dispList.append(vec)
        # print("vec ", vec.requires_grad)
        if not self.registration:
            return vec
        else:
            return vec, dispList


class Svf(nn.Module):
    def __init__(self, inshape, steps=7):
        super().__init__()
        self.nsteps = steps
        assert self.nsteps >= 0, 'nsteps should be >= 0, found: %d' % self.nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)
    

    # def integrate(self, pos_flow):
    # def Svf_shooting(self, pos_flow):  #pos_flow: [b, 2, 64, 64]  (b,64,64,2)
    def forward(self, pos_flow):  #pos_flow: [b, 2, 64, 64]  (b,64,64,2)
        dims = len(pos_flow.shape)-2
        if dims == 2:
            b,c,w,h = pos_flow.shape
            if c != 2 and c != 3:
                pos_flow = pos_flow.permute(0,3,1,2)
        elif dims == 3:
            b,c,w,h,d = pos_flow.shape
            if c != 3:
                pos_flow = pos_flow.permute(0,4,1,2,3)

        vec = pos_flow
        dispList = []
        
        vec = vec * self.scale
        dispList.append(vec)


        for _ in range(self.nsteps):
            scratch,_ = self.transformer(vec, vec)
            vec = vec + scratch
            dispList.append(vec)

            # print(vec.shape)     #[70, 2, 64, 64]
            # assert 4>8888
        
        return vec, dispList   #len

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        if(len(y_pred.shape) == 5):
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0

            if self.loss_mult is not None:
                grad *= self.loss_mult
            return grad
        elif(len(y_pred.shape) == 4):
            # print("y_pred   ",y_pred.shape)
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
          
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0

            if self.loss_mult is not None:
                grad *= self.loss_mult
            return grad






def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        from pycuda import gpuarray
        if isinstance(arr, gpuarray.GPUArray):
            return arr.get()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
    except ImportError:
        pass

    raise Exception(f"Cannot convert type {type(arr)} to numpy.ndarray.")

def Mgridplot(u, Hpath, Nx=64, Ny=64, displacement=True, color='red', dpi=128, scale=1, linewidth=0.2,**kwargs):
    """Given a displacement field, plot a displaced grid"""
    u = to_numpy(u)

    assert u.shape[0] == 1, "Only send one deformation at a time"
   
    # plt.figure(dpi= 128)
    plt.figure(figsize=(1,1))
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    

    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[0,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])

    # now reset to actual Nx Ny that we achieved
    Nx = h.shape[1]
    Ny = h.shape[2]
    # adjust displacements for downsampling
    h[0,...] /= float(u.shape[2])/Nx
    h[1,...] /= float(u.shape[3])/Ny

    if displacement: # add identity
        '''
            h[0]: 
        '''
        h[0,...] += np.arange(Nx).reshape((Nx,1))  #h[0]:  (118, 109)  add element: 118*1
        h[1,...] += np.arange(Ny).reshape((1,Ny))

    # put back into original index space
    h[0,...] *= float(u.shape[2])/Nx
    h[1,...] *= float(u.shape[3])/Ny
    # create a meshgrid of locations
    for i in range(h.shape[1]):
        plt.plot( h[0,i,:], h[1,i,:], color=color, linewidth=linewidth, **kwargs)
    for i in range(h.shape[2]):
        plt.plot(h[0,:,i], h[1,:,i],  color=color, linewidth=linewidth, **kwargs)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    # plt.savefig(Hpath,dpi= dpi*20)
    plt.savefig(Hpath,dpi= dpi*scale,transparent=True)
    # plt.savefig(Hpath, dpi= dpi*20, transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.cla()
    plt.clf()
    plt.close()
    plt.close('all')

def identity(defshape, dtype=np.float32):
    """
    Given a deformation shape in NCWH(D) order, produce an identity matrix (numpy array)
    """
    dim = len(defshape)-2
    ix = np.empty(defshape, dtype=dtype)
    for d in range(dim):
        ld = defshape[d+2]
        shd = [1]*len(defshape)
        shd[d+2] = ld
        ix[:,d,...] = np.arange(ld, dtype=dtype).reshape(shd)
    return ix

def Mquiver(u, Nx=32, Ny=32, color='black', units='xy', angles='xy', scale=1.0, **kwargs):
    """Given a displacement field, plot a quiver of vectors"""
    u = to_numpy(u)
    assert u.shape[0] == 1, "Only send one deformation at a time"
    assert u.ndim == 4, "Only 2D deformations can use quiver()"
    from matplotlib import pyplot as plt
    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[:,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])
    ix = identity(u.shape, u.dtype)[:,:,::u.shape[2]//Nx, ::u.shape[3]//Ny]
    # create a meshgrid of locations
    plt.quiver(ix[0,1,:,:], ix[0,0,:,:], h[0,1,:,:], h[0,0,:,:], color=color,
               angles=angles, units=units, scale=scale, **kwargs)
    plt.axis('equal')
    # plt.gca().invert_yaxis()
    plt.show()

def drawImage(GTphiinv):
    GTphiinv = GTphiinv.astype(np.float32)
    a1=GTphiinv[...,0]
    a2=GTphiinv[...,1]
    a3=GTphiinv[...,2]
    r = Image.fromarray(a1).convert('L')
    g = Image.fromarray(a2).convert('L')
    b = Image.fromarray(a3).convert('L')
    GTphiinv = Image.merge('RGB',(r,g,b))
    GTphiinv = GTphiinv.convert('RGB')
    return GTphiinv

import cv2
class imsave_edge():
    def __init__(self):
        self.kernel = self.get_kernel()

    def get_kernel(self):
        kernel = np.array([[-1, -1, -1], [-1, 7.5, -1], [-1, -1, -1]], dtype=np.float32)
        return kernel

    def change_gray(self, inputs, min_value=-200, max_value=800):
        outputs = np.array(inputs, dtype=np.float32)
        outputs[outputs > max_value] = max_value
        outputs[outputs < min_value] = min_value
        outputs = (outputs - min_value) / (max_value - min_value)
        return outputs

    def get_edge(self, seg):
        outputs = cv2.filter2D(seg, -1, self.kernel)
        outputs = np.sign(outputs)
        return outputs

def Hausdorff_distance(tensor_a, tensor_b):
    while tensor_a.shape[0] == 1:
        tensor_a = tensor_a.squeeze(0)
    while tensor_b.shape[0] == 1:
        tensor_b = tensor_b.squeeze(0)

    getEdge = imsave_edge()
    tensor_a = tensor_a.clone().numpy().astype(np.float32)
    tensor_b = tensor_b.clone().numpy().astype(np.float32)
    # swich to numpy
    edge_a = getEdge.get_edge(tensor_a)
    edge_b = getEdge.get_edge(tensor_b)

    position_a = np.where(edge_a == 1)
    position_b = np.where(edge_b == 1)

    xyz_a = np.array([position_a[0], position_a[1]]).T
    xyz_b = np.array([position_b[0], position_b[1]]).T

    # print(xyz_a.shape, xyz_b.shape) (55777, 3) (57202, 3)

    # distances1to2 = torch.cdist(torch.tensor(xyz_a, dtype=torch.float32), torch.tensor(xyz_b, dtype=torch.float32)).min(dim=1).values
    # distances2to1 = torch.cdist(torch.tensor(xyz_b, dtype=torch.float32), torch.tensor(xyz_a, dtype=torch.float32)).min(dim=1).values

    distances1to2 = torch.cdist(torch.tensor(xyz_a, dtype=torch.float32).cuda(), torch.tensor(xyz_b, dtype=torch.float32).cuda())
    distances2to1 = torch.cdist(torch.tensor(xyz_b, dtype=torch.float32).cuda(), torch.tensor(xyz_a, dtype=torch.float32).cuda())

    # print(distances1to2.shape, distances2to1.shape)   torch.Size([55777, 57202]) torch.Size([57202, 55777])

    distances1to2 = torch.min(distances1to2, dim=1).values
    distances2to1 = torch.min(distances2to1, dim=1).values

    # print(distances1to2.shape, distances2to1.shape)   torch.Size([55777]) torch.Size([57202])


    hausdorff_distance = torch.max(torch.max(distances1to2), torch.max(distances2to1))

    # print(hausdorff_distance)   tensor(5.3852)
    #delete data
    del edge_a, edge_b, position_a, position_b, xyz_a, xyz_b, distances1to2, distances2to1, tensor_a, tensor_b
    return hausdorff_distance.item()

def dice_coefficient(tensor_a, tensor_b, return_mean=True):
    # print("dice_tensor_shape:  ",tensor_a.shape, tensor_b.shape)    #[10, 1, 128, 128] [10, 1, 128, 128]
    tensor_a = tensor_a.clone()
    tensor_b = tensor_b.clone()

    tensor_a[tensor_a<0.5] = 0
    tensor_a[tensor_a>=0.5] = 1

    
    ndim = len(tensor_a.shape)
   
 
    if ndim == 3 or ndim == 2:
        intersection = torch.sum(tensor_a * tensor_b)
        union = torch.sum(tensor_a) + torch.sum(tensor_b)
    elif ndim == 4:
        intersection = torch.sum(tensor_a * tensor_b, dim=(1,2,3))  # Element-wise multiplication
        union = torch.sum(tensor_a, dim=(1,2,3)) + torch.sum(tensor_b,dim=(1,2,3))
    elif ndim == 5:
        intersection = torch.sum(tensor_a * tensor_b, dim=(1,2,3,4))  # Element-wise multiplication
        union = torch.sum(tensor_a, dim=(1,2,3,4)) + torch.sum(tensor_b,dim=(1,2,3,4))
    else:
        assert 4>9, "ndim is not 4 or 5 or 3 or 2"
    
    # print(intersection)
    # print(union)
    # if union == 0:
    #     return 1.0  # Handle the case where both tensors are empty.
    union[union==0] = 0.0000000001
    res = (2.0 * intersection) / union

    if return_mean:
        return torch.mean(res).item()
    res = [item.item() for item in res]
    return res

def CDX(u):              #DX   Y-dir grad
    if len(u.shape)==2:
        res = u[:, 1:] - u[:, :-1]
        res = np.concatenate((res[:, 0:1], res), axis=-1)
    elif len(u.shape)==3:
        pass
    return res

def CDY(u):              #DY   X-dir grad
    if len(u.shape)==2:
        res = u[1:, :] - u[:-1, :]
        res = np.concatenate((res[0:1, :], res), axis=0)
    elif len(u.shape)==3:
        pass
    return res

def Cjacobian_determinant4(phiinv):
    # check inputs
    volshape = phiinv.shape[:-1]
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    
    Ju_x, Ju_y = CDX(phiinv[...,0]),  CDY(phiinv[...,0])
    Jv_x, Jv_y = CDX(phiinv[...,1]),  CDY(phiinv[...,1])
    
    det = Ju_x * Jv_y - Jv_x * Ju_y
    negative_count = np.sum(det < 0)
    total_count = det.size
    return negative_count, total_count, negative_count / total_count
    return Ju_x * Jv_y - Jv_x * Ju_y

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
