import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchreg.settings.settings import get_ndims
from torchvision import models

NDIMS = 1

def get_ndims():
    """
    Gets the number of data dimensions. 
    """
    global NDIMS
    if NDIMS is None:
        raise Exception(
            """NDIMS not set. Specify the dimensionality with 'torchreg.settings.set_ndims(ndims)'"""
        )
    return NDIMS

    import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np

def thirdOrderSplineKernel(u):
    abs_u = u.abs()
    sqr_u = abs_u.pow(2.0)

    result = torch.FloatTensor(u.size()).zero_()
    result = result.to(u.device)

    mask1 = abs_u<1.0
    mask2 = (abs_u>=1.0 )&(abs_u<2.0)

    result[mask1] = (4.0 - 6.0 * sqr_u[mask1] + 3.0 * sqr_u[mask1] * abs_u[mask1]) / 6.0
    result[mask2] = (8.0 - 12.0 * abs_u[mask2] + 6.0 * sqr_u[mask2] - sqr_u[mask2] * abs_u[mask2]) / 6.0

    return result

class MILoss(nn.Module):
    def __init__(self, num_bins=16):
        super(MILoss, self).__init__()
        self.num_bins = num_bins


    def forward(self, moving, fixed):
        moving = moving.view(moving.size(0), -1)
        fixed = fixed.view(fixed.size(0), -1)

        padding = float(2)
        batchsize = moving.size(0)

        fixedMin = fixed.min(1)[0].view(batchsize,-1)
        fixedMax = fixed.max(1)[0].view(batchsize,-1)

        movingMin = moving.min(1)[0].view(batchsize,-1)
        movingMax = moving.max(1)[0].view(batchsize,-1)
        #print(fixedMax,movingMax)

        JointPDF = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()
        movingPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        fixedPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        JointPDFSum = torch.FloatTensor(batchsize).zero_()
        JointPDF_norm = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()

        if JointPDF.device != moving.device:
            JointPDF = JointPDF.to(moving.device)
            movingPDF = movingPDF.to(moving.device)
            fixedPDF = fixedPDF.to(moving.device)
            JointPDFSum = JointPDFSum.to(moving.device)
            JointPDF_norm = JointPDF_norm.to(moving.device)

        #print(JointPDF.device)

        fixedBinSize = (fixedMax - fixedMin) / float((self.num_bins - 2 * padding))
        movingBinSize = (movingMax - movingMin) / float(self.num_bins - 2 * padding)

        fixedNormalizeMin = fixedMin / fixedBinSize - float(padding)
        movingNormalizeMin = movingMin / movingBinSize - float(padding)

        #print(fixed.shape,fixedBinSize.shape,fixedNormalizeMin.shape)
        fixed_winTerm = fixed / fixedBinSize - fixedNormalizeMin

        fixed_winIndex = fixed_winTerm.int()
        fixed_winIndex[fixed_winIndex < 2] = 2
        fixed_winIndex[fixed_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        moving_winTerm = moving / movingBinSize - movingNormalizeMin

        moving_winIndex = moving_winTerm.int()
        moving_winIndex[moving_winIndex < 2] = 2
        moving_winIndex[moving_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        for b in range(batchsize):
            a_1_index = moving_winIndex[b] - 1
            a_2_index = moving_winIndex[b]
            a_3_index = moving_winIndex[b] + 1
            a_4_index = moving_winIndex[b] + 2

            a_1 = thirdOrderSplineKernel((a_1_index - moving_winTerm[b]))
            a_2 = thirdOrderSplineKernel((a_2_index - moving_winTerm[b]))
            a_3 = thirdOrderSplineKernel((a_3_index - moving_winTerm[b]))
            a_4 = thirdOrderSplineKernel((a_4_index - moving_winTerm[b]))
            for i in range(self.num_bins):
                fixed_mask = (fixed_winIndex[b] == i)
                fixedPDF[b][i] = fixed_mask.sum()
                for j in range(self.num_bins):
                    JointPDF[b][i][j] = a_1[fixed_mask & (a_1_index == j)].sum() + a_2[
                        fixed_mask & (a_2_index == j)].sum() + a_3[fixed_mask & (a_3_index == j)].sum() + a_4[
                                            fixed_mask & (a_4_index == j)].sum()

            #JointPDFSum[b] = JointPDF[b].sum()
            #norm_facor = 1.0 / JointPDFSum[b]
            #print(JointPDF[b])
            JointPDF_norm[b] = JointPDF[b] / JointPDF[b].sum()
            fixedPDF[b] = fixedPDF[b] / fixed.size(1)

        movingPDF = JointPDF_norm.sum(1)
        
        #print(JointPDF_norm)

        MI_loss = torch.FloatTensor(batchsize).zero_().to(moving.device)
        for b in range(batchsize):
            JointPDF_mask = JointPDF_norm[b] > 0
            movingPDF_mask = movingPDF[b] > 0
            fixedPDF_mask = fixedPDF[b] > 0

            MI_loss[b] = (JointPDF_norm[b][JointPDF_mask] * JointPDF_norm[b][JointPDF_mask].log()).sum() \
                         - (movingPDF[b][movingPDF_mask] * movingPDF[b][movingPDF_mask].log()).sum() \
                         - (fixedPDF[b][fixedPDF_mask] * fixedPDF[b][fixedPDF_mask].log()).sum()

        #print(MI_loss)
        loss = MI_loss.mean()
        return -1.0*loss

class MIND_loss(torch.nn.Module):
    """
    Implementation retrieved from 
    https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
    """
    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class NMI(nn.Module):
    """
    Normalized mutual information, using gaussian parzen window estimates.
    Adapted from https://github.com/qiuhuaqi/midir/blob/master/model/loss.py
    """

    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(
            self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(
            start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        # (N, #bins, #bins) / (N, 1, 1)
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.
        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))
        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - \
            torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)

class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss. Normalized to window [0,2], with 0 being perfect match.
    """

    def __init__(self, window=9, squared=False, eps=1e-6, reduction='mean'):
        super().__init__()
        self.win = window
        self.squared = squared
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        def compute_local_sums(I, J):
            # calculate squared images
            I2 = I * I
            J2 = J * J
            IJ = I * J

            # take sums
            I_sum = conv_fn(I, filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

            # take means
            win_size = np.prod(filt.shape)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            # calculate cross corr components
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            return I_var, J_var, cross

        # get dimension of volume
        ndims = get_ndims()
        channels = y_true.shape[1]

        # set filter
        filt = torch.ones(channels, channels, *([self.win] * ndims)).type_as(y_true)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)
        stride = 1
        padding = self.win // 2

        # calculate cc
        var0, var1, cross = compute_local_sums(y_true, y_pred)
        if self.squared:
            cc = cross ** 2 / (var0 * var1).clamp(self.eps)
        else:
            cc = cross / (var0.clamp(self.eps) ** 0.5 * var1.clamp(self.eps) ** 0.5)

        # mean and invert for minimization
        if self.reduction == 'mean':
            return -torch.mean(cc) + 1
        else:
            return cc


def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x, y)
    dev_xx = torch.mul(x, x)
    dev_yy = torch.mul(y, y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    if (torch.isclose(torch.tensor([-1.0]).to("cuda"), ncc).any()):
        ncc = ncc + torch.tensor([0.01]).to("cuda")

    elif (torch.isclose(torch.tensor([1.0]).to("cuda"), ncc).any()):
        ncc = ncc - torch.tensor([0.01]).to("cuda")

    return ncc, ncc_map


class NormalizedCrossCorrelation(nn.Module):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 eps=1e-8,
                 return_map=False,
                 reduction='mean'):

        super(NormalizedCrossCorrelation, self).__init__()

        self._eps = eps
        self._return_map = return_map
        self._reduction = reduction

    def forward(self, x, y):
        return normalized_cross_correlation(x, y,self._return_map, self._reduction, self._eps)

class MutualInformation(nn.Module):

    def __init__(self, sigma=0.4, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()

        self.sigma = 2 * sigma ** 2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device='cuda').float(), requires_grad=True)

    def marginalPdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def getMutualInformation(self, input1, input2):
        '''
            input1: B, C, H, W, D
            input2: B, C, H, W, D
            return: scalar
        '''

        # Torch tensors for images between (0, 1)
        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W, D = input1.shape
        assert ((input1.shape == input2.shape))

        x1 = input1.view(B, H * W * D, C)
        x2 = input2.view(B, H * W * D, C)

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)
        #print(torch.mean(mutual_information))
        return torch.mean(mutual_information)

    def forward(self, input1, input2):
        '''
            input1: B, C, H, W
            input2: B, C, H, W
            return: scalar
        '''
        return self.getMutualInformation(input1, input2)
        

class NMI_torch:

    def __init__(self, bin_centers, vol_size, sigma_ratio=0.5, max_clip=1, local=False, crop_background=False, patch_size=1):
        """
        Mutual information loss for image-image pairs.
        Author: Courtney Guo
        If you use this loss function, please cite the following:
        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        #print("vxm info: mutual information loss is experimental", file=sys.stderr)
        self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.mi = self.local_mi if local else self.global_mi
        self.vol_bin_centers = torch.tensor(bin_centers)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = torch.tensor(  1 / (  2 * np.square(self.sigma) )   ).to("cuda")
        #self.o = [1, 1, np.prod([10])]
        #self.vbc = torch.reshape(self.vol_bin_centers, self.o).to("cuda")
        self.y_pred_shape1 = torch.tensor([1,2097152,1])
        # print(y_pred_shape1)
        self.nb_voxels = self.y_pred_shape1[1]

    def local_mi(self, y_true, y_pred):
        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, self.num_bins]
        vbc = torch.reshape(self.vol_bin_centers, o)

        # compute padding sizes
        patch_size = self.patch_size
        x, y, z = self.vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0,0]]
        pad_dims.append([x_r//2, x_r - x_r//2])
        pad_dims.append([y_r//2, y_r - y_r//2])
        pad_dims.append([z_r//2, z_r - z_r//2])
        pad_dims.append([0,0])
        padding = torch.tensor(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = torch.exp(- self.preterm * torch.square(torch.nn.functional.pad(y_true, padding, 'constant')  - vbc))
        I_a /= torch.sum(I_a, 0, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(torch.nn.functional.pad(y_pred, padding, 'constant')  - vbc))
        I_b /= torch.sum(I_b, 0, keepdim=True)

        I_a_patch = torch.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_a_patch = torch.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = torch.reshape(I_a_patch, [-1, patch_size**3, self.num_bins])

        I_b_patch = torch.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_b_patch = torch.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = torch.reshape(I_b_patch, [-1, patch_size**3, self.num_bins])

        # compute probabilities
        I_a_permute = torch.permute(I_a_patch, (0,2,1))
        pab = torch.bmm(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size**3
        pa = torch.mean(I_a_patch, 1, keepdims=True)
        pb = torch.mean(I_b_patch, 1, keepdims=True)

        papb = torch.bmm(torch.permute(pa, (0,2,1)), pb) + K.epsilon()
        return torch.mean(torch.sum(torch.sum(pab * torch.log(pab/papb + 1e-8), 1), 1))

    def global_mi(self, y_true, y_pred):
        if self.crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = torch.ones([ 1, 1, padding_size, padding_size, padding_size])

            smooth = torch.nn.Conv3d(y_true, filt, padding=[1, 1, 1, 1, 1])
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = torch.masked_select(y_pred, mask)
            y_true = torch.masked_select(y_true, mask)
            y_pred = torch.unsqueeze(torch.unsqueeze(y_pred, 0), 2)
            y_true = torch.unsqueeze(torch.unsqueeze(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = torch.reshape(y_true, (-1, torch.prod(torch.tensor([*(y_true.shape)])[1:])))
            y_true = torch.unsqueeze(y_true, 2)
            y_pred = torch.reshape(y_pred, (-1, torch.prod(torch.tensor([*(y_pred.shape)])[1:])))
            y_pred = torch.unsqueeze(y_pred, 2)

        
        #nb_voxels = self.y_pred_shape1[1]
        nb_voxels = torch.tensor(y_pred.shape[1])

        # reshape bin centers to be (1, 1, B)
        vol_bin_centers = torch.tensor(self.vol_bin_centers)
        o = [1, 1, 10]
        vbc = torch.reshape(vol_bin_centers, o).to("cuda")      

        # compute image terms
        I_a = torch.exp(- self.preterm * torch.square(y_true  - vbc))
        I_a = I_a/torch.sum(I_a, -1, keepdims=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred  - vbc))
        I_b = I_b/torch.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = torch.permute(I_a, (0,2,1))
        pab = torch.bmm(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab = pab/nb_voxels
        pa = torch.mean(I_a, 1, keepdims=True)
        pb = torch.mean(I_b, 1, keepdims=True)

        papb = torch.bmm(torch.permute(pa, (0,2,1)), pb) + 1e-7
        return torch.sum(torch.sum(pab * torch.log(pab/papb + 1e-7), 1), 1)

    def loss(self, y_true, y_pred):
        y_pred = torch.clip(y_pred, 0, self.max_clip)
        y_true = torch.clip(y_true, 0, self.max_clip)
        return -self.mi(y_true, y_pred)