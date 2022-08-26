import torch.optim as optim
import torch.nn as nn
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import torchvision
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import os
from models.register_3d import Register3d
from models.transformer import Transformer
from utils.earlystopping import EarlyStopping
from utils.data_MONAI import train_val_to_df, test_to_df, set_transforms
from metrics.metric import NCC, NMI, MIND_loss
from monai.losses import BendingEnergyLoss
from tqdm import tqdm
from monai.data import DataLoader, CacheDataset, Dataset, IterableDataset
from dist_train import parse
import nibabel as nib
import numpy as np
import argparse
import gc
import wandb
import transforms3d

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3" 

wandb.login()
wandb.init(project='hutom-registration', entity='yon-ninii')

args = parse()

wandb.config = {
    "epochs": args.epochs,
    "batch_size": args.batch_size
}

gc.collect()
torch.cuda.empty_cache()
# affine_transform = opt.linear
alpha = beta = 1e-6
num_epochs = args.epochs # opt.epochs
batch_size = args.batch_size  
learning_rate = 1e-3
data_path = '/mnt/pre_data_2'

device = ("cuda" if torch.cuda.is_available() else "cpu")

test_file = test_to_df(data_path)

test_trans, origin_trans = set_transforms(mode='test')

test_dataset = Dataset(test_file, transform = test_trans)
origin_dataset = Dataset(test_file, transform = origin_trans)

testloader = DataLoader(test_dataset, batch_size = batch_size, 
num_workers = 4, pin_memory = True)
originloader = DataLoader(origin_dataset, batch_size = batch_size, 
num_workers = 4, pin_memory = True)

WEIGHTS = torch.load('/home/ym/0819reg/yong/image-registration-cnn/checkpoints/checkpoint_29.pth', map_location = 'cpu')

model = Register3d(B=args.batch_size)
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
model.to(device)
model = nn.DataParallel(model, device_ids = [0,1,2,3])
model.load_state_dict(WEIGHTS)
trans = Transformer(linear = True)

loss_func = nn.MSELoss() #+ BendingEnergyLoss()
#loss_func = MIND_loss()# + BendingEnergyLoss()
#loss_func = NCC() #+ BendingEnergyLoss()
#loss_func = NMI() + BendingEnergyLoss()

print('test start!')

test_bar = tqdm(testloader)
total_test_loss = 0
with torch.no_grad():
    for i, batch in enumerate(test_bar):
        origin_batch = next(iter(originloader))
        moving_image, fixed_image = batch['moving_image'], batch['fixed_image']
        #moving_img = test_file[0]['moving_image']
        moving = nib.load('/mnt/pre_data_2/pre_artery/pre_artery1.nii.gz')
        fixed = nib.load('/mnt/pre_data_2/pre_vein/pre_vein1.nii.gz')
        move_affine = moving.affine
        fixed_affine = fixed.affine
        movout = moving.get_fdata()
        fixout = moving.get_fdata()
        moving_image, fixed_image = moving_image.cuda(), fixed_image.cuda()
        moving_origin, fixed_origin = origin_batch['moving_image'], origin_batch['fixed_image']
        out, test_theta = model(moving_image, fixed_image) 
        theta = test_theta.cpu()#.squeeze()
        out_origin = trans(moving_origin, test_theta.cpu())
        theta = np.append(theta.numpy(), np.array([[0, 0, 0, 1]]), axis=0)
        warped_log = out_origin[0, 0, :, 128, :].numpy()
        outout = out_origin.squeeze().numpy()
        mov_check = moving_origin.squeeze().numpy()
        print(out_origin.shape, outout.shape, movout.shape)
        #T, R, Z, S = transforms3d.affines.decompose44(theta)
        #Z = np.ones(3)
        #S = np.zeros(3)
        #print(T, R, Z, S)
        #A = transforms3d.affines.compose(T, R, Z, S)
        #print(T, R, Z, S)
        mov_mov = nib.Nifti1Image(movout, affine = np.eye(4))
        fix_mov = nib.Nifti1Image(fixout, affine = np.eye(4))
        out_mov = nib.Nifti1Image(outout, affine = np.eye(4))
        check = nib.Nifti1Image(mov_check, affine = np.eye(4))
        out_out = nib.Nifti1Image(movout, affine = test_theta.squeeze().numpy())
        print(out_out.header)
        nib.save(mov_mov, '/home/ym/0819reg/yong/image-registration-cnn/checkpoints2/fix_out.nii.gz')
        nib.save(fix_mov, '/home/ym/0819reg/yong/image-registration-cnn/checkpoints2/mov_mov.nii.gz')
        nib.save(out_mov, '/home/ym/0819reg/yong/image-registration-cnn/checkpoints2/out_out.nii.gz')
        nib.save(out_out, '/home/ym/0819reg/yong/image-registration-cnn/checkpoints2/mov_out.nii.gz')
        nib.save(check, '/home/ym/0819reg/yong/image-registration-cnn/checkpoints2/check.nii.gz')

        test_loss = loss_func(out, fixed_image)
        test_loss += alpha*torch.sum(torch.abs(test_theta - torch.eye(3, 4).cuda()))
        total_test_loss += test_loss.item()
#total_test_loss /= len(testloader)
wandb.log({'Test Loss':total_test_loss})
test_bar.set_description(desc=f"test loss: {total_test_loss:.4f}")
print('finished testing')
wandb.finish()


'''
        output = out_origin.view(1, 1, 256, 256, 256)
        wandb.log({"Moving": [wandb.Image(moving_origin[0, 0, :, :, 128].numpy(), caption="Moving")]})
        wandb.log({"Fixed": [wandb.Image(fixed_origin[0, 0, :, :, 128].numpy(), caption="Fixed")]})
        wandb.log({"Predicted": [wandb.Image(warped_log, caption="Predicted")]})
        wandb.log({"Predicted_warp": [wandb.Image(output.numpy(), caption="Predicted_warp")]})
'''