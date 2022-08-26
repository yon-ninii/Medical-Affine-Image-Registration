import torch.optim as optim
import torch.nn as nn
import torch
import torch.distributed as dist
import torchvision
import numpy as np
import os
from models.register_3d import Register3d
from utils.earlystopping import EarlyStopping
from utils.data_MONAI import train_val_to_df, set_transforms
from utils.ddp import parse
from metrics.metric import NCC, NMI, MIND_loss, NormalizedCrossCorrelation, MutualInformation, NMI_torch, MILoss
from monai.losses import BendingEnergyLoss, LocalNormalizedCrossCorrelationLoss, GlobalMutualInformationLoss
from tqdm import tqdm
from monai.data import DataLoader, Dataset
from monai.transforms import Affine
import argparse
import gc
import wandb
from math import radians, pi

# Set GPU, CUDA
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3, 4, 5, 6, 7" 

# Set WandB
wandb.login()
wandb.init(project='hutom-registration', entity='yon-ninii')

args = parse()

wandb.config = {
    "learning_rate": 0.00001,
    "epochs": args.epochs,
    "batch_size": args.batch_size
}

# Set Device 
device = ("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()

# Train Config
alpha = beta = 1e-6
num_epochs = args.epochs
batch_size = args.batch_size  
val_batch_size = 2
learning_rate = 1e-5
data_path = '/nas3/pre_data_2'

# Set Data Loader 
train_file, val_file = train_val_to_df(data_path)

train_trans, val_trans = set_transforms(mode='train')

train_dataset = Dataset(train_file, transform = train_trans)
val_dataset = Dataset(val_file, transform = val_trans)

trainloader = DataLoader(train_dataset, batch_size = batch_size, 
num_workers = 32, pin_memory = True, shuffle=True)
valloader = DataLoader(val_dataset, batch_size = val_batch_size, 
num_workers = 32, pin_memory = True, shuffle=True)

# Set Model
model = Register3d()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
model.to(device)
model = nn.DataParallel(model, device_ids = [0,1,2,3,4,5,6,7])


# Set Loss Function
#loss_func = nn.MSELoss() #+ BendingEnergyLoss()
#loss_func = MIND_loss()# + BendingEnergyLoss()
#loss_func = NormalizedCrossCorrelation() #+ BendingEnergyLoss()
#loss_func = MILoss()# + BendingEnergyLoss()
loss_func = GlobalMutualInformationLoss()
#loss_reg = BendingEnergyLoss()

# Set Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                 patience=50, verbose=True)
early_stop = EarlyStopping(patience=100, verbose=True)

print('train start!')

for epoch in range(num_epochs):
    train_bar = tqdm(trainloader)
    for i, batch in enumerate(train_bar):
        # Load Image
        moving_image, fixed_image = batch['moving_image'], batch['fixed_image']
        moving_image, fixed_image = moving_image.to(device), fixed_image.to(device)
        B = moving_image.size()[0]

        optimizer.zero_grad()
        model.train() 
        x = model(moving_image, fixed_image)

        # Warp image by batches (Need to convert rotation params to radians)
        l, l_theta = [], []
        for j in range(B):
            t1, t2, t3 = x[j, 0].clone().detach(), x[j, 1].clone().detach(), x[j, 2].clone().detach()
            r1, r2, r3 = radians(x[j, 3].clone().detach() * pi), radians(x[j, 4].clone().detach() * pi), radians(x[j, 5].clone().detach() * pi)
            affines0 = Affine(rotate_params=(r1, r2, r3), translate_params=(t1, t2, t3), mode='bilinear', device=torch.device(device))
        #    affines0 = Affine(affine=x[j, :, :], mode='bilinear', device=torch.device(device))
            o0, t0 = affines0(moving_image[j,:,:,:,:])
            l.append(o0)
            l_theta.append(t0)
        output = torch.stack(l, dim=0).cuda()
        theta = torch.stack(l_theta, dim=0)

        # Compute Loss
        loss = loss_func(output, fixed_image) #+ loss_reg(output)
        loss += alpha * torch.sum(torch.abs(theta - torch.eye(4, 4).cuda()))

        wandb.log({'Train Loss':loss.item()})
        loss.requires_grad_(True)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()

        optimizer.step()
        train_bar.set_description(desc=f"[{epoch + 1}/{num_epochs}loss: {loss:.7f} lr: {learning_rate:.4f}]")
    
    model.eval()

    val_bar = tqdm(valloader)
    total_val_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(val_bar):
            # Load Image
            moving_image, fixed_image = batch['moving_image'], batch['fixed_image']
            moving_image, fixed_image = moving_image.to(device), fixed_image.to(device)
            B = moving_image.size()[0]
            x = model(moving_image, fixed_image) 

            # Warp image by batches (Need to convert rotation params to radians)
            l, l_theta = [], []
            for j in range(B):    
                t1, t2, t3 = x[j, 0].clone().detach(), x[j, 1].clone().detach(), x[j, 2].clone().detach()
                r1, r2, r3 = radians(x[j, 3].clone().detach() * pi), radians(x[j, 4].clone().detach() * pi), radians(x[j, 5].clone().detach() * pi)
                affines0 = Affine(rotate_params=(r1, r2, r3), translate_params=(t1, t2,t3), mode='bilinear', device=torch.device(device))#, image_only=True)
            #    affines0 = Affine(affine=x[j, :, :], mode='bilinear', device=torch.device(device))
                o0, t0 = affines0(moving_image[j,:,:,:,:])
                l.append(o0)
                l_theta.append(t0)
            output = torch.stack(l, dim=0)
            val_theta = torch.stack(l_theta, dim=0)

            # Compute Val Loss
            val_loss = loss_func(output, fixed_image) #+ loss_reg(output)
            val_loss += alpha*torch.sum(torch.abs(val_theta - torch.eye(4, 4).cuda()))

            # Log Results to WandB
            warped_log = np.clip(output[0, 0, :, 32, :].detach().cpu().numpy(), 0.0, 1.0)
            wandb.log({"Moving": [wandb.Image(np.clip(moving_image[0, 0, :, 32, :].cpu().numpy(), 0.0, 1.0), caption="Moving")],
                    "Fixed": [wandb.Image(np.clip(fixed_image[0, 0, :, 32, :].cpu().numpy(), 0.0, 1.0), caption="Fixed")],
                    "Predicted": [wandb.Image(warped_log, caption="Predicted")]})

            total_val_loss += val_loss.item()
    total_val_loss /= len(valloader)
    wandb.log({'Val Loss':total_val_loss})
    val_bar.set_description(desc=f"val loss: {total_val_loss:.4f}")

    scheduler.step(total_val_loss)
    early_stop(model, total_val_loss, epoch)
    if early_stop.early_stop:
        print("Early Stopping")
        break

print('finished training')
wandb.finish()