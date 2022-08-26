import os
import torch
import torch.distributed as dist
import wandb
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size per process (default: 8)')
    #parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
    #                    metavar='LR', help='Initial learning rate')
   
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    args = parser.parse_args()
    return args