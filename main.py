import os
import warnings
import torch
from args import parameter_parser
import argparse
import random
import numpy as np
import torch
from train import train

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    train(args, device)
