import os
import torch
import argparse
from torch.backends import cudnn

from util import pyutils

from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer
from module.train import train_cls

from settings import NUM_CLASSES, HOME_ROOT

cudnn.enabled = True
torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    # session
    parser.add_argument("--session", default="BASELINE", type=str)
    parser.add_argument('--start_iteration', default=0, type=int)

    # data
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--dataset", default='voc12', type=str)
    parser.add_argument("--saliency_root", type=str)
    parser.add_argument("--train_list", default=f"{HOME_ROOT}/metadata/voc12/train_{NUM_CLASSES}.txt", type=str)
    parser.add_argument("--save_root", default='/log')

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--resize_size", default=(256, 512), type=int, nargs='*')

    # network
    parser.add_argument("--network", default="network.resnet50_cls", type=str)

    # optimizer
    parser.add_argument("--lr", default=0.0099, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--max_iters", default=10000, type=int)

    args = parser.parse_args()

    args.num_classes = int(NUM_CLASSES)

    return args


if __name__ == '__main__':

    # get arguments
    args = get_arguments()

    # set log
    args.log_folder = os.path.join(args.save_root, args.session)
    os.makedirs(args.log_folder, exist_ok=True)

    pyutils.Logger(os.path.join(args.log_folder, 'log_cls.log'))
    print(vars(args))

    # load dataset
    train_loader = get_dataloader(args)

    max_step = args.max_iters

    # load network and its pre-trained model
    model = get_model(args)

    model = model.cuda()

    # set optimizer
    optimizer = get_optimizer(args, model, max_step)

    # load previous checkpoint
    checkpoint_path = os.path.join(args.log_folder, 'checkpoint_cls.pth')
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        args.start_iteration = checkpoint['iteration']
        args.lr = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer.to(torch.device('cuda'))
    
    # train
    model.train()

    train_cls(train_loader, model, optimizer, max_step, args)
