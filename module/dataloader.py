import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from metadata.dataset import \
    ClassificationDataset

from util import imutils
from util.imutils import Normalize
from module.HaS import HaS

IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]

def get_dataloader(args):

    train_dataset = ClassificationDataset(
        args.dataset,
        args.train_list,
        img_root=args.data_root,
        transform=transforms.Compose([
            imutils.RandomResizeLong
            (args.resize_size[0], args.resize_size[1]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            HaS(32, 0.4, IMAGE_MEAN_VALUE),
            transforms.ToPILImage(),
            transforms.ColorJitter
            (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            np.asarray,
            Normalize(),
            imutils.RandomCrop(args.crop_size), 
            imutils.HWC_to_CHW,
            torch.from_numpy
        ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)

    return train_loader
