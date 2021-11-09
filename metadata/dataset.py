import random
import os.path
import PIL.Image
import numpy as np

import torch
from torch.utils.data import Dataset


def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels_10.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


class ImageDataset(Dataset):
    """
    Base image dataset. This returns 'img_id' and 'image'
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img_id, img


class ClassificationDataset(ImageDataset):
    """
    Classification Dataset (base)
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        super().__init__(dataset, img_id_list_file, img_root, transform)
        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        label = torch.from_numpy(self.label_list[idx])
        return name, img, label
