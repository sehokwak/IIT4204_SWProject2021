import torch
import torch.nn as nn
import torch.nn.functional as F

from network.resnet50d import resnet50


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


class Net(nn.Module):

    def __init__(self, num_classes=20,
                 dilations=(1, 1, 1, 1), strides=(2, 2, 2, 1)):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.resnet50 = resnet50(pretrained=True,
                                 dilations=dilations, strides=strides)

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, self.num_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])


    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        cam = self.classifier(x)
        pred = gap2d(cam, keepdims=False)

        return pred, cam

    def forward_cam(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        cam = self.classifier(x)
        cam = F.relu(cam)

        return cam

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
