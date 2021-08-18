import torchvision.models as models
import torch
from parser import opt

class resnet_modified(torch.nn.Module):
    def __init__(self, n_classes=10, pretrained=True, net_num=0):
        super(resnet_modified, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained

        if net_num == 0:
            self.densenet = models.densenet121(pretrained=pretrained)
        elif net_num == 1:
            self.densenet = models.densenet161(pretrained=pretrained)
        elif net_num == 2:
            self.densenet = models.densenet169(pretrained=pretrained)
        elif net_num == 3:
            self.densenet = models.densenet201(pretrained=pretrained)

        self.densenet.classifier = torch.nn.Linear(1024, n_classes)
        self.SoftMax = torch.nn.Softmax(dim=1)

    def forward(self, x=False):
        x = self.densenet.forward(x)
        x = self.SoftMax(x)
        return x

class net_densenet(object):
    def __init__(self, n_classes=opt.n_classes, pretrained=True, lr=opt.lr, net_num=0):
        super(net_densenet, self).__init__()
        self.net = resnet_modified(n_classes=n_classes, pretrained=pretrained, net_num=0)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999),
                                                eps=1e-07, weight_decay=1e-04,
                                                amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-06, factor=0.5)

