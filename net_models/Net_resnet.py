import torchvision.models as models
import torch
from parser import opt

class resnet_modified(torch.nn.Module):
    def __init__(self, n_classes=10, pretrained=True, net_num=0):
        super(resnet_modified, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        #self.resnet = models.Inception3
        #self.resnet = models.resnet101(pretrained=pretrained)
        if net_num == 0:
            self.resnet = models.resnet18(pretrained=pretrained)
            self.resnet.fc = torch.nn.Linear(in_features=512, out_features=n_classes)
        elif net_num == 1:
            self.resnet = models.resnet34(pretrained=pretrained)
            self.resnet.fc = torch.nn.Linear(in_features=512, out_features=n_classes)
        elif net_num == 2:
            self.resnet = models.resnet50(pretrained=pretrained)
            self.resnet.fc = torch.nn.Linear(in_features=2048, out_features=n_classes)
        elif net_num == 3:
            self.resnet = models.resnet101(pretrained=pretrained)
            self.resnet.fc = torch.nn.Linear(in_features=2048, out_features=n_classes)
        elif net_num == 4:
            self.resnet = models.resnet152(pretrained=pretrained)

        self.SoftMax = torch.nn.Softmax(dim=1)


    def forward(self, x=False):
        x = self.resnet.forward(x)
        x = self.SoftMax(x)
        return x

class net_resnet(object):
    def __init__(self, n_classes=opt.n_classes, pretrained=True, lr=opt.lr, net_num=0):
        super(net_resnet, self).__init__()
        self.net = resnet_modified(n_classes=n_classes, pretrained=pretrained, net_num=net_num)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999),
                                                eps=1e-07, weight_decay=1e-04,
                                                amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-06, factor=0.5)





























