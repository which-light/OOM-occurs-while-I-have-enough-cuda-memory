import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            torch.nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            #nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            torch.nn.BatchNorm2d(in_features, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            #nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            torch.nn.BatchNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class DetectorResNet(torch.nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=1):
        super(DetectorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            torch.nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            #nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            torch.nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                torch.nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                #nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                torch.nn.BatchNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                torch.nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                torch.nn.MaxPool2d(3, stride=2, padding=1),
                #nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                torch.nn.BatchNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2
#####
        # Output layer
        #model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]
#####

        self.G_A_P = torch.nn.AdaptiveAvgPool2d((1, 1))
        Liner = [torch.nn.BatchNorm1d(64), torch.nn.Tanh()]
        #Liner += [torch.nn.Linear(64, 32), torch.nn.BatchNorm1d(32), torch.nn.Tanh()]
        #Liner += [torch.nn.Linear(32, 16), torch.nn.BatchNorm1d(16), torch.nn.Tanh()]
        Liner += [torch.nn.Linear(64, 2), torch.nn.Softmax(dim=1)]

        self.model = torch.nn.Sequential(*model)
        self.Liner = torch.nn.Sequential(*Liner)


    def forward(self, x, c):
        c = c.float()
        c = c.view(-1, 1, 1, 1)
        c = c.repeat(1, 1, x.shape[2], x.shape[3])
        x = torch.cat((x, c), 1)
        x = self.model(x)
        x = self.G_A_P(x).squeeze(-1).squeeze(-1)
        return self.Liner(x)

class net_detector(object):
    def __init__(self,lr=0.001):
        super(net_detector, self).__init__()
        self.net = DetectorResNet(img_shape=(3, 128, 128), res_blocks=9, c_dim=1)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999),
                                     eps=1e-07, weight_decay=0.001,
                                     amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-06, factor=0.5)
        self.scheduler_epoch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.1)



































