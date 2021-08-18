import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import os

from transform__ import get_update_num
from loader.clothing_loader import ImageFolder
from loader.cloth_data_set import ClothDataset
from net_models.Net_resnet import net_resnet
from net_models.Net_densenet import net_densenet
from TruncatedLoss import TruncatedLoss
from parser import opt


# file_type: cleaned_label.txt noised_label.txt path_clean_noise.txt
# file_root = r'/data/wql/clothing1m/'
file_root = opt.base_path
file_type1 = 'cleaned_label.txt'
file_type2 = 'noised_label.txt'
file_type3 = 'path_clean_noise.txt'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

transform=transforms.Compose([transforms.ToTensor()])
norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
nbCls = 14
transformTrain = transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])
# transformation of the test set
transformTest = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

folder_train= ImageFolder(file_root=file_root, file_type=file_type2, select_ratio=0.01)
folder_test= ImageFolder(file_root=file_root, file_type=file_type1)

data_train = ClothDataset(loader=folder_train, transform=transformTrain)
data_test = ClothDataset(loader=folder_test, transform=transformTest)

loader_train = DataLoader(data_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
loader_test = DataLoader(data_test, batch_size=opt.batch_size, shuffle=True, drop_last=True)

if __name__=="__main__":


    criterion = torch.nn.CrossEntropyLoss()
    criterion_T = TruncatedLoss(trainset_size=data_train.__len__())
    criterion = criterion.cuda(opt.cuda_device)
    criterion_T = criterion_T.cuda(opt.cuda_device)

    resnet = net_resnet(lr=opt.lr, net_num=3)
    #resnet.net = torch.nn.DataParallel(resnet.net)
    resnet.net = resnet.net.cuda(opt.cuda_device)

    densenet = net_densenet(lr=opt.lr, net_num=3)
    #densenet.net = torch.nn.DataParallel(densenet.net)
    densenet.net = densenet.net.cuda(opt.cuda_device)

    for i in range(opt.n_epochs):

        update_all_num1 = 0
        update_all_num2 = 0

        if i % 20 == 0 and i != 0:
            for image, target, index in loader_train:
                with torch.no_grad():
                    image, target = image.cuda(opt.cuda_device), target.cuda(opt.cuda_device)
                    # densenet.net.zero_grad()
                    outputs = densenet.net.forward(image)
                    criterion_T.update_weight(outputs, target, index)

        for image, target, index in tqdm(loader_train, total=int(data_train.__len__()/opt.batch_size)):
        #for image, target, index in loader_train:
            target = target.cuda(opt.cuda_device)
            # Target = F.one_hot(target[2].cuda(opt.cuda_device), opt.n_classes)
            image = image.cuda(opt.cuda_device)

            ########## train resnet1
            resnet.net.zero_grad()
            out1 = resnet.net.forward(image)
            ########## train resnet1
            densenet.net.zero_grad()
            out2 = densenet.net.forward(image)

            loss1 = criterion_T(out1, target, index)
            loss2 = criterion_T(out2, target, index)
            loss1.backward()
            resnet.optimizer.step()

            loss2.backward()
            densenet.optimizer.step()


        ###### print resnet1
        data_len = loader_train.batch_size * loader_train.__len__()
        print("the %d epoch" % (i + 1), end=' ')

        ###### print resnet2
        print('right ratio: ', "{:.5}".format(update_all_num1 / data_len), end=' ')
        print(" {:.5}".format(update_all_num2 / data_len), end=' ')

        acc1 = 0
        acc2 = 0
        for image, target, _ in loader_test:  # 0 fake_label 1 true_false 2 true_label 3
            with torch.no_grad():
                image = image.cuda(opt.cuda_device)
                ####### test resnet1
                test_out1 = resnet.net.forward(image)
                _, pre_target1 = torch.max(test_out1.data, 1)
                acc1 = acc1 + np.where(torch.eq(target[2], pre_target1.cpu()).numpy() == 1)[0].size

                ####### test resnet2
                test_out2 = densenet.net.forward(image)
                _, pre_target2 = torch.max(test_out2.data, 1)
                acc2 = acc2 + np.where(torch.eq(target[2], pre_target2.cpu()).numpy() == 1)[0].size

        ####### reduce learning rate
        accuracy1 = acc1 / (loader_test.__len__() * loader_test.batch_size)
        print("---", end='')
        print("accuracy1:", "{:.5}".format(accuracy1), end=' ')
        resnet.scheduler.step(accuracy1)
        ######
        accuracy2 = acc2 / (loader_test.__len__() * loader_test.batch_size)
        print("accuracy2:", "{:.5}".format(accuracy2), end=' ')
        densenet.scheduler.step(accuracy2)

        print("learning rate: ", "{:.1}".format(resnet.optimizer.param_groups[0]['lr']), end=' ')
        print("{:.1}".format(densenet.optimizer.param_groups[0]['lr']))

