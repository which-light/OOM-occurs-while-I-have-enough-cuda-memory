import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--n_classes", type=int, default=14, help="the number of classes")
parser.add_argument("--cuda_device", type=int, default=0, help="which gpu is used to train the model")
parser.add_argument("--base_path", type=str, default="/data/guest1/clothing1m/", help="which gpu is used to train the model")

opt = parser.parse_args()
print(opt)