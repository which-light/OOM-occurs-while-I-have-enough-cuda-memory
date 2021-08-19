from torch.utils.data import Dataset
from torch import rand
from random import randint
from PIL import Image
from parser import opt



class ClothDataset(Dataset):
    def __init__(self, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform

    def __getitem__(self, index):
        img = rand(3, opt.img_size, opt.img_size)
        return img, randint(0, 13), index

    def __len__(self):
        return 10000


