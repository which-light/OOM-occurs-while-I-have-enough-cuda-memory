from torch.utils.data import Dataset
from PIL import Image
from parser import opt


class ClothDataset(Dataset):
    def __init__(self,loader, transform=None):
        super(Dataset, self).__init__()
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(opt.base_path+self.loader.files_path[index][0])
        if self.transform is not None:
            if img.layer != 3:
                img = img.convert('RGB')
            img = self.transform(img)
        if self.loader.file_type == 'path_clean_noise.txt':
            return img, self.loader.files_path[index][1], self.loader.files_path[index][2], index
        else:
            return img, self.loader.files_path[index][1], index

    def __len__(self):
        return len(self.loader.files_path)

    def change_label(self, pre, false_index, index):
        self.loader.change_label(pre, false_index, index)


