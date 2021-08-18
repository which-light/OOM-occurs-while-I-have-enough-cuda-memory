from random import shuffle, sample

class ImageFolder(object):
    # file_type: cleaned_label.txt noised_label.txt path_clean_noise.txt
    def __init__(self, file_root, file_type, init_shuffle=True, select_ratio=1):
        self.file_root = file_root + file_type
        self.file_type = file_type
        self.files_path = []
        self.select_ratio = select_ratio
        self.get_file_path()
        if init_shuffle:
            self.shuffle()

    def get_file_path(self):
        with open(self.file_root, "r") as f:  # 设置文件对象
            path_label = f.readlines()
        f.close()
        if self.file_type != "path_clean_noise.txt":
            for element in path_label:
                splited = element.split()
                self.files_path.append([splited[0], int(splited[1])])
        else:
            for element in path_label:
                splited = element.split()
                self.files_path.append([splited[0], int(splited[1]), int(splited[2])])
        self.files_path = sample(self.files_path, int(len(self.files_path)*self.select_ratio))



    def shuffle(self):
        shuffle(self.files_path)

    def change_label(self, pre, false_index, index):
        for i in false_index:
            self.files_path[index[i]][1] = pre[i]
