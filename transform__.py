from random import random, choice
from torchvision import transforms as T
from torch import tensor, zeros, ones
import numpy as np
from PIL import Image

import pandas as pd

class add_noise(object):
    def __init__(self, noise_ratio, class_num):
        self.noise_ratio = noise_ratio
        self.class_num = class_num

    def __call__(self, target):
        #if True:
        if random() < self.noise_ratio:
            others = [i for i in range(self.class_num)]
            del others[target]
            target = [choice(others), False, target]
        else:
            target = [target, True, target]
        return target


def Fake_labels(true_label, class_num):
    fake_label = []
    for label in true_label:
        others = [i for i in range(class_num)]
        del others[label]
        fake_label.append(choice(others))
    return tensor(fake_label)

def updata_index(ratio, pre):
    zeros_index = np.where(pre == 0)[0]
    ones_index = np.where(pre == 1)[0]

    np.random.shuffle(zeros_index)
    np.random.shuffle(ones_index)

    index_true = ones_index[0: int(ratio * len(ones_index))]
    index_false = zeros_index[0: int(ratio * len(zeros_index))]

    return index_true, index_false

def choose_detector(index_true, index_false, ind, update_num, threshold):
    detector_ture = set(index_true) & set(ind[:update_num])
    detector_false = set(index_false) & set(ind[len(ind) - update_num:])
    if (len(detector_ture) + len(detector_false)) / (2 * update_num) >= threshold:
        return True
    else:
        return False

def statistic_loss(loss, iter_num, epoch_loss):
    epoch_loss = (epoch_loss * (iter_num - 1) + loss) / iter_num


def save_to_xlsx(all_loss):
    data_df = pd.DataFrame(all_loss)
    writer = pd.ExcelWriter('all_loss.xlsx', mode="w")
    data_df.to_excel(writer, 'page_1', float_format='%.5f', index=False, header=False)
    writer.save()
    writer.close()

def random_ind(ind, ratio):
    return np.random.choice(ind, int(len(ind) * ratio), replace=False)

def print_test_acc(model, true_num, data_len, end='\n'):
    acc = true_num / data_len
    print("the accuracy of " + model + ": ", "{:.5}".format(acc), end=end)
    return acc

def print_true_false(model, true_num, data_len, end='\n'):
    acc_as_true = 1 - true_num[0]/data_len
    acc_as_false = 1 - true_num[1] / data_len
    acc_as_true_half = 1 - true_num[2] / data_len
    acc_as_false_half = 1 - true_num[3] / data_len
    print("the true and false by " + model + ": " +
          "{:.5}".format(acc_as_true) + " " + "{:.5}".format(acc_as_false), end=" ")

    print("the half true and false by " + model + ": " +
          "{:.5}".format(acc_as_true_half) + " " + "{:.5}".format(acc_as_false_half), end=end)
    return [acc_as_true, acc_as_false, acc_as_true_half, acc_as_false_half]


def get_update_num(L):
    return np.argmax(np.diff(L)) + 1

















































