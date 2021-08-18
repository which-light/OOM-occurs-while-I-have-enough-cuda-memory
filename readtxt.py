
with open('clean_label_kv.txt',"r") as f:    #设置文件对象
    clean_labels = f.readlines()

clean_dic = {}
for i in range(len(clean_labels)):
    clean_ = clean_labels[i].split()
    clean_dic [clean_[0]] = int(clean_[1])

with open('noisy_label_kv.txt',"r") as f:    #设置文件对象
    noisy_labels = f.readlines()
f.close()

noisy_dic = {}
for i in range(len(noisy_labels)):
    noisy_ = noisy_labels[i].split()
    noisy_dic [noisy_[0]] = int(noisy_[1])

del clean_labels,noisy_labels

print(len(clean_dic))
print(len(noisy_dic))
print()
path_clean_noise = []
for element in clean_dic.keys()&noisy_dic.keys():
    path_clean_noise.append([element, clean_dic[element], noisy_dic[element]])
    del clean_dic[element], noisy_dic[element]

print(len(clean_dic))
print(len(noisy_dic))
print()
print(len(path_clean_noise))
print()

clean_labels = list(zip(clean_dic.keys(),clean_dic.values()))
noisy_labels = list(zip(noisy_dic.keys(),noisy_dic.values()))

with open('cleaned_label.txt',"w") as clean_test:    #设置文件对象
    for key, values in clean_labels:
        clean_test.write(key + " " + str(values) + "\r")
    clean_test.close()

with open('noised_label.txt',"w") as noised_test:    #设置文件对象
    for key, values in noisy_labels:
        noised_test.write(key + " " + str(values) + "\r")
    noised_test.close()

with open('path_clean_noise.txt',"w") as path_clean_noise_:    #设置文件对象
    for key, clean, noise in path_clean_noise:
        path_clean_noise_.write(key + " " + str(clean) + " " + str(noise) + "\r")
    path_clean_noise_.close()

print(type(clean_labels))
print(type(noisy_labels))













