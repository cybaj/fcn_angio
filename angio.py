from random import sample, seed
from torch.utils import data
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

# divider number of total data
_DIV = 60

# seed
seed(100)

class ANGIODataset(data.Dataset):
    def __init__(self, mode='train', transform = None, logdir = None):
        from dataset import metadata, hrhcase_metadata, get_dataset, get_testset

        targets = metadata.values()
        _hrhcase_targets = hrhcase_metadata.values()

        self.total_targets = list(targets) + list(_hrhcase_targets)
        self.data_root = os.path.join('../tensorflow-deeplab-v3/dataset/boramae/transformed_dir')

        if mode == 'train':
            self.trainset = get_dataset(self.total_targets, total_ok=False, selection=True, set_operator='and') + get_dataset(self.total_targets, total_ok=True, set_operator='or')
        elif mode == 'valid':
            self.trainset = get_testset()
    
        self.total_input_path = []
        self.total_label_path = []
        self.class_names = ["NO", "EA"]
        
        _input_dir = 'jpegs'
        _label_dir = 'transformed'
        for target in tqdm(self.trainset):
            name = target['dirname']
            target_root = os.path.join(self.data_root, name)
            input_path = os.path.join(target_root, _input_dir)
            label_path = os.path.join(target_root, _label_dir)
            file_name = []
            with open(os.path.join(target_root, f'{name}_filelist.txt')) as fp:
                file_name = fp.readlines()
                if mode == 'valid':
                    length = len(file_name)
                    if length >= _DIV:
                        file_name = sample(file_name, length // _DIV)
                if logdir:
                    with open(os.path.join(logdir, f'{name}_filelist.txt'), 'w') as _fp:
                        for item in file_name:
                            _fp.write(item + '\n')
                for item in file_name:
                    self.total_input_path.append(os.path.join(input_path, f'{item[:-1]}.jpg'))
                    self.total_label_path.append(os.path.join(label_path, f'{item[:-1]}.png'))
        self.mode = mode
        self.transform = transform
#         if self.mode == 'train':
#             if 'dog' in self.file_list[0]:
#                 self.label = 1 
#             else :
#                 self.label = 0 
    # need to define __len__
    def __len__(self):
        return len(self.total_input_path)
    # need to define __getitem__
    def __getitem__(self, idx):
        input_img = np.array(Image.open(self.total_input_path[idx]))
        (h, w, c) = input_img.shape
        input_img = input_img.reshape((c, h, w))
        label_img = np.array(Image.open(self.total_label_path[idx]))
        if self.transform:
            input_img = self.transform(input_img)
            label_img = self.transform(label_img)
        if self.mode == 'train':
            return input_img.astype('float32'), label_img.astype('float32'), (self.total_input_path[idx].split('/')[-1], self.total_label_path[idx].split('/')[-1])
        else :
            return input_img.astype('float32'), label_img.astype('float32'), (self.total_input_path[idx].split('/')[-1], self.total_label_path[idx].split('/')[-1])
