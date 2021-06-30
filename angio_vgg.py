from random import sample, seed
from torch.utils import data
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from datetime import datetime
from preprocessing import preprocess_tensor, preprocess_resize

# divider number of total data
_DIV = 60

# seed
seed(100)


class ANGIODataset(data.Dataset):
    def __init__(self, mode='train', index=0, transform = None, logdir = None):
        from dataset import metadata, hrhcase_metadata, get_dataset, get_testset, get_trainsubset

        targets = metadata.values()
        _hrhcase_targets = hrhcase_metadata.values()

        self.total_targets = list(targets) + list(_hrhcase_targets)
        self.data_root = os.path.join('../tensorflow-deeplab-v3/dataset/boramae/transformed_dir')

        if mode == 'train':
            self.trainset = get_trainsubset(index=index)
        elif mode == 'valid':
            self.trainset = get_testset()
    
        self.total_input_path = []
        self.total_label_path = []
        self.class_names = ["NO", "EA"]
        

        if logdir and not os.path.exists(os.path.join(logdir, 'filelist')):
            os.mkdir(os.path.join(logdir, 'filelist'))

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
                    if not _DIV == 1 and length >= _DIV:
                        file_name = sample(file_name, length // _DIV)
                if logdir:
                    with open(os.path.join(logdir, 'filelist', f'{name}_filelist.txt'), 'w') as _fp:
                        for item in file_name:
                            _fp.write(item + '\n')
                for item in file_name:
                    self.total_input_path.append(os.path.join(input_path, f'{item[:-1]}.jpg'))
                    self.total_label_path.append(os.path.join(label_path, f'{item[:-1]}.png'))
        self.mode = mode
        self.transform = transform

        # self.default_transform = preprocess_tensor(512,512,2,3)
        self.default_transform = preprocess_resize(224,224,2,3)

    # need to define __len__
    def __len__(self):
        return len(self.total_input_path)
    # need to define __getitem__
    def __getitem__(self, idx):
        input_img = np.array(Image.open(self.total_input_path[idx]))
        label_img = np.array(Image.open(self.total_label_path[idx]))

        ih, iw, ic = input_img.shape

        try:
            if self.transform:
                label_img = label_img.reshape(ih,iw,1)
                stack_img = np.dstack((label_img, input_img))
        except:
            with open('failed.txt', 'a') as fp:
                fp.write(self.total_input_path[idx]+' '+self.total_label_path[idx]+' '+str(datetime.now())+'\n')
                fp.flush()

            # sample dataset for error set
            input_img = np.array(Image.open(self.total_input_path[5]))
            label_img = np.array(Image.open(self.total_label_path[5]))

            ih, iw, ic = input_img.shape
            if self.transform:
                label_img = label_img.reshape(ih,iw,1)
                stack_img = np.dstack((label_img, input_img))
        
        if self.transform:
            stack_img = self.transform(stack_img)
            label_img, input_img = torch.split(stack_img, [1,3], 0)
            label_img = (label_img * (1/0.003)).byte()
        else:
            input_img = self.default_transform(input_img)
            label_img = self.default_transform(label_img)
            label_img = (label_img * (1/0.003)).byte()
            # input_img = torch.tensor(input_img.reshape(3, ih, iw))
        if self.mode == 'train':
            # return batch shape
            #   input_img shape : N x 3 x 512 x 512, 
            #   label_img shape : N x 1 x 512 x 512
            return input_img, label_img, (self.total_input_path[idx].split('/')[-1], self.total_label_path[idx].split('/')[-1])
        else :
            return input_img, label_img, (self.total_input_path[idx].split('/')[-1], self.total_label_path[idx].split('/')[-1])
