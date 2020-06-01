import torch
from torch.utils import data
import os

import argparse
import datetime
import yaml

from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
parser.add_argument('--resume', help='checkpoint path')

parser.add_argument(
    '--max-iteration', type=int, default=100000, help='max iteration'
)
parser.add_argument(
    '--lr', type=float, default=1.0e-10, help='learning rate',
)
parser.add_argument(
    '--weight-decay', type=float, default=0.0005, help='weight decay',
)
parser.add_argument(
    '--momentum', type=float, default=0.99, help='momentum',
)
args = parser.parse_args()

args.model = 'FCN32s'

now = datetime.datetime.now()
here = os.path.dirname(os.path.abspath(__file__))

args.out = os.path.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

os.makedirs(args.out)
with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
    yaml.safe_dump(args.__dict__, f, default_flow_style=False)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()

torch.manual_seed(1337)
if cuda:
    torch.cuda.manual_seed(1337)

class ANGIODataset(data.Dataset):
    def __init__(self, mode='train', transform = None):
        from dataset import metadata
        from dataset import hrhcase_metadata

        targets = metadata.values()
        _hrhcase_targets = hrhcase_metadata.values()

        self.total_targets = list(targets) + list(_hrhcase_targets)
        self.data_root = os.path.join('../tensorflow-deeplab-v3/dataset/boramae/transformed_dir')

        self.total_input_path = []
        self.total_label_path = []
        
        _input_dir = 'jpegs'
        _label_dir = 'transformed'
        for target in tqdm(self.total_targets):
            name = target['dirname']
            target_root = os.path.join(self.data_root, name)
            input_path = os.path.join(target_root, _input_dir)
            label_path = os.path.join(target_root, _label_dir)
            file_name = []
            with open(os.path.join(target_root, f'{name}_filelist.txt')) as fp:
                file_name = fp.readlines()
            for item in file_name:
                self.total_input_path.append(os.path.join(input_path, f'{file_name}.jpg'))
                self.total_label_path.append(os.path.join(label_path, f'{file_name}.png'))
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
        input_img = Image.open(self.total_input_path[idx])
        label_img = Image.open(self.total_label_path[idx])
        if self.transform:
            input_img = self.transform(input_img)
            label_img = self.transform(label_img)
        if self.mode == 'train':
            input_img = input_img.numpy()
            label_img = label_img.numpy()
            return input_img.astype('float32'), label_img.astype('float32')
        else :
            input_img = input_img.numpy()
            label_img = label_img.numpy()
            return input_img.astype('float32'), label_img.astype('float32')

cuda = False
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

angio_dataset = ANGIODataset('train', transform=None)

import pdb; pdb.set_trace()

train_loader = torch.utils.data.DataLoader(
    angio_dataset,
    batch_size=1, shuffle=True, **kwargs)

from model import model

