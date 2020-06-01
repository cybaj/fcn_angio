import torch
import os

import argparse
import datetime
import yaml

from tqdm import tqdm

from model import get_model
from angio import ANGIODataset
from trainer import Trainer

from torchsummary import summary

def get_parameters(model, bias=None):
    for name, param in model.named_parameters():
        if name.startswith('backbone'):
            pass
        elif name.startswith('classifier'):
            names = name.split('.')
            if bias and 'bias' in names:
                yield param
            elif not bias and 'weight' in names:
                yield param
        else:
            raise ValueError('Unexpected module: %s' % str(m))

def main():

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

    args.model = 'FCN'
    
    # PATH
    now = datetime.datetime.now()
    here = os.path.dirname(os.path.abspath(__file__))

    args.out = os.path.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    # SEED
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    
    # DATASET LOADER
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    train_angio_dataset = ANGIODataset('train', transform=None, logdir=args.out)
    valid_angio_dataset = ANGIODataset('valid', transform=None, logdir=args.out)

    train_loader = torch.utils.data.DataLoader(train_angio_dataset,
                                                batch_size=1, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_angio_dataset,
                                                batch_size=1, shuffle=True, **kwargs)

    # RESUME
    model = get_model()
    model.cuda()
    # summary(model, (3, 128, 128))

    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        # vgg16 = torchfcn.models.VGG16(pretrained=True)
        # model.copy_params_from_vgg16(vgg16)
        pass
    if cuda:
        model = model.cuda()

    
    # OPTIMIZER
    optim = torch.optim.SGD(
    [
        {'params': get_parameters(model, bias=False)},
        {'params': get_parameters(model, bias=True),
         'lr': args.lr * 2, 'weight_decay': 0},
    ],
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
    
    # TRAIN
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=valid_loader, # no val_loader defined
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    
if __name__ == "__main__":
    main()    
