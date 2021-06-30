import torch
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from visdom import Visdom
import pdb

viz = Visdom(server='http://0.0.0.0', port=6006)

def hook_fn(module, input, output):
    # print('hooked')
    # print(f'input : {input[0].shape}')
    # print(f'output : {output[0].shape}')
    # print(f'{module} input : {input[0].shape}')
    # print(f'{module} output : {output["out"].shape}')
    pass

def hook_base_fn(module, input, output):
    print(f'input : {input[0].shape}')
    print(f'output : {output[0].shape}')
    viz.image(input[0].cpu().numpy()[0], opts=dict(
        title='input image'
        ))

def hook_pool4_fn(module, input, output):
    print(f'input : {input[0].shape}')
    print(f'output : {output[0].shape}')
    viz.image(output[0].cpu().numpy()[0], opts=dict(
        title='pool4 output 0 channel'
        ))
    viz.image(output[0].cpu().numpy()[1], opts=dict(
        title='pool4 output 1 channel'
        ))

def hook_pool5_fn(module, input, output):
    print(f'input : {input[0].shape}')
    print(f'output : {output[0].shape}')
    viz.image(output[0].cpu().numpy()[0], opts=dict(
        title='pool5 output 0 channel'
        ))
    viz.image(output[0].cpu().numpy()[1], opts=dict(
        title='pool5 output 1 channel'
        ))


def hook_feature_fn(module, input, output):
    print(f'input : {input[0].shape}')
    print(f'output : {output[0].shape}')
    viz.image(input[0].cpu().numpy()[0][0], opts=dict(
        title='input 0 channel'
        ))
    viz.image(input[0].cpu().numpy()[0][1], opts=dict(
        title='input 1 channel'
        ))
    viz.image(output[0].cpu().numpy()[0], opts=dict(
        title='output 0 channel'
        ))
    viz.image(output[0].cpu().numpy()[1], opts=dict(
        title='output 1 channel'
        ))
    pdb.set_trace()
     
    pass


def hook_vision_fn(module, input, output):
    # print('hooked')
    # print(f'input : {input[0].shape}')
    # print(f'output : {output["out"].shape}')
    pass

def get_model():
    model = torch.hub.load('pytorch/vision:v0.5.0', 'fcn_resnet101', pretrained=False, num_classes=2)
    model.eval()
    # print(model)
    # model.backbone.conv1.register_forward_hook(hook_fn)
    # model.classifier[0].register_forward_hook(hook_fn)
    # model.classifier[4].register_forward_hook(hook_fn)
    model.register_forward_hook(hook_vision_fn)
    return model

def test_model():
    model = torch.hub.load('pytorch/vision:v0.5.0', 'fcn_resnet101', pretrained=True)
    model.eval()
    return model

class FCN32(nn.Module):
  def __init__(self):
    super(FCN32, self).__init__()
    vgg16 = models.vgg16(pretrained=True)
    for param in vgg16.features.parameters():
        param.requires_grad = False

    self.features = vgg16.features
    self.classifier = nn.Sequential(
      nn.Conv2d(512, 4096, 7),
      nn.ReLU(inplace=True),
      #nn.Dropout2d(),
      nn.Conv2d(4096, 4096, 1),
      nn.ReLU(inplace=True),
      #nn.Dropout2d(),
      nn.Conv2d(4096, 21, 1),
      nn.ConvTranspose2d(21, 1, 224, stride=32)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

class FCN16(nn.Module):
  def __init__(self):
    super(FCN16, self).__init__()
    vgg16 = models.vgg16(pretrained=True)
    for param in vgg16.features.parameters():
        param.requires_grad = False

    self.features = vgg16.features
    # print(self.features)
    # self.classifier = nn.Sequential(OrderedDict([
    #     ('conv1', nn.Conv2d(512, 4096, 3)),
    #     ('batchnorm1', nn.BatchNorm2d(512)),
    #     ('relu1', nn.ReLU(inplace=True)),
    #     ('dropout1', nn.Dropout(p=0.1, inplace=False)),
    #     ('conv2', nn.Conv2d(512, 2, 1)),
    #     ('relu2', nn.ReLU(inplace=True)),
    #     ('conv3', nn.Conv2d(4096, 2, 1))
    #   ])
    # )

    self.classifier = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(512, 4096, 1)), # fc6
        # ('bn1', nn.BatchNorm2d(4096)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(4096, 4096, 1)), # fc7
        # ('bn2', nn.BatchNorm2d(4096)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(4096, 2, 1)), # score_fr
        ('bn2', nn.BatchNorm2d(2))
      ])
    )
    self.score_pool4 = nn.Conv2d(512, 2, 1)
    self.upscore2 = nn.ConvTranspose2d(2, 2, 2, stride=2, bias=False) # upscore2
    self.upscore16 = nn.ConvTranspose2d(2, 2, 16, stride=16, bias=False)
    self.upscore16.weight = nn.parameter.Parameter(torch.stack((torch.full((2,16,16), 0.), torch.full((2,16,16), 1.)), dim=1))
    # self.upscore16.weight = nn.parameter.Parameter(torch.full((1,2,16,16), 1.))

    self.features_1 = self.features[:-7]
    self.features_2 = self.features[-7:]

    self.bn_pool4 = nn.BatchNorm2d(2)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    pool4 = self.features_1(x)
    pool5 = self.features_2(pool4)
    pool5_upscored = self.upscore2(self.classifier(pool5))
    pool4_scored = self.score_pool4(pool4)
    pool4_scored = self.bn_pool4(pool4_scored)
    combined = pool4_scored + pool5_upscored
    res = self.upscore16(combined)
    res = self.sigmoid(res)
    return res

def get_vgg():
    model = FCN16()
    model.train()
    print(model)
    # model.register_forward_hook(hook_fn)
    # model.classifier.conv1.register_forward_hook(hook_fn)
    # model.classifier.conv2.register_forward_hook(hook_fn)
    # model.classifier.conv3.register_forward_hook(hook_fn)
    # model.score_pool4.register_forward_hook(hook_fn)
    # model.features_2.register_forward_hook(hook_fn)
    # model.classifier.relu1.register_forward_hook(hook_fn)

    # model.upscore2.register_forward_hook(hook_pool5_fn)
    # model.features_1.register_forward_hook(hook_base_fn)
    # model.score_pool4.register_forward_hook(hook_pool4_fn)
    # model.upscore16.register_forward_hook(hook_feature_fn)
    return model

