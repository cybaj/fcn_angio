import torch
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

def hook_fn(module, input, output):
    # print('hooked')
    # print(f'input : {input[0].shape}')
    # print(f'output : {output[0].shape}')
    # print(f'{module} input : {input[0].shape}')
    # print(f'{module} output : {output["out"].shape}')
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
        ('conv1', nn.Conv2d(512, 4096, 7)), # fc6
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(4096, 4096, 1)), # fc7
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(4096, 2, 1)) # score_fr
      ])
    )
    self.score_pool4 = nn.Conv2d(512, 2, 1)
    self.upscore2 = nn.ConvTranspose2d(2, 2, 14, stride=2, bias=False) # upscore2
    self.upscore16 = nn.ConvTranspose2d(2, 2, 16, stride=16, bias=False)

    self.features_1 = self.features[:-7]
    self.features_2 = self.features[-7:]

    self.conv_1 = nn.Conv2d(512,512,3)
    self.conv_1 = nn.Conv2d(512,512,3)

  def forward(self, x):
    pool4 = self.features_1(x)
    pool5 = self.features_2(pool4)
    pool5_upscored = self.upscore2(self.classifier(pool5))
    pool4_scored = self.score_pool4(pool4)
    combined = pool4_scored + pool5_upscored
    res = self.upscore16(combined)
    return res

def get_vgg():
    model = FCN16()
    model.train()
    model.register_forward_hook(hook_fn)
    # model.classifier.conv1.register_forward_hook(hook_fn)
    # model.classifier.conv2.register_forward_hook(hook_fn)
    # model.classifier.conv3.register_forward_hook(hook_fn)
    # model.score_pool4.register_forward_hook(hook_fn)
    # model.upscore2.register_forward_hook(hook_fn)
    # model.features_1.register_forward_hook(hook_fn)
    # model.features_2.register_forward_hook(hook_fn)
    # model.classifier.relu1.register_forward_hook(hook_fn)
    # model.upscore16.register_forward_hook(hook_fn)
    return model

