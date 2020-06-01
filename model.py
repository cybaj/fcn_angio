import torch

def get_model():
    model = torch.hub.load('pytorch/vision:v0.5.0', 'fcn_resnet101', pretrained=False, num_classes=2)
    model.eval()
    return model
