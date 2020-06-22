from torchvision import transforms
import math

def preprocess_image(target_height, target_width, min_scale, max_scale):
    # image and label are PIL Image.
    # image.shape = (height, width, channel)
    # label.shape = (1, height, width) 

    # shape = image.size
    # h = shape[0] 
    # w = shape[1]
    # c = 1
    # if len(shape) == 3:
    #     c = shape[2]
    # scale = (max_scale - min_scale) * torch.rand(1) + min_scale

    # v_pad = math.ceil(max(target_height, h) // 2)
    # h_pad = math.ceil(max(target_width, w) // 2)
    
    composing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(0, None, (min_scale, max_scale)), 
            transforms.RandomCrop((target_height, target_width), 
                pad_if_needed=True), 
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ToTensor()
            ])

    return composing


def preprocess_empty(target_height, target_width, min_scale, max_scale):
    composing = transforms.Compose([
            ])
    return composing

def preprocess_PIL(target_height, target_width, min_scale, max_scale):
    composing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((target_height, target_width), 
                pad_if_needed=True), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            ])
    return composing

def preprocess_tensor(target_height, target_width, min_scale, max_scale):
    composing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ])
    return composing

def preprocess_resize(target_height, target_width, min_scale, max_scale):
    composing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_height, target_width)), 
            transforms.ToTensor(),
            ])
    return composing

