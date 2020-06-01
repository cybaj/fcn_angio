def preprocess_image(image, label, is_training):
    # image.shape = (height, width, channel)
    # label.shape = (1, height, width) 
     

    return image, label

def random_rescale_image_and_label(image, label, min_scale, max_scale):

    if min_scale <= 0:
      raise ValueError('\'min_scale\' must be greater than 0.')
    elif max_scale <= 0:
      raise ValueError('\'max_scale\' must be greater than 0.')
    elif min_scale >= max_scale:
      raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

    c, h, w = image.shape

    scale = (max_scale - min_scale) * torch.rand(1) + min_scale
    new_height = h * scale
    new_width = w * scale

    image = 
    
    
    return image, label

def random_crop_or_pad_image_and_label(image, label, crop_height, crop_width, ignore_label):

    return image, label

def random_flip_left_right_image_and_label(image, label):

    return image, label

