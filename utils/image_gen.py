import torch
from PIL import Image as PILImage
import numpy as np
from torch import nn
import cv2
from functools import reduce

def segmentation_to_image(segmentation,image,palette, output_size=(244, 244)):
    grayed = gray_image(image,output_size)
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    segmentation = interp(segmentation.unsqueeze(0)).cpu().numpy().transpose(0,2,3,1)
    seg_pred = np.asarray(np.argmax(segmentation, axis=3), dtype=np.uint8)
    output_im = PILImage.fromarray(seg_pred[0])
    output_im.putpalette(palette)
    output_im = output_im.convert('RGB')
    result = cv2.addWeighted(np.array(output_im), 0.7, grayed, 0.3, 0)
    return result

def attention_to_images(image,attention_map,output_size=(244,244), normalize='local'):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    cvImage = gray_image(image,output_size)
    attention_map = interp(attention_map).squeeze(1).cpu().detach().numpy()
    ticks=[(attention_map.min()),(attention_map.max())]
    normalized = global_normalize(cvImage, attention_map) if normalize == 'global' else local_normalize(cvImage, attention_map)
    return normalized, ticks

def masked_attention_images(original,segmentation, attention_map, output_size=(244,244)):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    seg= interp(segmentation.unsqueeze(0)).permute([0,1,2,3]).squeeze(0)
    seg = torch.from_numpy(np.asarray(np.argmax(seg, axis=0), dtype=np.uint8)).long()
    seg = torch.nn.functional.one_hot(seg, num_classes=19).permute([2,0,1]).float()
    attention_matrix = interp(attention_map).squeeze()
    masked = torch.mul(seg, attention_matrix).numpy()
    cvImage = gray_image(original,output_size)
    return masked, np.array(global_normalize(cvImage, masked, 0))
    

def shape_attention(attention_map, dim=None):
    attention_map = attention_map.mean(dim=1, keepdim=True)
    attention_size = attention_map.size()
    if dim is None:
        single_dim = int(attention_size[2]**(0.5))
        dim = (single_dim,single_dim)
    attention_map = attention_map.view((attention_size[0],1,dim[0],dim[1]))
    return attention_map

def clear_zeros(attention):
    _min = np.where(attention != 0.0, attention, np.inf).min()
    return np.where(attention != 0.0, attention, _min)

def gray_image(image,output_size):
    try:
        cvImage = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR)
    except AttributeError:
        cvImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR) #we neeed a 3 dimensional gray image
    cvImage = cv2.resize(cvImage, output_size)
    return cvImage

def global_normalize(image,attention_map, mask_value=-1):
    images = []
    heatmap_img = None
    heatmap_img = normalize_attention(attention_map, mask_value)
    for single_map in heatmap_img:
        single_img = cv2.applyColorMap(single_map, cv2.COLORMAP_JET)
        result = cv2.addWeighted(single_img, 0.5, image, 0.5, 0)
        result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        images.append(result)
    return images

def local_normalize(image,attention_map, mask_value=-1):
    images = []
    for single_map in attention_map:
        heatmap_img = None
        heatmap_img = normalize_attention(single_map, mask_value)
        img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        result = cv2.addWeighted(img, 0.5, image, 0.5, 0)
        result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        images.append(result)
    return images

def normalize_attention(attention, mask_value=-1):
    input_min = np.min(attention[attention > mask_value])
    input_max = attention.max()
    return (np.maximum((attention - input_min),0)/(input_max - input_min) * 255).astype('uint8')

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def image_log(segmentation,original,attention_map,palette, normalize='local'):
    seg_img = segmentation_to_image(segmentation, original, palette)
    attentions, ticks = attention_to_images(original, attention_map, normalize=normalize)
    return seg_img, attentions, ticks
