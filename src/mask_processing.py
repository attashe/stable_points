import numpy as np
import cv2
import torch
import torch.nn.functional as F

def smooth_mask(mask, size, sigma) -> np.ndarray:
    mask = mask.astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (size, size), sigma)
    return mask

def dilate_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return mask

def erode_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=iterations)
    
    return mask

def close_small_holes(mask, min_size=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_size:
            cv2.drawContours(mask, [contour], -1, 0, -1)
            
    return mask


def maxpool2d_closing(image: np.ndarray, mask: np.ndarray):

    im_floodfill = mask.copy()

    h, w = image.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    
    ret, imf, maskf, rect = cv2.floodFill(im_floodfill, flood_mask, (0,0), 255)
    maskf = maskf[1:-1, 1:-1]
    
    img_conv = torch.tensor(image, dtype=torch.float32)
    img_conv = torch.permute(img_conv, (2, 0, 1)).unsqueeze(0)

    img_max = F.max_pool2d(img_conv, 3, 1, 1)
    img_max_numpy = img_max.permute(0, 2, 3, 1).numpy()[0]
    img_max_numpy = img_max_numpy.astype(np.uint8)
    
    mask_conv = 255 - mask
    mask_conv = mask_conv.reshape(1, 1, h, w)

    mask_max = F.max_pool2d(torch.tensor(mask_conv.astype(np.float32)), 3, 1, 1)
    mask_max_numpy = mask_max.numpy().astype(np.uint8)[0][0]
    mask_max_numpy = 255 - mask_max_numpy
    
    img_max_numpy[maskf == 1] = 0
    mask_max_numpy[maskf == 1] = 255