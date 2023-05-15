import os
import sys
import torch
import math
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as T

import cv2
import numpy as np
from einops import rearrange, repeat
from PIL import Image

from utils import resize_padding_pil

sys.path.append('G:/Python Scripts/AdaBins')
from infer import InferenceHelper

sys.path.append('G:/Python Scripts/MiDaS')
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

sys.path.append('repos/AdelaiDepth/LeReS/Minist_Test/')
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt


class DepthModel():
    def __init__(self, device, resolution=384):
        self.depth_min = 1000
        self.depth_max = -1000
        self.device = device
        self.midas_model = None
        self.resolution = resolution
        self.midas_transform = None
        self.model_path = 'G:/GitHub/DeforumStableDiffusionLocal/models'

    def load_midas(self, half_precision=True):
        self.midas_model = DPTDepthModel(
            # path=f"{self.model_path}/dpt_large-midas-2f21e586.pt",
            # backbone="vitl16_384",
            # path=f"{self.model_path}/dpt_swin2_large_384.pt",
            # backbone="swin2l24_384",
            path=f"{self.model_path}/dpt_beit_large_512.pt",
            backbone="beitl16_512",
            non_negative=True,
        )
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.midas_transform = T.Compose([
            Resize(
                self.resolution, self.resolution,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            self.normalization,
            PrepareForNet()
        ])

        self.midas_model.eval()    
        if half_precision and self.device == torch.device("cuda"):
            self.midas_model = self.midas_model.to(memory_format=torch.channels_last)
            self.midas_model = self.midas_model.half()
        self.midas_model.to(self.device)
        
    def set_resolution(self, resolution):
        del self.midas_transform
        
        self.resolution = resolution
        self.midas_transform = T.Compose([
            Resize(
                self.resolution, self.resolution,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            self.normalization,
            PrepareForNet()
        ])

    def predict(self, img_cv2) -> torch.Tensor:
        w, h = img_cv2.shape[1], img_cv2.shape[0]

        if self.midas_model is not None:
            # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
            img_midas = img_cv2.astype(np.float32) / 255.0
            img_midas_input = self.midas_transform({"image": img_midas})["image"]

            # MiDaS depth estimation implementation
            sample = torch.from_numpy(img_midas_input).float().to(self.device).unsqueeze(0)
            if self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            with torch.no_grad():            
                midas_depth = self.midas_model.forward(sample)
            midas_depth = torch.nn.functional.interpolate(
                midas_depth.unsqueeze(1),
                size=img_midas.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            midas_depth = midas_depth.cpu().numpy()
            torch.cuda.empty_cache()

            # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
            midas_depth = np.subtract(50.0, midas_depth)
            midas_depth = midas_depth / 19.0

            depth_map = midas_depth

            depth_map = np.expand_dims(depth_map, axis=0)
            depth_tensor = torch.from_numpy(depth_map).squeeze().to(self.device)
        else:
            depth_tensor = torch.ones((h, w), device=self.device)
        
        return depth_tensor

    def save(self, filename: str, depth: torch.Tensor):
        depth = depth.cpu().numpy()
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=0)
        self.depth_min = min(self.depth_min, depth.min())
        self.depth_max = max(self.depth_max, depth.max())
        print(f"  depth min:{depth.min()} max:{depth.max()}")
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        temp = repeat(temp, 'h w 1 -> h w c', c=3)
        Image.fromarray(temp.astype(np.uint8)).save(filename)
    
        
class AdaBinsDepthPredict:
    
    def __init__(self, model_path='G:/GitHub/DeforumStableDiffusionLocal/', device='cuda'):
        self.adabins_helper = InferenceHelper(dataset='nyu', device=device, model_path=model_path)
        
    def predict(self, image_cv2):
        h, w = image_cv2.shape[:2]
        # predict depth with AdaBins  
        print(f"Estimating depth of {w}x{h} image with AdaBins...")
        MAX_ADABINS_AREA = 500000
        MIN_ADABINS_AREA = 448*448

        # resize image if too large or too small
        img_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR))
        image_pil_area = w*h
        resized = True
        if image_pil_area > MAX_ADABINS_AREA:
            scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS) # LANCZOS is good for downsampling
            print(f"  resized to {depth_input.width}x{depth_input.height}")
        elif image_pil_area < MIN_ADABINS_AREA:
            scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            print(f"  resized to {depth_input.width}x{depth_input.height}")
        else:
            depth_input = img_pil
            resized = False

        # predict depth and resize back to original dimensions
        _, adabins_depth = self.adabins_helper.predict_pil(depth_input)
        if resized:
            adabins_depth = torchvision.transforms.functional.resize(
                torch.from_numpy(adabins_depth), 
                torch.Size([h, w]),
                interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC
            )
        adabins_depth = np.array(adabins_depth.squeeze())
        print(f"  depth min:{adabins_depth.min()} max:{adabins_depth.max()}")
        print(adabins_depth.dtype)
        
        return adabins_depth
    
    def save(self, save_path, depth: np.ndarray):
        # transform = T.ToTensor()
        # image = np.asarray(Image.open(img_path), dtype='float32') / 255.
        # image = transform(image).unsqueeze(0).to(self.device)

        # centers, final = self.predict(image)
        # final = final.squeeze().cpu().numpy()

        # final = (depth * self.saving_factor).astype('uint16')
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype('uint8')

        Image.fromarray(depth.squeeze()).save(save_path)
     

# =========================================
#               LeReS Block
# =========================================
def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = T.Compose([T.ToTensor(),
                               T.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

class args:
    load_ckpt = 'J:/Weights/leres_weights/res101.pth'
    backbone = 'resnext101'

class LeResInfer:
    
    def __init__(self, resolution: int = 448, backbone='resnext101') -> None:
        # create depth model
        self.depth_model = RelDepthModel(backbone=backbone)
        self.depth_model.eval()
        
        self.resolution = resolution
        
        # load checkpoint
        load_ckpt(args, self.depth_model, None, None)
        self.depth_model.cuda()
    
    def predict_depth(self, image: np.ndarray, is_rgb=True, 
                      resolution: int = None, save_depth=False, keep_ratio=False,
                      image_dir_out='./'):
        """Predict depth for monocular image

        Args:
            image (np.ndarray): RGB image array
        """
        rgb = image if is_rgb else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_c = image[:, :, ::-1].copy()
        
        if resolution is not None:
            self.resolution = resolution
        # gt_depth = None
        if keep_ratio:
            A_resize, out_crop = resize_padding_pil(rgb_c, size=self.resolution)
        else:
            A_resize = cv2.resize(rgb_c, (self.resolution, self.resolution))
        # rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = self.depth_model.inference(img_torch).cpu().numpy().squeeze()
        
        if keep_ratio:
            pred_depth = np.array(pred_depth)[out_crop[1][0] : out_crop[1][1], out_crop[0][0] : out_crop[0][1]]
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # if GT depth is available, uncomment the following part to recover the metric depth
        # pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)
        # TODO: fix save depth function
        if save_depth:
            img_name = v.split('/')[-1]
            cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
            # save depth
            plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
            cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
        
        return pred_depth, pred_depth_ori
    
    def predict_points(self, image):
        raise Exception('Not implemented error')
    
    
class ZoeInfer:
    
    def __init__(self) -> None:
        model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=False)
        model_zoe_n.load_state_dict(torch.load('J:/GitHub/ZoeDepth/ZoeD_M12_N.pt')['model'])
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = model_zoe_n.to(DEVICE)

    
    def predict(self, image, resolution=384):
        # depth_numpy = self.zoe.infer_pil(image)  # as numpy
        # depth_pil = self.zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

        depth_tensor = self.zoe.infer_pil(image, output_type="tensor")  # as torch tensor
        
        return depth_tensor
