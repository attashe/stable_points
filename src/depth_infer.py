import sys
import torch
import math
import torchvision
import torchvision.transforms as T

import cv2
import numpy as np
from einops import rearrange, repeat
from PIL import Image

sys.path.append('G:/Python Scripts/AdaBins')
from infer import InferenceHelper

sys.path.append('G:/Python Scripts/MiDaS')
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

class DepthModel():
    def __init__(self, device):
        self.depth_min = 1000
        self.depth_max = -1000
        self.device = device
        self.midas_model = None
        self.midas_transform = None
        self.model_path = 'G:/GitHub/DeforumStableDiffusionLocal/models'

    def load_midas(self, half_precision=True):
        self.midas_model = DPTDepthModel(
            path=f"{self.model_path}/dpt_large-midas-2f21e586.pt",
            backbone="vitl16_384",
            non_negative=True,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.midas_transform = T.Compose([
            Resize(
                384, 384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()
        ])

        self.midas_model.eval()    
        if half_precision and self.device == torch.device("cuda"):
            self.midas_model = self.midas_model.to(memory_format=torch.channels_last)
            self.midas_model = self.midas_model.half()
        self.midas_model.to(self.device)

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