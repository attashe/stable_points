import sys
import time
from pathlib import Path
from loguru import logger
from PIL import Image
import torchvision
import numpy as np
import dearpygui.dearpygui as dpg

from context import Context
from utils import show_info, open_image, create_pointcloud, convert_from_uvd_numpy

from mask_processing import *
from image_panel import ImagePanel

from render.render import Render
# from depth_panel import init_depth
# sys.path.append('repos/ResizeRight')
from resize_right import resize_right

def upscale_image(image, factor):
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w * Context.upscale), int(h * Context.upscale)), interpolation=cv2.INTER_LANCZOS4)
    # image = resize_right.resize(image, out_shape=(w * Context.upscale, h * Context.upscale))
    logger.debug(f'Image upscaled from {w}x{h} -> {image.shape[1]} x {image.shape[0]}')
    return image


class ImageWrapper():
    
    def __init__(self, image) -> None:
        self.orig_image = image
        self.upscaled_image = upscale_image(self.orig_image, Context.upscale)
        
        self._load_points(self.upscaled_image)
        
        # h, w = self.orig_image.shape[:2]
        # Downscaled image
        # self.image = cv2.resize(self.orig_image, (int(w / Context.downscale), int(h / Context.downscale)))
        Context.image_width = image.shape[1]
        Context.image_height = image.shape[0]
        
        self._setup_render()
        
        self._init_render()
        
        self.render_image()
        
    def _load_points(self, image):
        self.points, self.colors = self._create_pointcloud(image)
        
    def _setup_render(self):
        # if Context.image_height < Context.image_width:
        Context.canvas_height = 1.0
        Context.canvas_width = Context.image_width / Context.image_height
        # else:
        #     Context.canvas_height = Context.image_height / Context.image_width
        #     Context.canvas_width = 1.0
        
        ratio = Context.image_width / Context.image_height
        # size_sum = Context.canvas_height + Context.canvas_width
        # Context.canvas_height /= size_sum 
        # Context.canvas_width /= size_sum
        
        render = Render(Context.image_width, Context.image_height,
                        Context.canvas_width, Context.canvas_height,
                        focal_length=Context.focal_length / ratio, device="cuda")
        
        Context.render = render
        
    def _init_render(self):
        Context.render.points = self.points
        Context.render.colors = self.colors
        
    def reload_image(self):
        if Context.render is not None:
            self._load_points(self.upscaled_image)
            self._init_render()
            self.render_image()
    
    def render_image(self):
        render = Context.render
        
        render.camera.set_position(0, 0, 0)
        render.camera.radius = 0.01  # Context.image_height * Context.downscale
        render.camera.alpha = 0
        render.camera.beta = - np.pi / 2
        
        depth = Context.depth_panel.predict_img(self.orig_image)

        resizer = torchvision.transforms.Resize(size=self.upscaled_image.shape[:2])
        logger.info(f'{depth.shape=}')
        depth = resizer(torch.tensor(depth).unsqueeze(0).unsqueeze(0))
        # depth = cv2.resize(depth, (Context.image_width, Context.image_height))
        logger.info(f'{depth.shape=}')
        # logger.info(f'{depth=}')
        depth = depth.numpy()
        
        depth = depth.astype(np.float32)
        data_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5) * Context.depthscale
        points_depth = data_depth.reshape(-1, 1)
        
        self.points[:, 2] = points_depth[:, 0]
        
        # convert plane points to perspective projection with the given focal length and depth
        points = convert_from_uvd_numpy(self.points, points_depth, Context.focal_length)
        render.points = points
    
    def _create_pointcloud(self, image):
        points, colors = create_pointcloud(image)
        # substract (width / 2, height / 2) to make center of pointcloud in (0, 0) point
        points -= np.array([self.upscaled_image.shape[1] / 2, self.upscaled_image.shape[0] / 2, 0])
        # points += np.random.rand(*points.shape) * 0.00001
        
        return points, colors
    
    def mask2img(self, mask):
        mask_exp = np.zeros((mask.shape[0], mask.shape[1], 3))
        mask_exp[..., :] = mask[..., :] if len(mask.shape) == 3 else mask[..., None]
        return mask_exp
    
    def depth2img(self, depth):
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth_exp = np.zeros((depth.shape[0], depth.shape[1], 3))
        depth_exp[..., :] = depth[..., :] if len(depth.shape) == 3 else depth[..., None]

        return (depth_exp).astype(np.uint8)


class ViewTriPanel:
    _n = 0    
    def __init__(self, width=512, height=512) -> None:
        ViewTriPanel._n += 1
        with dpg.child_window(tag=f'stack_panel_{ViewTriPanel._n}', horizontal_scrollbar=True):
            with dpg.group(horizontal=True):
                self.render_image = ImagePanel(width, height)
                self.mask_image = ImagePanel(width, height)
                self.inpaint_image = ImagePanel(width, height)
                
    def update(self, render=None, mask=None, inpaint=None):
        if render is not None:
            self.render_image.set_image(render)
        if mask is not None:
            self.mask_image.set_image(mask)
        if inpaint is not None:
            self.inpaint_image.set_image(inpaint)
            
    def set_size(self, width, height):
        self.render_image.change_size(new_w=width, new_h=height)
        self.mask_image.change_size(new_w=width, new_h=height)
        self.inpaint_image.change_size(new_w=width, new_h=height)


def update_render_view():
    start = time.time()
    image, depth = Context.render.render()
    
    Context.rendered_image = image
    Context.rendered_depth = depth
    
    # img_f = image.astype(np.float32) / 255
    # np.copyto(Context.texture_data, img_f)
    
    inpaint_mask = depth == 0
    inpaint_mask = inpaint_mask.astype(np.uint8) * 255
    Context.mask = inpaint_mask
    # np.copyto(Context.mask_data[..., 0], inpaint_mask.astype(np.float32) / 255)
    if Context.use_depthmap_instead_mask:
        depth_img = Context.image_wrapper.depth2img(Context.rendered_depth)
        Context.view_panel.update(render=Context.rendered_image, mask=depth_img)
    else:
        mask_img = Context.image_wrapper.mask2img(Context.mask)
        Context.view_panel.update(render=Context.rendered_image, mask=mask_img)
    
    logger.info(f'update_render_view: {(time.time() - start) * 1000 :.1f} ms')
