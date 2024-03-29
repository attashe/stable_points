import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from loguru import logger

import torch
import torch.nn.functional as F

from context import Context
from utils import show_info
from render_panel import update_render_view
from mask_processing import erode_mask, dilate_mask, close_small_holes, maxpool2d_closing, smooth_mask


def refine_render():

    if Context.rendered_image is None:
        logger.warning("Rendered image is None")
        return

    image = Context.rendered_image
    mask = Context.mask
    
    mask_bin = mask == 0
    
    # Create layer with count of non zero neighbors pixels in 3x3 window for each pixel in mask
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    h, w = mask.shape[:2]
    mask_conv = mask_bin.reshape(1, 1, h, w)
    mask_conv = torch.tensor(mask_conv, dtype=torch.uint8)
    kernel = torch.tensor(kernel.reshape(1, 1, 3, 3))

    count_neighbors = F.conv2d(mask_conv, kernel, padding=1)
    
    mask_count = count_neighbors.numpy()[0, 0]
    
    # Create layer with mean of neighbors pixels in 3x3 window for each pixel in image
    img_conv = torch.tensor(image, dtype=torch.float32)
    img_conv = torch.permute(img_conv, (2, 0, 1)).unsqueeze(0)

    kernel = kernel.to(torch.float32)

    sum_colors = torch.zeros(1, 3, h, w)
    sum_colors[:, 0, ...] = F.conv2d(img_conv[:, 0, ...], kernel, padding=1)
    sum_colors[:, 1, ...] = F.conv2d(img_conv[:, 1, ...], kernel, padding=1)
    sum_colors[:, 2, ...] = F.conv2d(img_conv[:, 2, ...], kernel, padding=1)
    
    image_filled = sum_colors.permute(0, 2, 3, 1).numpy()[0]
    
    # Same thing for depthmask
    # TODO: Think about optional depthmap editing, but this needs for controlnet consistency
    depth = Context.rendered_depth[..., None] if len(Context.rendered_depth.shape) == 2 else Context.rendered_depth
    depth_conv = torch.tensor(depth, dtype=torch.float32)
    depth_conv = torch.permute(depth_conv, (2, 0, 1)).unsqueeze(0)
    
    sum_depth = F.conv2d(depth_conv, kernel, padding=1)
    
    depth_filled = sum_depth.permute(0, 2, 3, 1).numpy()[0]

    # Fill image with mean of neighbors pixels in 3x3 window for each non zero pixel in image with count of non zero neighbors pixels in 3x3 window more than 0
    m = (mask_bin == 0) & (mask_count > 4)
    image[m] = image_filled[m] / (mask_count[m]).reshape(-1, 1)
    mask[m] = 0
    depth[m] = depth_filled[m] / (mask_count[m]).reshape(-1, 1)
    
    # np.copyto(Context.rendered_image, image)
    # np.copyto(Context.mask, mask)
    update_depth_mask()
    
    # np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
    # np.copyto(Context.texture_data, Context.rendered_image.astype(np.float32) / 255)


def update_depth_mask():
    if Context.use_depthmap_instead_mask:
        depth_img = Context.image_wrapper.depth2img(Context.rendered_depth)
        Context.view_panel.update(render=Context.rendered_image, mask=depth_img)
    else:
        mask_img = Context.image_wrapper.mask2img(Context.mask)
        Context.view_panel.update(render=Context.rendered_image, mask=mask_img)


def depthmap_checker(sender):
    val = dpg.get_value(sender)
    logger.debug(f'Set depthmap checker to {val}')
    if val:
        Context.use_depthmap_instead_mask = True
    else:
        Context.use_depthmap_instead_mask = False

    # update_render_view()
    update_depth_mask()

def kernel_erode_input_callback(sender):
    val = dpg.get_value(sender)
    logger.debug(f'Erode kernel try set to {val}')
    if val % 2 == 0:
        dpg.set_value(sender, val - 1)
        return
    Context.kernel_erode = val

def iters_erode_input_callback(sender):
    val = dpg.get_value(sender)
    logger.debug(f'Erode iterations set to {val}')
    Context.iters_erode = val

def kernel_dilate_input_callback(sender):
    val = dpg.get_value(sender)
    logger.debug(f'Dilate kernel try set to {val}')
    if val % 2 == 0:
        dpg.set_value(sender, val - 1)
        return
    Context.kernel_dilate = val

def iters_dilate_input_callback(sender):
    val = dpg.get_value(sender)
    logger.debug(f'Dilate iterations set to {val}')
    Context.iters_dilate = val


class MaskPanel:
    
    def __init__(self) -> None:
        with dpg.collapsing_header(label="Mask edit"):
            dpg.add_checkbox(label='Use depthmap instead mask', callback=depthmap_checker, default_value=False)
            
            self.kernel_slider_erode = dpg.add_slider_int(label="kernel", default_value=Context.kernel_erode, min_value=1, max_value=31,
                                                     callback=kernel_erode_input_callback)
            self.iterations_cnt_erode = dpg.add_slider_int(label="iterations", default_value=Context.iters_erode, min_value=1, max_value=5,
                                                      callback=iters_erode_input_callback)
                
            dpg.add_button(label="Erode", callback=self.erode_mask_callback)
            
            dpg.add_separator()
            
            # Dilate
            kernel_slider_dilate = dpg.add_slider_int(label="kernel", default_value=Context.kernel_dilate, min_value=1, max_value=31,
                                                      callback=kernel_dilate_input_callback)
            iterations_cnt_dilate = dpg.add_slider_int(label="iterations", default_value=Context.iters_dilate, min_value=1, max_value=5,
                                                       callback=iters_dilate_input_callback)
            
            def dilate_mask_callback(sender, app_data):
                kernel = dpg.get_value(kernel_slider_dilate)
                # assert kernel % 2 == 1, "Kernel size must be odd"
                if kernel % 2 == 0:
                    show_info("Error", "Kernel size must be odd")
                    return

                iterations = dpg.get_value(iterations_cnt_dilate)
                Context.mask = dilate_mask(Context.mask, kernel, iterations)
                Context.rendered_depth = dilate_mask(Context.rendered_depth, kernel, iterations)
                
                if Context.use_depthmap_instead_mask:
                    depth_img = Context.image_wrapper.depth2img(Context.rendered_depth)
                    Context.view_panel.update(mask=depth_img)
                else:
                    mask_img = Context.image_wrapper.mask2img(Context.mask)
                    Context.view_panel.update(mask=mask_img)
                # Context.mask = dilate_mask(Context.mask, kernel, iterations)
                # np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
            
            dpg.add_button(label="Dilate", callback=dilate_mask_callback)
            
            dpg.add_separator()
            # Close the small holes in the mask
            
            hole_min_size_slider = dpg.add_slider_int(label="hole_min_size", default_value=10, min_value=1, max_value=100)
        
            def close_holes_callback(sender, app_data):
                hole_min_size = dpg.get_value(hole_min_size_slider)
                Context.mask = 255 - close_small_holes(255 - Context.mask, hole_min_size)
                Context.rendered_image[Context.mask != 0] = 0
                Context.rendered_depth[Context.mask != 0] = 0
                
                if Context.use_depthmap_instead_mask:
                    depth_img = Context.image_wrapper.depth2img(Context.rendered_depth)
                    Context.view_panel.update(mask=depth_img)
                else:
                    mask_img = Context.image_wrapper.mask2img(Context.mask)
                    Context.view_panel.update(mask=mask_img)
                # np.copyto(Context.texture_data, Context.rendered_image.astype(np.float32) / 255)
                # np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
            
            dpg.add_button(label="Close", callback=close_holes_callback)
            
            dpg.add_button(label="Refine render aliasing", callback=refine_render)
            
            def max_pulling_callback(sender, app_data):
                # Context.mask = max_pulling(Context.mask)
                mask = Context.mask
                image = Context.rendered_image
                
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
                
                np.copyto(Context.rendered_image, img_max_numpy)
                np.copyto(Context.mask, mask_max_numpy)
                
                np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
                np.copyto(Context.texture_data, Context.rendered_image.astype(np.float32) / 255)
            
            dpg.add_button(label='Max pulling', callback=max_pulling_callback)
            
            size_slider = dpg.add_slider_int(label="size", default_value=Context.smooth_size, min_value=1, max_value=31,
                                             callback=self.size_smooth_callback)
            sigma_slider = dpg.add_slider_float(label="sigma", default_value=Context.smooth_sigma, min_value=0.0, max_value=10.0,
                                                callback=self.sigma_smooth_callback)
            
            def smooth_mask_callback(sender, app_data):
                Context.mask = smooth_mask(Context.mask, dpg.get_value(size_slider), dpg.get_value(sigma_slider))
                # np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
                if Context.use_depthmap_instead_mask:
                    show_info('Depthmap show', 'Cannot smoothing depthmap, transformation applied only for mask')
                else:
                    mask_img = Context.image_wrapper.mask2img(Context.mask)
                    Context.view_panel.update(mask=mask_img)
            
            dpg.add_button(label='Smooth Mask', callback=smooth_mask_callback)
            
            def reset_mask_callback(sender, app_data):
                Context.mask = (Context.rendered_depth == 0).astype(np.uint8) * 255
                # np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
                mask_img = Context.image_wrapper.mask2img(Context.mask)
                Context.view_panel.update(mask=mask_img)
            
            dpg.add_button(label='Reset mask', callback=reset_mask_callback)

    def size_smooth_callback(self, sender):
        val = dpg.get_value(sender)
        logger.debug(f'Smooth size set to {val}')
        Context.smooth_size = val
    
    def sigma_smooth_callback(self, sender):
        val = dpg.get_value(sender)
        logger.debug(f'Smooth sigma set to {val}')
        Context.smooth_sigma = val
    
    def erode_mask_callback(self):
        kernel = Context.kernel_dilate
        # assert kernel % 2 == 1, "Kernel size must be odd"
        if kernel % 2 == 0:
            show_info("Error", "Kernel size must be odd")
            return
        iterations = Context.iters_dilate

        Context.mask = erode_mask(Context.mask, kernel, iterations)
        Context.rendered_depth = erode_mask(Context.rendered_depth, kernel, iterations)
        if Context.use_depthmap_instead_mask:
            depth_img = Context.image_wrapper.depth2img(Context.rendered_depth)
            Context.view_panel.update(mask=depth_img)
        else:
            mask_img = Context.image_wrapper.mask2img(Context.mask)
            Context.view_panel.update(mask=mask_img)
        
        # np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
    
    def update_mask(self):
        # TODO: make special function for show depthmap and mask
        pass