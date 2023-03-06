"""
Create a 3d model from a 2d image with filling in the missing parts using a pretrained model.

Steps:
1. Load image
2. Depthmap
    2.1. Generate depthmap with MiDaS model
    2.2. Load depthmap from file
3. Generate pointcloud from image and depthmap
4. Refine pointcloud
5. Render pointcloud
6. Move camera to new position
7. Re-render pointcloud
8. Refine rendered image
    8.1. Fill holes in image
    8.2. Delete noise
9. Load image model
10. Inpaint image with model
11. Save image
12. Goto to step 3
"""
import os
# import sys
# import time
# import threading

import cv2

import dearpygui.dearpygui as dpg
import numpy as np
# import torch
# import torchvision
# import torch.nn.functional as F

from pathlib import Path
from loguru import logger
from PIL import Image
# from prompt_toolkit import prompt
# from depth_infer import DepthModel, AdaBinsDepthPredict
from render.render import Render
# from inpaint import Inpainter, InpainterStandart

from context import Context
from utils import show_info, open_image, create_pointcloud, convert_from_uvd_numpy
from inpaint_panel import InpaintPanelWidget
from camera_panel import CameraPanelWidget
from keyboard_controller import on_key_press
from mask_processing import *
# from image_panel import ImagePanel
from render_panel import ViewTriPanel, update_render_view, ImageWrapper
from depth_panel import DepthPanel
from mask_processing_panel import MaskPanel


class MaskProcessingWidget:
    
    def __init__(self) -> None:
        with dpg.group(label='Mask processing'):
            # Add a buttons for erode, dilate and close the small holes in the mask
            # Erode
            
            kernel_slider_erode = dpg.add_slider_int(label="kernel", default_value=3, min_value=1, max_value=31)
            iterations_cnt_erode = dpg.add_slider_int(label="iterations", default_value=1, min_value=1, max_value=5)
            
            def erode_mask_callback(sender, app_data):
                kernel = dpg.get_value(kernel_slider_erode)
                # assert kernel % 2 == 1, "Kernel size must be odd"
                if kernel % 2 == 0:
                    show_info("Error", "Kernel size must be odd")
                    return

                iterations = dpg.get_value(iterations_cnt_erode)
                Context.mask = erode_mask(Context.mask, kernel, iterations)
                np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
                
            dpg.add_button(label="Erode", callback=erode_mask_callback)
            
            dpg.add_separator()
            # Dilate
            
            kernel_slider_dilate = dpg.add_slider_int(label="kernel", default_value=3, min_value=1, max_value=31)
            iterations_cnt_dilate = dpg.add_slider_int(label="iterations", default_value=1, min_value=1, max_value=5)
            
            def dilate_mask_callback(sender, app_data):
                kernel = dpg.get_value(kernel_slider_dilate)
                # assert kernel % 2 == 1, "Kernel size must be odd"
                if kernel % 2 == 0:
                    show_info("Error", "Kernel size must be odd")
                    return

                iterations = dpg.get_value(iterations_cnt_dilate)
                Context.mask = dilate_mask(Context.mask, kernel, iterations)
                np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
            
            dpg.add_button(label="Dilate", callback=dilate_mask_callback)
            
            dpg.add_separator()
            # Close the small holes in the mask
            
            hole_min_size_slider = dpg.add_slider_int(label="hole_min_size", default_value=10, min_value=1, max_value=100)
        
            def close_holes_callback(sender, app_data):
                hole_min_size = dpg.get_value(hole_min_size_slider)
                Context.mask = 255 - close_small_holes(255 - Context.mask, hole_min_size)
                Context.rendered_image[Context.mask != 0] = 0
                np.copyto(Context.texture_data, Context.rendered_image.astype(np.float32) / 255)
                np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
            
            dpg.add_button(label="Close", callback=close_holes_callback)
            
            dpg.add_button(label="Refine render aliasing", callback=refine_render)
            
            dpg.add_button(label='Max pulling', callback=max_pulling_callback)
            
            def smooth_mask(mask, size, sigma) -> np.ndarray:
                mask = mask.astype(np.uint8)
                mask = cv2.GaussianBlur(mask, (size, size), sigma)
                return mask
            
            size_slider = dpg.add_slider_int(label="size", default_value=3, min_value=1, max_value=31)
            sigma_slider = dpg.add_slider_float(label="sigma", default_value=0.0, min_value=0.0, max_value=10.0)
            
            def smooth_mask_callback(sender, app_data):
                Context.mask = smooth_mask(Context.mask, dpg.get_value(size_slider), dpg.get_value(sigma_slider))
                np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
            
            dpg.add_button(label='Smooth Mask', callback=smooth_mask_callback)



class SettingPanelWidget:
    
    def __init__(self) -> None:
        pass


class EditorWindow:
    
    def __init__(self) -> None:
        pass


def get_depth_tensor():
    pass


def init_render():
    image = Context.init_image
    Context.image_width = image.shape[1]
    Context.image_height = image.shape[0]
    logger.info(f'Image size: {Context.image_width}x{Context.image_height} was loaded')
    
    points, colors = create_pointcloud(image)
    points -= np.array([Context.image_width / 2, Context.image_height / 2, 0])
    points += np.random.rand(*points.shape) * 0.00001
    
    if not os.path.exists(depth_path):
        generate_depthmap_midas(image, depth_path)
        # generate_depthmap_adabins(image, depth_path)
    
    logger.info("Loading depthmap")
    depth = open_image(depth_path)
    depth = cv2.resize(depth, (Context.image_width, Context.image_height))

    depth = depth[..., 0].astype(np.float32)
    data_depth = (depth - depth.min()) / (depth.max() - depth.min()) * Context.depthscale
    points_depth = data_depth.reshape(-1, 1)
    
    print(f'img.shape: {image.shape}')
    print(f'depth.shape: {depth.shape}')
    
    # set z-values according to depth
    points[:, 2] = points_depth[:, 0]
    
    # threshold = np.percentile(points_depth, 35)
    # points = points[points_depth[:, 0] < threshold]
    # colors = colors[points_depth[:, 0] < threshold]
    # points_depth = points_depth[points_depth[:, 0] < threshold]
    # points = rescale_depth(points, 0.3, 15.0, 1.0)
    
    # points[:, 2] = (points[:, 2] - np.median(points[:, 2])) * 25

    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w / Context.downscale), int(h / Context.downscale)))
    Context.image_width = image.shape[1]
    Context.image_height = image.shape[0]
    depth = cv2.resize(depth, (Context.image_width, Context.image_height))

    Context.canvas_height = 1.0
    Context.canvas_width = Context.image_width / Context.image_height
    
    render = Render(Context.image_width, Context.image_height, Context.canvas_width, Context.canvas_height,
                    focal_length=Context.focal_length, device="cuda")
    render.points = points
    render.colors = colors
    
    render.camera.set_position(0, 0, - Context.image_height * 2)
    render.camera.radius = 0.01# Context.image_height * Context.downscale
    render.camera.alpha = 0
    render.camera.beta = - np.pi / 2
    render.camera.theta = 0
    
    Context.render = render
    
    # convert plane points to perspective projection with the given focal length and depth
    points = convert_from_uvd_numpy(points, points_depth, Context.focal_length)
    render.points = points
    
    image, depth = Context.render.render()
    
    Context.render_image_idx += 1
    
    return image, depth, points, colors


def restart_render():
    (Path(Context.log_folder) / "render").mkdir(exist_ok=True)
    render_save_path = Path(Context.log_folder) / "render" / ('render_' + str(Context.render_image_idx).zfill(5) + '.png')
    
    Image.fromarray(Context.inpainted_image).save(str(render_save_path))
    
    Context.image_path = render_save_path
    
    # load_image_file(Context.image_path)
    render_new_image(Context.inpainted_image)
    
    Context.render_image_idx += 1


def restart_render_with_current_image_callback(sender, app_data):
    restart_render()


def add_textures_zeros(tag_prefix="",):
    # Init texture data with zero values
    Context.render_data = np.zeros((Context.image_height, Context.image_width, 3), dtype=np.float32)
    Context.mask_data = np.zeros((Context.image_height, Context.image_width, 3), dtype=np.float32)
    Context.inpaint_data = np.zeros((Context.image_height, Context.image_width, 3), dtype=np.float32)
    # Init GUI textures
    with dpg.texture_registry(show=False):
        # Texture for rendering the point cloud
        dpg.add_raw_texture(width=Context.image_width, height=Context.image_height,
                            default_value=Context.render_data, format=dpg.mvFormat_Float_rgb,
                            tag="render_tag")
        # Texture for rendering the mask of the point cloud (selcted by depth and optionally dilated/eroded)
        dpg.add_raw_texture(width=Context.image_width, height=Context.image_height,
                            default_value=Context.mask_data, format=dpg.mvFormat_Float_rgb,
                            tag="mask_tag")
        # Texture for show inpainting results
        dpg.add_raw_texture(width=Context.image_width, height=Context.image_height,
                            default_value=Context.inpaint_data, format=dpg.mvFormat_Float_rgb,
                            tag="inpaint_tag")
        
        # print(f'inpaint_data.shape: {Context.inpaint_data.shape}')


def add_view_panel():
    Context.view_panel = ViewTriPanel()


def add_view_widgets():
    with dpg.child_window(parent='main_table_row', width=-1, height=-2, tag='render_window'):
        # dpg.add_text("Render will be here")
        dpg.add_image("render_tag", tag='render_image')
    
    with dpg.child_window(parent='main_table_row', width=-1, height=-2, tag='mask_window'):
        # dpg.add_text("Mask for inpainting will be here")
        dpg.add_image("mask_tag", tag='mask_image')
    
    with dpg.child_window(parent='main_table_row', width=-1, height=-2, tag='inpaint_window'):
    #     # dpg.add_text("Inpaint result will be here")
        dpg.add_image("inpaint_tag", tag='inpaint_image')


def render_new_image(image):
    h, w = image.shape[:2]
    logger.info(f'Initial image size is {w}x{h}px')
    
    Context.image_height = h #* Context.upscale // Context.downscale
    Context.image_width = w #* Context.upscale // Context.downscale
    Context.init_image = image
    
    Context.image_wrapper = ImageWrapper(image)
    Context.view_panel.set_size(w, h)
    
    update_render_view()


def load_image_file(image_path):
    image = open_image(image_path)
    render_new_image(image)


def image_select_callback(sender, app_data, user_data):
    print("Sender: ", sender)
    print("App Data: ", app_data)
    
    print(app_data)
    print(user_data)
    
    # Read image from path
    Context.image_path = app_data['file_path_name']
    load_image_file(Context.image_path)


def init_log_folder():
    result_path = Path(Context.results_folder)
    result_path.mkdir(parents=True, exist_ok=True)
    
    i = 0
    for dir in result_path.iterdir():
        if dir.is_dir():
            idx = int(dir.name.split('_')[-1])
            i = max(i, idx)
    
    log_path = result_path / (Context.basename + str(i + 1))
    log_path.mkdir()
    
    Context.log_folder = str(log_path)


def main():
    logger.info("Starting program")
    
    logger.info('Initializing output folder')
    init_log_folder()
    
    dpg.create_context()
    
    with dpg.file_dialog(directory_selector=False, show=False, callback=image_select_callback, id="file_dialog_id",
                         height=600, modal=True):
        dpg.add_file_extension("Image files (*.png *.jpg *.bmp){.png,.jpg,.bmp}", color=(0, 255, 255, 255))
    
    with dpg.window(label="render", tag='main_window', autosize=True):
        # with dpg.menu_bar():
        # dpg.toggle_viewport_fullscreen()
        with dpg.table(header_row=False, tag='table'):
            dpg.add_table_column(width=300, width_fixed=True)
            dpg.add_table_column()
            with dpg.table_row(tag='main_table_row'):
                with dpg.group(label="SidePanel"):
                    camera_widget = CameraPanelWidget()
                    
                    dpg.add_separator()
                    
                    Context.depth_panel = DepthPanel()
                    
                    sd_widget = InpaintPanelWidget()
                        
                    dpg.add_separator()

                    # Final buttons
                    def reset_mask_callback(sender, app_data):
                        Context.mask = (Context.rendered_depth == 0).astype(np.uint8) * 255
                        np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
                    
                    dpg.add_button(label='Reset mask', callback=reset_mask_callback)
                    
                    dpg.add_separator()
                    
                    dpg.add_button(label='Reset render', callback=restart_render_with_current_image_callback)
                    
                    dpg.add_separator()
                    
                    Context.mask_panel = MaskPanel()

                add_textures_zeros()
                # add_view_widgets()
                add_view_panel()
    
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=on_key_press)

    dpg.create_viewport(title='Pointcloud Engine', width=1280, height=720)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    dpg.start_dearpygui()
    # below replaces, start_dearpygui()
    # while dpg.is_dearpygui_running():
    #     # insert here any code you would like to run in the render loop
    #     # you can manually stop by using stop_dearpygui()
    #     # print("this will run every frame")
        
    #     if Context.changed:
    #         Context.changed = False
    #         # Context.render()
        
    #     dpg.render_dearpygui_frame()
    dpg.destroy_context()

    logger.info("Ending program")


if __name__ == "__main__":
    main()