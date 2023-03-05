import dearpygui.dearpygui as dpg
from loguru import logger

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from context import Context
from sd_scripts.inpaint import Inpainter, InpainterStandart


def save_inpaint_callback(sender, app_data):
    filename_image = Path(Context.log_folder) / ('image_' + str(Context.save_idx).zfill(5) + '.png')
    filename_mask = Path(Context.log_folder) / ('mask_' + str(Context.save_idx).zfill(5) + '.png')
    filename_inpaint = Path(Context.log_folder) / ('inpaint_' + str(Context.save_idx).zfill(5) + '.png')
    
    saved = 0    
    if Context.rendered_image is not None:
        Image.fromarray(Context.rendered_image).save(str(filename_image))
        saved += 1
    if Context.mask is not None:
        Image.fromarray(Context.mask).save(str(filename_mask))
        saved += 1
    if Context.inpainted_image is not None:
        Image.fromarray(Context.inpainted_image).save(str(filename_inpaint))
        saved += 1
    
    logger.info(f'Saved {saved} images to {Context.log_folder} with index {Context.save_idx}')
    
    if saved > 0:
        Context.save_idx += 1


class InpaintPanelWidget:

    def __init__(self) -> None:
        with dpg.group(label='Stable Diffusion'):
            with dpg.theme() as self.button_yellow:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 150, 20), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 200, 20), category=dpg.mvThemeCat_Core)
                    
            with dpg.theme() as self.button_green:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 150, 20), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (20, 220, 20), category=dpg.mvThemeCat_Core)

            with dpg.group(horizontal=True):
            # Add inpainting button
                dpg.add_button(label='Load Model', tag='load_model_button_tag', callback=self.load_model_callback)
                self.color_button = dpg.add_button(label='   ')
                dpg.bind_item_theme(self.color_button, self.button_yellow)

            dpg.add_button(label="Inpaint", callback=self.inpaint_callback)

            # Add a inputs for inpainting parameters
            dpg.add_text("Inpainting parameters")
            dpg.add_input_text(label='Prompt', tag='prompt', multiline=True, default_value='Naked Emma Watson in photo studio, Canon EOS 50mm')
            dpg.add_input_int(label='seed', tag='seed', default_value=123, min_value=0, max_value=100000000)
            dpg.add_button(label='random seed',
                        callback=lambda _, __: dpg.set_value('seed', np.random.randint(0, 100000000)))

            dpg.add_slider_int(label='ddim steps', tag='ddim_steps', default_value=20, min_value=1, max_value=150)
            dpg.add_slider_float(label='scale', tag='scale', default_value=7, min_value=1, max_value=20, format='%.1f',
                                callback=lambda _, __: dpg.set_value('scale', round(dpg.get_value('scale') / 0.5) * 0.5))

            # Save results
            dpg.add_button(label='Save', tag='save', callback=save_inpaint_callback)
    
    def inpaint_callback(self, sender, app_data):
        if Context.inpainter is None:
            self.init_inpainting()
        
        # seed = 228
        # prompt = "Naked woman in photo studio"
        # ddim_steps = 20
        # num_samples = 1
        # scale = 7
        seed = dpg.get_value('seed')
        prompt = dpg.get_value('prompt')
        ddim_steps = dpg.get_value('ddim_steps')
        scale = dpg.get_value('scale')
        num_samples = 1
        
        w, h = Context.image_width, Context.image_height
        # w, h = w // 2, h // 2  # Inpainting is slow and needs too much GPU, so we downscale the image
        new_w, new_h = w - w % 64, h - h % 64
        inpaint_mask = cv2.resize(Context.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        image_resized = cv2.resize(Context.rendered_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        print(image_resized.shape)
        # print(image_resized)
        
        print(inpaint_mask.shape)
        print(type(inpaint_mask))
        
        results = Context.inpainter.inpaint(Image.fromarray(image_resized), Image.fromarray(inpaint_mask), prompt,
                                seed, scale, ddim_steps, num_samples, w=new_w, h=new_h)
        
        res = np.array(results[0])
        res = cv2.resize(res, (Context.image_width, Context.image_height), interpolation=cv2.INTER_LANCZOS4)
        Context.inpainted_image = res
        
        Context.view_panel.update(inpaint=Context.inpainted_image)
        # np.copyto(Context.inpaint_data, res.astype(np.float32) / 255)
    
    def load_model_callback(self, sender, app_data):
        logger.info('Start loading inpaint model')
        self.init_inpainting()
        logger.info('Model was loaded')
        
    def init_inpainting(self):
        if Context.inpainter is not None: return
        config = 'G:/GitHub/stable-diffusion/configs/stable-diffusion/v1-inpainting-inference.yaml'
        ckpt = 'G:/Weights/stable-diffusion/sd-v1-5-inpainting.ckpt'
        Context.inpainter = Inpainter(ckpt, config, device='cuda')
        dpg.bind_item_theme(self.color_button, self.button_green)

        # default stable diffusion v1.4
        # config = 'G:/GitHub/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
        # ckpt = 'G:/Weights/sd-v-1-4-original/sd-v1-4.ckpt'
        # Context.inpainter = InpainterStandart(ckpt, config, device='cuda')