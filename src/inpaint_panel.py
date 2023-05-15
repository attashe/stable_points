import dearpygui.dearpygui as dpg
from loguru import logger

import cv2
import webuiapi
import numpy as np
from PIL import Image
from pathlib import Path

from context import Context
from sd_scripts.inpaint import Inpainter, InpainterStandart
from utils import show_info, split_grid, combine_grid


class InpaintPanelWidget:

    def __init__(self) -> None:
        self.color_corrention = False
        self.strict_mask = False
        self.start_shift = False
        self.sampler_name = 'Euler a'
        self.inpaint_fill = 0
        
        # with dpg.group(label='Stable Diffusion'):
        with dpg.collapsing_header(label="Stable Diffusion"):
            with dpg.theme() as self.button_yellow:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 150, 20), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 200, 20), category=dpg.mvThemeCat_Core)
                    
            with dpg.theme() as self.button_green:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 150, 20), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (20, 220, 20), category=dpg.mvThemeCat_Core)

            dpg.add_checkbox(label='Use automatic1111 API', callback=self.automatic_checker, default_value=Context.use_automatic_api)

            with dpg.group(horizontal=True):
            # Add inpainting button
                dpg.add_button(label='Load Model', tag='load_model_button_tag', callback=self.load_model_callback)
                self.color_button = dpg.add_button(label='   ')
                dpg.bind_item_theme(self.color_button, self.button_yellow)
                
            dpg.add_checkbox(label='color correction', default_value=self.color_corrention,
                             callback=self.color_correction_callback)
            dpg.add_checkbox(label='strict mask', default_value=self.strict_mask,
                             callback=self.strict_mask_callback)
            dpg.add_checkbox(label='Start code shift', default_value=self.start_shift,
                             callback=self.start_shift_callback)

            dpg.add_checkbox(label='tiling inpaint', default_value=Context.tiling_inpaint,
                             callback=self.tiling_callback)
            dpg.add_slider_int(label='Tile size', default_value=Context.tile_size,
                               min_value=512, max_value=1024, callback=self.tile_size_slider_callback)
            dpg.add_slider_int(label='Tile overlap', default_value=Context.tile_overlap,
                               min_value=0, max_value=384, callback=self.tile_overlap_slider_callback)
            dpg.add_button(label="Inpaint", callback=self.inpaint_callback)

            # Add a inputs for inpainting parameters
            dpg.add_text("Inpainting parameters")
            dpg.add_input_text(label='Prompt', tag='prompt', multiline=True, default_value='Emma Watson portrait')
            dpg.add_input_text(label='Negative prompt', tag='negative_prompt', multiline=True, default_value='blurry, ugly, malformed')
            dpg.add_input_int(label='seed', tag='seed', default_value=1234, min_value=0, max_value=100000000)
            dpg.add_button(label='random seed',
                        callback=lambda _, __: dpg.set_value('seed', np.random.randint(0, 100000000)))

            dpg.add_slider_int(label='ddim steps', tag='ddim_steps', default_value=20, min_value=1, max_value=150)
            dpg.add_slider_float(label='scale', tag='scale', default_value=7, min_value=1, max_value=20, format='%.1f',
                                callback=lambda _, __: dpg.set_value('scale', round(dpg.get_value('scale') / 0.5) * 0.5))
        
        with dpg.collapsing_header(label="Img2Img SD"):
            dpg.add_text('Img2Img settings')
            dpg.add_checkbox(label='Use ControlNet', tag='controlnet_checker',
                             default_value=Context.use_controlnet, callback=self.controlnet_checker)
            dpg.add_combo(
                    [Context.api_model_name,],
                    tag='model_list_selector',
                    callback=self.load_automatic_model_callback,
                )
            dpg.add_checkbox(label='Transform inpaint result', tag='transform_inpaint_checker',
                             default_value=Context.transform_inpaint, callback=self.use_inpaint_checker)
            dpg.add_slider_float(label='denoising', tag='denoising_strength_slider',
                                 default_value=0.7, min_value=0, max_value=1, format='%.2f')
            dpg.add_button(label='Run Img2Img model', tag='img2img_button', callback=self.img2img_api)
            
            dpg.add_slider_int(label='inpaint fill', default_value=self.inpaint_fill, min_value=0, max_value=3,
                              callback=self.inpaint_fill_callback)
            
            dpg.add_combo(
                    ['DDIM', 'Euler a'],
                    default_value=self.sampler_name,
                    tag='sampler_selector',
                    callback=self.sampler_selector_callback,
                )

    def init_api(self):
        dpg.bind_item_theme(self.color_button, self.button_yellow)
        Context.api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)
        Context.api.util_set_model(Context.api_model_name)
        
        try:
            models = Context.api.util_get_model_names()
            dpg.configure_item('model_list_selector', items=models)
        except Exception() as e:
            show_info('Error', 'Cannot get list of Automatic models')
            logger.error('Cannot get list of Automatic models')
        dpg.bind_item_theme(self.color_button, self.button_green)
    
    def tiling_callback(self, sender):
        val = dpg.get_value(sender)
        Context.tiling_inpaint = val
    
    def inpaint_fill_callback(self, sender):
        val = dpg.get_value(sender)
        self.inpaint_fill = val
    
    def color_correction_callback(self, sender):
        val = dpg.get_value(sender)
        self.color_corrention = val
        
    def strict_mask_callback(self, sender):
        val = dpg.get_value(sender)
        self.strict_mask = val
        
    def start_shift_callback(self, sender):
        val = dpg.get_value(sender)
        self.start_shift = val
        
    def sampler_selector_callback(self, sender):
        val = dpg.get_value(sender)
        self.sampler_name = val
        
    def tile_size_slider_callback(self, sender):
        val = dpg.get_value(sender)
        val_fix = val - val % 64
        logger.debug('Tile size set to {val_fix}, rounded from {val}')
        dpg.set_value(sender, val_fix)
        Context.tile_size = val_fix
        
    def tile_overlap_slider_callback(self, sender):
        val = dpg.get_value(sender)
        val_fix = val - val % 32
        logger.debug('Tile overlap set to {val_fix}, rounded from {val}')
        dpg.set_value(sender, val_fix)
        Context.tile_overlap = val_fix
    
    def load_automatic_model_callback(self, sender):
        val = dpg.get_value(sender)
        Context.api_model_name = val
        Context.api.util_set_model(Context.api_model_name)
        
    def img2img_api(self):
        if Context.use_automatic_api == False:
            show_info('Need API', "Img2Img transform doesn't work without Automatic API enabled")
            return
        if Context.api is None:
            self.init_api()
            
        use_inpaint_result = Context.transform_inpaint
        
        seed = dpg.get_value('seed')
        prompt = dpg.get_value('prompt')
        negative_prompt = dpg.get_value('negative_prompt')
        ddim_steps = dpg.get_value('ddim_steps')
        scale = dpg.get_value('scale')
        strength = dpg.get_value('denoising_strength_slider')
        
        if use_inpaint_result:
            img_pil = Image.fromarray(Context.inpainted_image)
        else:
            img_pil = Image.fromarray(Context.rendered_image)

        # unit1 = webuiapi.ControlNetUnit(input_image=img_pil, module='canny', model='control_canny-fp16 [e3fe7712]')
        unit2 = webuiapi.ControlNetUnit(input_image=img_pil, module='depth', model='control_depth-fp16 [400750f6]', weight=1.0)
        controlnets = [unit2,] if Context.use_controlnet else []
        
        logger.info(f'Start inference with automatic API with next parameters:')
        logger.info(f'{prompt=}, {use_inpaint_result=}, {ddim_steps=}, {scale=}, {strength=}, {Context.use_controlnet=}, {seed=}')
        logger.info(f'size = {Context.image_width}x{Context.image_height}')
        
        img2img_result = Context.api.img2img(prompt=prompt,
                    negative_prompt=negative_prompt,
                    images=[img_pil], 
                    width=Context.image_width,
                    height=Context.image_height,
                    # controlnet_units=[unit1, unit2],
                    controlnet_units=controlnets,
                    sampler_name="Euler a",
                    steps=ddim_steps,
                    cfg_scale=scale,
                    seed=seed,
                    eta=1.0,
                    denoising_strength=strength,
                )
        res = np.array(img2img_result.image)
        Context.inpainted_image = res
        
        Context.view_panel.update(inpaint=Context.inpainted_image)
    
    def automatic_inference(self):
        if Context.api is None:
            self.init_api()
        
        seed = dpg.get_value('seed')
        prompt = dpg.get_value('prompt')
        negative_prompt = dpg.get_value('negative_prompt')
        ddim_steps = dpg.get_value('ddim_steps')
        scale = dpg.get_value('scale')
        sampler = self.sampler_name
        inpainting_fill = self.inpaint_fill
        
        # TODO: remove this assert
        assert Context.rendered_image.shape[0] == Context.image_height and Context.rendered_image.shape[1] == Context.image_width
        
        inpainting_result = Context.api.img2img(
            images=[Image.fromarray(Context.rendered_image)],
            mask_image=Image.fromarray(Context.mask),
            inpainting_fill=inpainting_fill,
            width=Context.image_width,
            height=Context.image_height,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=ddim_steps,
            cfg_scale=scale,
            mask_blur=8,
            eta=1.0,
            denoising_strength=1.0,
            sampler_name=sampler,
        )

        res = np.array(inpainting_result.image)
        Context.inpainted_image = res
        
        Context.view_panel.update(inpaint=Context.inpainted_image)
    
    def inpaint_callback(self):
        if Context.use_automatic_api:
            self.automatic_inference()
            return
        
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
        
        if Context.tiling_inpaint:
            image_resized = Image.fromarray(image_resized)
            inpaint_mask = Image.fromarray(inpaint_mask)
            
            grid_image = split_grid(image_resized, tile_w=Context.tile_size,
                                    tile_h=Context.tile_size, overlap=Context.tile_overlap)
            
            grid_mask = split_grid(inpaint_mask, tile_w=Context.tile_size,
                                   tile_h=Context.tile_size, overlap=Context.tile_overlap)
            
            work_results = []
            for (y, h, row_img), (y1, h1, row_mask) in zip(grid_image.tiles, grid_mask.tiles):
                for tiledata_img, tiledata_mask in zip(row_img, row_mask):
                    # work.append(tiledata_img[2], tiledata_mask[2])
                    result = Context.inpainter.inpaint(
                        tiledata_img[2], tiledata_mask[2],
                        prompt, seed, scale, ddim_steps, num_samples,
                        w=Context.tile_size, h=Context.tile_size)
                    
                    work_results += result
            
            image_index = 0
            for y, h, row in grid_image.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (Context.tile_size, Context.tile_size))
                    image_index += 1

            combined_image = combine_grid(grid_image)
            res = np.array(combined_image)
        else:
            results = Context.inpainter.inpaint(Image.fromarray(image_resized), Image.fromarray(inpaint_mask),
                        prompt, seed, scale, ddim_steps, num_samples, w=new_w, h=new_h,
                        color_correction=self.color_corrention, masking=self.strict_mask, start_shift=self.start_shift)
            
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
        
    def automatic_checker(self, sender):
        val = dpg.get_value(sender)
        logger.debug(f'Set depthmap checker to {val}')
        if val:
            Context.use_automatic_api = True
        else:
            Context.use_automatic_api = False
            
    def controlnet_checker(self, sender):
        val = dpg.get_value(sender)
        logger.debug(f'Set controlnet checker to {val}')
        if val:
            Context.use_controlnet = True
        else:
            Context.use_controlnet = False
            
    def use_inpaint_checker(self, sender):
        val = dpg.get_value(sender)
        logger.debug(f'Set inpaint checker to {val}')
        if val:
            Context.transform_inpaint = True
        else:
            Context.transform_inpaint = False
