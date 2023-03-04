"""Panel for depth model parameters and select type of depth model

1. Depth model selector and status
2. Depthscale slide
3. Apply focal correction
"""
import time
import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from context import Context
from depth_infer import DepthModel, AdaBinsDepthPredict, LeResInfer
from utils import dpg_get_value

def update_focal_length(sender):
    value = dpg.get_value(sender)
    Context.focal_length = value
    
    Context.render.camera.focal_length = Context.focal_length
    update_render_view()


def reset_camera():
    # TODO: function for reset camera position
    pass



def init_depth():
    logger.info('Loading depth model')
    start_time = time.time()
    Context.depth_model = DepthModel(device='cuda')
    Context.depth_model.load_midas()
    logger.info(f'Depth model loaded in {time.time() - start_time} seconds')


def calculate_depth(image) -> np.ndarray:
    pass

class DepthPanel:
    
    def __init__(self) -> None:
        self.loaded_model = None  # 'midas' | 'adabins' | 'leres'
        
        with dpg.group(label="Depth"):
            def update_depth_scale(sender):
                Context.depthscale = dpg.get_value(sender)
                print(f'depth_scale: {Context.depthscale}')
            
            # Add a depthscale slider
            dpg.add_text("Depth Scale")
            depth_scale_slider = dpg.add_slider_float(label='Depthscale', tag="float_depth_scale", default_value=Context.depthscale, min_value=0.1, max_value=1000.0)
            dpg.set_item_callback(depth_scale_slider, update_depth_scale)
            
            # TODO: unify all depth models to one API and remove duplicates for model types
            self.selector_tag = 'depth_model_selector'
            dpg.add_combo(
                    ['midas', 'adabins', 'leres'],
                    default_value=Context.depth_type,
                    tag=self.selector_tag,
                    callback=self.load_model_callback
                )

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
                dpg.add_button(label='Load Model', tag='load_depthmodel_button_tag', callback=self.load_model_callback)
                self.color_button = dpg.add_button(label='   ')
                dpg.bind_item_theme(self.color_button, self.button_yellow)

        dpg.add_separator()

    def load_model_callback(self, sender):
        self.init_model()

    def init_model(self):
        if self.loaded_model == dpg_get_value(self.selector_tag):
            return
        logger.info('Loading depth model')
        start_time = time.time()
        dpg.bind_item_theme(self.color_button, self.button_yellow)
        
        del Context.depth_model
        Context.depth_model = None
        if Context.depth_type == 'midas':
            Context.depth_model = DepthModel(device='cuda')
            Context.depth_model.load_midas()
            self.loaded_model = 'midas'
        elif Context.depth_type == 'adabins':
            raise Exception('Not implemented error')
        elif Context.depth_type == 'leres':
            Context.depth_model = LeResInfer()
            self.loaded_model = 'leres'
        
        logger.info(f'Depth model loaded in {time.time() - start_time} seconds')
        dpg.bind_item_theme(self.color_button, self.button_green)

    def predict_img(self, img):
        if Context.depth_model is None:
            self.init_model()
        
        if self.loaded_model == 'midas':
            depth = Context.depth_model.predict(img)
            depth = depth.cpu().numpy()
        elif self.loaded_model == 'adabins':
            raise Exception('Not implemented error')
        elif self.loaded_model == 'leres':
            pred, pred_ori = Context.depth_model.predict_depth(img)
            depth = pred_ori
            
        return depth