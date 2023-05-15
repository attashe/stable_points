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
from depth_infer import DepthModel, AdaBinsDepthPredict, LeResInfer, ZoeInfer
from utils import dpg_get_value
from render_panel import update_render_view


def update_focal_length(sender):
    value = dpg.get_value(sender)
    Context.focal_length = value
    
    Context.render.camera.focal_length = Context.focal_length
    update_render_view()


def update_depth_resolution(sender):
    value = dpg.get_value(sender)
    Context.depth_resolution = value


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

def remove_percent_points(points, threshold, lower_bound=True) -> tuple[np.ndarray, np.ndarray]:
    percentile = np.percentile(points[:, 2], 100 - threshold * 100)
    
    if lower_bound:
        indices = points[:, 2] >= percentile
    else:
        indices = points[:, 2] < percentile
        
    return points[indices], indices

def depth_threshold_callback(sender):
    value = dpg.get_value(sender)
    logger.debug(f'Change depth threshold to {value}')
    Context.depth_thresh = value

def remove_foreground_callback(sender):
    points, indices = remove_percent_points(Context.render.points, Context.depth_thresh, lower_bound=True)
    
    Context.render.points = points
    Context.render.colors = Context.render.colors[indices]
    
    update_render_view()

def remove_background_callback(sender):
    points, indices = remove_percent_points(Context.render.points, Context.depth_thresh, lower_bound=False)
    
    Context.render.points = points
    Context.render.colors = Context.render.colors[indices]
    
    update_render_view()


class DepthPanel:
    
    def __init__(self) -> None:
        self.loaded_model = None  # 'midas' | 'adabins' | 'leres'
        
        # with dpg.group(label="Depth"):
        with dpg.collapsing_header(label="Depth"):
            def update_depth_scale(sender):
                Context.depthscale = dpg.get_value(sender)
                print(f'depth_scale: {Context.depthscale}')
            
            # Add a depthscale slider
            dpg.add_text("Depth Scale")
            depth_scale_slider = dpg.add_slider_float(label='Depthscale', tag="float_depth_scale", 
                                                      default_value=Context.depthscale,
                                                      min_value=0.1, max_value=250.0)
            dpg.set_item_callback(depth_scale_slider, update_depth_scale)
            
            dpg.add_text('Depth inference resolution')
            depth_resolution_slider = dpg.add_input_int(label='resolution', tag='depth_res_slider',
                                                        default_value=Context.depth_resolution,
                              min_value=256, max_value=1536)
            dpg.set_item_callback(depth_resolution_slider, update_depth_resolution)
            
            # TODO: unify all depth models to one API and remove duplicates for model types
            self.selector_tag = 'depth_model_selector'
            dpg.add_combo(
                    ['midas', 'adabins', 'leres', 'zoe_depth'],
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
            dpg.add_text('Remove fore/background')
            dpg.add_slider_float(label='Threshold', min_value=0.0, max_value=1.0, default_value=Context.depth_thresh,
                                 callback=depth_threshold_callback)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label='Remove Background', callback=remove_background_callback)
                dpg.add_button(label='Remove Foreground', callback=remove_foreground_callback)
            
            dpg.add_separator()
            dpg.add_text('Depth gamma correction')
            dpg.add_slider_float(label='gamma', tag='gamma_slider', default_value=Context.depth_gamma, min_value=0.01, max_value=5)
            dpg.add_button(label='Apply gamma correction', tag='gamma_corr_button', callback=self.apply_gamma_corr)
            
            dpg.add_separator()
            dpg.add_text('Depth sigmoid correction')
            dpg.add_slider_float(label='alpha', tag='alpha_slider', default_value=Context.depth_alpha, min_value=0.1, max_value=10)
            dpg.add_button(label='Apply sigmoid correction', tag='sigmoid_corr_button', callback=self.apply_sigmoid_correction)

        dpg.add_separator()

    def apply_gamma_corr(self, sender):
        gamma = dpg.get_value('gamma_slider')
        Context.depth_gamma = gamma

        eps = 1e-5
        
        # Get depth points
        depth_points = Context.render.points[:, 2]
        dmin = depth_points.min()
        dmax = depth_points.max()
        
        if abs(dmax - dmin) < eps:
            logger.info('Depth map is uniform')
            return

        # Transform to [0, 1] range
        depth_points = (depth_points - dmin) / (dmax - dmin)
        # Apply gamma correction
        depth_points = depth_points ** gamma
        # Return range back
        depth_points = depth_points * (dmax - dmin) + dmin
        
        Context.render.points[:, 2] = depth_points
        
        update_render_view()       
        
    def apply_sigmoid_correction(self, sender):
        alpha = dpg.get_value('alpha_slider')
        Context.depth_alpha = alpha

        eps = 1e-5
        
        # Get depth points
        depth_points = Context.render.points[:, 2]
        dmin = depth_points.min()
        dmax = depth_points.max()
        dmean = depth_points.mean()

        if abs(dmax - dmin) < eps:
            logger.info('Depth map is uniform')
            return

        # Transform to [-1, 1] range
        depth_points = (depth_points - dmean) / (dmax - dmin) * 2
        # Apply gamma correction
        sigmoid = lambda x: 1 / (1 + np.e ** (- alpha * x))
        # [-1, 1] -> [sigm(-1), sigm(1)] with center in (0.5) -> | alpha -> inf | -> [0, 1]
        depth_points = sigmoid(depth_points)
        # Set range to [0, 1]
        depth_points = (depth_points - depth_points.min()) / (depth_points.max() - depth_points.min())
        # Set range back
        depth_points = depth_points * (dmax - dmin) + dmin
        
        Context.render.points[:, 2] = depth_points
        
        update_render_view()   
    
    def load_model_callback(self, sender):
        self.init_model()

    def init_model(self):
        model_name = dpg_get_value(self.selector_tag)
        if self.loaded_model == model_name:
            logger.info(f'Model {model_name} already loaded')
            return
        Context.depth_type = model_name
        logger.info(f'Loading depth model {model_name}')
        start_time = time.time()
        dpg.bind_item_theme(self.color_button, self.button_yellow)
        
        del Context.depth_model
        Context.depth_model = None
        if Context.depth_type == 'midas':
            Context.depth_model = DepthModel(device='cuda', resolution=Context.depth_resolution)
            Context.depth_model.load_midas()
            self.loaded_model = 'midas'
        elif Context.depth_type == 'adabins':
            raise Exception('Not implemented error')
        elif Context.depth_type == 'leres':
            Context.depth_model = LeResInfer()
            self.loaded_model = 'leres'
        elif Context.depth_type == 'zoe_depth':
            Context.depth_model = ZoeInfer()
            self.loaded_model = 'zoe_depth'
        
        logger.info(f'Depth model loaded in {time.time() - start_time} seconds')
        dpg.bind_item_theme(self.color_button, self.button_green)

    def dummy_depth(self, img):
        h, w = img.shape[:2]
        return np.ones((h, w)) * 255
        
    def predict_img(self, img):
        if Context.use_depth_model == False:
            return self.dummy_depth(img)

        if Context.depth_model is None:
            self.init_model()
        
        if self.loaded_model == 'midas':
            if Context.depth_model.resolution != Context.depth_resolution:
                Context.depth_model.set_resolution(Context.depth_resolution)
            depth = Context.depth_model.predict(img)
            depth = depth.cpu().numpy()
        elif self.loaded_model == 'adabins':
            raise Exception('Not implemented error')
        elif self.loaded_model == 'leres':
            pred, pred_ori = Context.depth_model.predict_depth(img, resolution=Context.depth_resolution)
            depth = pred_ori
        elif self.loaded_model == 'zoe_depth':
            logger.debug('Inference Zoe Depth model')
            depth = Context.depth_model.predict(img, resolution=Context.depth_resolution)
            depth = depth.cpu().numpy()
            
            
        return depth