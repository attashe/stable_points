import numpy as np
import dearpygui.dearpygui as dpg

from loguru import logger
from pathlib import Path
from PIL import Image

from context import Context
from utils import show_info, save_pil_image
from cycles.action import Action
from render_panel import update_render_view
from pcl_transform_panel import transform_pointcloud_with_vortex_torch
from mask_processing import erode_mask, dilate_mask, smooth_mask


class VortexAction(Action):
    
    def __init__(self, func=None) -> None:
        pass
    
    def run(self, state, params):
        vortex_strength = params['vortex_strength']


def create_vortex_pipeline():
    action1 = Action(func_1)
    action2 = Action(func_2)
    
    steps = 3

    scenario = Scenario([action1, action2])

def vortex_action():
    pointcloud = Context.render.points.copy()
    center = pointcloud.mean(axis=0)
    logger.debug(f'{center=}')
    vortex_strength = Context.vortex_strength / 100
    chaos_strength = Context.chaos_strength
    chaos_fraction = Context.chaos_fraction
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis = axis_map[dpg.get_value('axis_selector')]
    
    logger.debug(f'{Context.render.points.sum()}')
    logger.debug(f'{vortex_strength=}')
    logger.debug(f'{chaos_strength=}')
    

    Context.render.points = transform_pointcloud_with_vortex_torch(pointcloud, center[0], center[1], center[2],
                                                                    vortex_strength=vortex_strength, chaos=chaos_strength,
                                                                    rotate_axis=axis, chaos_fraction=chaos_fraction,
                                                                    device='cuda')


def save_render_action():
    render_save_path = Path(Context.log_folder) / "render" / ('render_' + str(Context.render_image_idx).zfill(5) + '.png')
    
    save_pil_image(render_save_path, Image.fromarray(Context.rendered_image))
    
    Context.image_path = render_save_path
    Context.render_image_idx += 1


def play_video(self):
    pass


def move_camera_action(dx, dy, dz, alpha, beta, theta):
    # TODO: function for reset camera position
    camera = Context.render.camera
    x, y, z = camera.get_position()
    camera.set_position(x + dx, y + dy, z + dz)
    # camera.radius = 0.01  # Context.image_height * Context.downscale
    camera.alpha += alpha
    camera.beta += beta
    camera.theta += theta
    update_render_view()


def dilate_action():
    kernel = Context.kernel_dilate
    iterations = Context.iters_dilate
    
    Context.mask = dilate_mask(Context.mask, kernel, iterations)
    Context.rendered_depth = dilate_mask(Context.rendered_depth, kernel, iterations)

def smooth_action():
    logger.debug(f'Smooth action with {Context.smooth_size} size and {Context.smooth_sigma} sigma')
    Context.mask = smooth_mask(Context.mask, Context.smooth_size, Context.smooth_sigma)

def update_mask_action():
    mask_img = Context.image_wrapper.mask2img(Context.mask)
    Context.view_panel.update(mask=mask_img)

def inpaint_action():
    Context.inpaint_panel.inpaint_callback()

def zoomin_action(zoom_scale):
    zoomout_action(-zoom_scale)

def zoomout_action(zoom_scale):
    # 1. Define parameters
    z_step = - zoom_scale
    alpha_step = Context.alpha_step * 0
    beta_step = Context.beta_step * 0
    theta_step = Context.theta_step * 0
    
    # 2. Set camera
    move_camera_action(0, 0, z_step, alpha_step, beta_step, theta_step)
    # 3. Dilate + Smooth mask
    dilate_action()
    smooth_action()
    update_mask_action()
    # 4. Inpaint
    inpaint_action()
    
def restart_image():
    pass
    

class ScenarioPanel:
    
    def __init__(self) -> None:
        self.repeats = 10
        self.zoom_scale = 1.0
        
        with dpg.collapsing_header(label='Scenarios'):
            dpg.add_input_int(label='Repeats', default_value=self.repeats, min_value=1, max_value=100,
                              callback=self.repeats_input_callback)
            
            dpg.add_button(label='Run vortex scenario', callback=self.vortex_scenario_callback)
            
            dpg.add_separator()
            dpg.add_input_float(label='Zoom scale', default_value=self.zoom_scale, min_value=0.01, max_value=50.0,
                                callback=self.zoom_input_callback)
            dpg.add_button(label='Run zoom in scenario', callback=self.zoomin_scenario_callback)
            dpg.add_button(label='Run zoom out scenario', callback=self.zoomout_scenario_callback)
    
    def repeats_input_callback(self, sender):
        val = dpg.get_value(sender)
        self.repeats = val
        
    def zoom_input_callback(self, sender):
        val = dpg.get_value(sender)
        self.zoom_scale = val
    
    def create_vortex_scenario(self):
        action2 = vortex_action()
        action1 = Action(update_render_view)
        action3 = save_render()
    
    def vortex_scenario_callback(self):
        # TODO: Replace to Scenario Framework
        for i in range(self.repeats):
            vortex_action()
            update_render_view()
            save_render_action()
    
    def zoomin_scenario_callback(self):
        for i in range(self.repeats):
            zoomin_action()
            update_render_view()
            Context.restart_function()
            # save_render_action()
    
    def zoomout_scenario_callback(self):
        for i in range(self.repeats):
            zoomout_action(self.zoom_scale)
            update_render_view()
            Context.restart_function()
            # save_render_action()