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


class ScenarioPanel:
    
    def __init__(self) -> None:
        
        with dpg.collapsing_header(label='Scenarios'):
            dpg.add_button(label='Run vortex scenario', callback=self.vortex_scenario_callback)
            dpg.add_button(label='Run zoom in scenario', callback=self.zoomin_scenario_callback)
            dpg.add_button(label='Run zoom out scenario', callback=self.zoomout_scenario_callback)
    
    def create_vortex_scenario(self):
        action2 = vortex_action()
        action1 = Action(update_render_view)
        action3 = save_render()
    
    def vortex_scenario_callback(self):
        # TODO: Replace to Scenario Framework
        count = 10
        for i in range(count):
            vortex_action()
            update_render_view()
            save_render_action()
    
    def zoomin_scenario_callback(self):
        show_info('Error', 'Function not implemented error')
    
    def zoomout_scenario_callback(self):
        show_info('Error', 'Function not implemented error')