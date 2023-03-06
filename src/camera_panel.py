import numpy as np
import dearpygui.dearpygui as dpg
from loguru import logger

from context import Context
from render_panel import update_render_view


def update_focal_length(sender):
    value = dpg.get_value(sender)
    Context.focal_length = value
    
    Context.render.camera.focal_length = Context.focal_length
    update_render_view()


def reset_camera():
    # TODO: function for reset camera position
    camera = Context.render.camera
    camera.set_position(0, 0, - Context.image_height * 2)
    camera.radius = 0.01# Context.image_height * Context.downscale
    camera.alpha = 0
    camera.beta = - np.pi / 2
    camera.theta = 0
    update_render_view()


def reload_image_callback():
    if Context.image_wrapper is not None:
        Context.image_wrapper.render_image()
        update_render_view()


class CameraPanelWidget:
    
    def __init__(self) -> None:
        with dpg.group(label="Camera"):
        # with dpg.collapsing_header(label="Camera", default_open=True):
            dpg.add_button(label="Init Image Selector", callback=lambda: dpg.show_item("file_dialog_id"))
            
            dpg.add_text("focal length")
            focal_length_slider = dpg.add_slider_float(label="float_f", default_value=Context.focal_length, min_value=0.1, max_value=10)
            dpg.set_item_callback(focal_length_slider, update_focal_length)
            
            dpg.add_separator()
            dpg.add_text("ArcBall Camera")
            dpg.add_text("Radius")

            # Add a buttong to reset the camera
            dpg.add_button(label="Reset Camera", callback=reset_camera)

            def update_camera_mode(sender, app_data):
                logger.info(f'update_camera_mode: {app_data}')
                logger.info(f'sender: {sender}')
                if app_data == 'Arcball':
                    Context.camera_mode = "arcball"
                elif app_data == 'Turntable':
                    Context.camera_mode = "turntable"
                else:
                    raise ValueError(f'Unknown camera mode: {app_data}')

            # Add a radio button to select the camera mode
            dpg.add_radio_button(items=["Arcball", "Turntable"], callback=update_camera_mode)
            
            dpg.add_separator()
            
            dpg.add_button(label='Reload image', tag='reload_button_tag', callback=reload_image_callback)
            dpg.add_separator()
            
            def update_control_mode(sender, app_data):
                logger.info(f'update_camera_mode: {app_data}')
                logger.info(f'sender: {sender}')
                
                if app_data == 'Translate':
                    Context.control_mode = "translate"
                elif app_data == 'Rotate':
                    Context.control_mode = "rotate"
                else:
                    raise ValueError(f'Unknown camera control mode: {app_data}')
            
            # Add a radio button to select the control mode
            dpg.add_radio_button(items=["Translate", "Rotate"], default_value="Rotate", callback=update_control_mode)
            # dpg.set_value("control_mode", "rotate")
        
        dpg.add_separator()
