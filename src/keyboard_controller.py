from loguru import logger
import dearpygui.dearpygui as dpg

from context import Context
from render_panel import update_render_view


def increase_camera_alpha():
    Context.render.camera.alpha += Context.alpha_step

def decrease_camera_alpha():
    Context.render.camera.alpha -= Context.alpha_step

def increase_camera_beta():
    Context.render.camera.beta += Context.beta_step

def decrease_camera_beta():
    Context.render.camera.beta -= Context.beta_step
    
def increase_camera_theta():
    Context.render.camera.theta += Context.theta_step

def decrease_camera_theta():
    Context.render.camera.theta -= Context.theta_step
    
def increase_camera_radius():
    Context.render.camera.radius += Context.radius_step 

def decrease_camera_radius():
    Context.render.camera.radius -= Context.radius_step

# Translate center of the arcball camera
def increase_camera_center_x():
    Context.render.camera.center[0] += Context.translation_step
    
def decrease_camera_center_x():
    Context.render.camera.center[0] -= Context.translation_step
    
def increase_camera_center_y():   
    Context.render.camera.center[1] += Context.translation_step

def decrease_camera_center_y():
    Context.render.camera.center[1] -= Context.translation_step
    
def increase_camera_center_z():
    Context.render.camera.center[2] += Context.translation_step
    
def decrease_camera_center_z():
    Context.render.camera.center[2] -= Context.translation_step

# Translate center of the Turntable camera
def increase_camera_translation_x():
    Context.render.camera.translation_x += Context.translation_step
    
def decrease_camera_translation_x():
    Context.render.camera.translation_x -= Context.translation_step
    
def increase_camera_translation_y():   
    Context.render.camera.translation_y += Context.translation_step

def decrease_camera_translation_y():
    Context.render.camera.translation_y -= Context.translation_step
    
def increase_camera_translation_z():
    Context.render.camera.translation_z += Context.translation_step
    
def decrease_camera_translation_z():
    Context.render.camera.translation_z -= Context.translation_step
    
def increase_camera_rotation_x():
    Context.render.camera.rotation_x += Context.rotation_step
    
def decrease_camera_rotation_x():
    Context.render.camera.rotation_x -= Context.rotation_step
    
def increase_camera_rotation_y():
    Context.render.camera.rotation_y += Context.rotation_step
    
def decrease_camera_rotation_y():
    Context.render.camera.rotation_y -= Context.rotation_step
    
def increase_camera_rotation_z():
    Context.render.camera.rotation_z += Context.rotation_step
    
def decrease_camera_rotation_z():
    Context.render.camera.rotation_z -= Context.rotation_step


def on_key_press(sender, app_data):
    if dpg.is_item_hovered("stack_panel_1"):
        # Select camera mode between turntable and arcball
        
        mode = Context.camera_mode
        control_mode = Context.control_mode
        
        if mode == 'arcball':
            if control_mode == 'rotate':
                logger.debug('rotate arcball camera')
                if app_data == dpg.mvKey_W:
                    logger.info('W')
                    increase_camera_beta()
                elif app_data == dpg.mvKey_S:
                    logger.info('S')
                    decrease_camera_beta()
                elif app_data == dpg.mvKey_A:
                    logger.info('A')
                    decrease_camera_alpha()
                elif app_data == dpg.mvKey_D:
                    logger.info('D')
                    increase_camera_alpha()
                elif app_data == dpg.mvKey_Z:
                    logger.info('Z')
                    decrease_camera_radius()
                elif app_data == dpg.mvKey_X:
                    logger.info('X')
                    increase_camera_radius()
                elif app_data == dpg.mvKey_Q:
                    logger.info('Q')
                    increase_camera_theta()
                elif app_data == dpg.mvKey_E:
                    logger.info('E')
                    decrease_camera_theta()
                else:
                    return
            elif control_mode == 'translate':
                logger.debug('translate arcball camera')
                if app_data == dpg.mvKey_W:
                    logger.info('W')
                    increase_camera_center_z()
                elif app_data == dpg.mvKey_S:
                    logger.info('S')
                    decrease_camera_center_z()
                elif app_data == dpg.mvKey_A:
                    logger.info('A')
                    decrease_camera_center_x()
                elif app_data == dpg.mvKey_D:
                    logger.info('D')
                    increase_camera_center_x()
                elif app_data == dpg.mvKey_Z:
                    logger.info('Z')
                    increase_camera_center_y()
                elif app_data == dpg.mvKey_X:
                    logger.info('X')
                    decrease_camera_center_y()
                else:
                    return
            else:
                raise ValueError(f'Unknown control mode {control_mode}')
        elif mode == 'turntable':
            if app_data == dpg.mvKey_W:
                logger.info('W')
                increase_camera_translation_x()
            
        with dpg.mutex():
            update_render_view()

    logger.info(f'Sender: {sender}')
    logger.info(f'App data: {app_data}')
    # logger.info(f'Mouse hover element: {dpg.is_item_hovered("render window")}')
