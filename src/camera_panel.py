import numpy as np
import dearpygui.dearpygui as dpg
from loguru import logger

from context import Context
from render_panel import update_render_view, ImageWrapper
from utils import clear_pointcloud


def update_focal_length(sender):
    value = dpg.get_value(sender)
    Context.focal_length = value
    
    Context.render.camera.focal_length = Context.focal_length
    update_render_view()


def reset_camera():
    # TODO: function for reset camera position
    camera = Context.render.camera
    camera.set_position(0, 0, 0)
    camera.radius = 0.01  # Context.image_height * Context.downscale
    camera.alpha = 0
    camera.beta = - np.pi / 2
    camera.theta = 0
    update_render_view()


def reload_image_callback():
    if Context.image_wrapper is not None:
        Context.image_wrapper.reload_image()
        update_render_view()
        
        
def clean_alone_points():
    points, indices = clear_pointcloud(Context.render.points, Context.points_thresh, Context.points_radius)
    
    Context.render.points = points
    Context.render.colors = Context.render.colors[indices]
    
    update_render_view()

def set_neighbors(sender, __):
    Context.points_thresh = dpg.get_value(sender)
    
def set_radius(sender, __):
    Context.points_radius = dpg.get_value(sender)

def upscale_callback(sender, __):
    Context.upscale = dpg.get_value(sender)
    logger.debug(f'Update upscale value to {Context.upscale}')

def set_resized_image(image):
    h, w = image.shape[:2]
    logger.info(f'Initial image size is {w}x{h}px')
    
    Context.image_height = h #* Context.upscale // Context.downscale
    Context.image_width = w #* Context.upscale // Context.downscale
    Context.init_image = image
    
    Context.image_wrapper = ImageWrapper(image)
    Context.view_panel.set_size(w, h)
    
    update_render_view()

class CameraPanelWidget:
    
    def __init__(self) -> None:
        self.up, self.down, self.left, self.right = 0, 0, 0, 0
        
        with dpg.group(label="Camera"):
        # with dpg.collapsing_header(label="Camera", default_open=True):
            dpg.add_button(label="Init Image Selector", callback=lambda: dpg.show_item("file_dialog_id"))
            dpg.add_slider_float(label='Pointcloud upscale', default_value=2, min_value=1, max_value=3,
                                 callback=upscale_callback)
            
            dpg.add_checkbox(label='Use depth model', tag='use_depth_checkbox', 
                             default_value=Context.use_depth_model , callback=self.use_depth_checker)
            
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
            
            dpg.add_button(label='Remove noise points', callback=clean_alone_points)
            dpg.add_slider_int(label='neighbors', default_value=Context.points_thresh, min_value=1, max_value=10,
                               callback=set_neighbors)
            dpg.add_slider_float(label='radius', default_value=Context.points_radius, min_value=0.1, max_value=5)
            
            dpg.add_separator()
            # Pad and crop controls
            # TODO: Finish this controls
            dpg.add_text("Pad & Crop")
            dpg.add_text("Up, Down, Left, Right")
            with dpg.group(horizontal=True):
                dpg.add_input_intx(callback=self.set_padcrop_vals)
                dpg.add_button(label='Crop', callback=self.crop_image_callback)
                dpg.add_button(label='Pad', callback=self.pad_image_callback)
                # dpg.add_input_intx(label='Left, Right')
        
        dpg.add_separator()
        
    def set_padcrop_vals(self, sender, data):
        logger.debug(f'InputX data is {data}')
        self.up, self.down, self.left, self.right = data
        
    def crop_image_callback(self, sender):
        logger.info('Crop image button was pressed')
        image = Context.init_image
        
        logger.debug(f'Resize image from image size {image.shape[1]}x{image.shape[0]}')
        
        f = lambda x: x if x != 0 else None
        
        cropped_image = image[f(self.up) : f(-self.down), f(self.left) : f(-self.right)]
        
        logger.debug(f'Image resized to image size {cropped_image.shape[1]}x{cropped_image.shape[0]}')
        
        set_resized_image(cropped_image)

    def pad_image_callback(self, sender):
        logger.info('Pad image button was pressed')
        image: np.ndarray = Context.init_image
        
        logger.debug(f'Resize image from image size {image.shape[1]}x{image.shape[0]}')
        
        f = lambda x: x if x != 0 else None
        
        w, h = image.shape[1], image.shape[0]
        new_w, new_h = w + self.left + self.right, h + self.up + self.down
        padded_img = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
        padded_img[f(self.up) : f(-self.down), f(self.left) : f(-self.right)] = image
        
        logger.debug(f'Image resized to image size {padded_img.shape[1]}x{padded_img.shape[0]}')
        
        set_resized_image(padded_img)

    def use_depth_checker(self, sender):
        val = dpg.get_value(sender)
        logger.debug(f'Set use depth model to {val}')
        if val:
            Context.use_depth_model = True
        else:
            Context.use_depth_model = False