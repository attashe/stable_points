import dearpygui.dearpygui as dpg
import numpy as np


class ImagePanel:
    """Based on code
    https://github.com/DataExplorerUser/CV_example/blob/main/dearpygui_cv
    """
    _n = 0
    
    def __init__(self, width, height) -> None:
        ImagePanel._n += 1
        self.width = width
        self.height = height
        self.image_data = np.zeros((height, width, 3), dtype=np.float32)
        self.textag = f"tex_tag_{ImagePanel._n}"
        self.wintag = f"img_window_{ImagePanel._n}"
        
        with dpg.texture_registry(show=False):  
            dpg.add_raw_texture(height=height, width=width, 
                                default_value=self.image_data,
                                format=dpg.mvFormat_Float_rgb,
                                tag=self.textag)
 
        
        with dpg.group(tag=self.wintag):
            dpg.add_image(self.textag, parent=self.wintag)
            
    def set_image(self, image: np.ndarray) -> None:
        np.copyto(self.image_data, image.astype(np.float32) / 255)

    def update_data(self, data):
        assert self.image_data.shape == data.shape
        np.copyto(self.image_data, data)
    
    def _reload_data(self):
        # img = fit_image(mat, dimensions)
        # imgdata = flat_img(img)
        # delete texture/image, re-add
        dpg.delete_item(self.wintag, children_only=True)
        dpg.delete_item(self.textag)
        self.image_data = np.zeros((self.height, self.width, 3), dtype=np.float32)
        with dpg.texture_registry(show=False):      
            dpg.add_raw_texture(height=self.height, width=self.width,
                                default_value=self.image_data,
                                format=dpg.mvFormat_Float_rgb,
                                tag=self.textag)
            dpg.add_image(self.textag, parent=self.wintag)
        
    def change_size(self, new_w, new_h):
        self.width = new_w
        self.height = new_h
        
        self._reload_data()
