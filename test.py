import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np


class ImagePanel:
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
        

class ViewTriPanel:
    
    def __init__(self) -> None:
        with dpg.window(tag='stack_panel'):
            with dpg.group(horizontal=True):
                self.p1 = ImagePanel(512, 512)
                self.p2 = ImagePanel(512, 512)
                self.p3 = ImagePanel(512, 512)


class ViewQuadPanel:
    
    def __init__(self) -> None:
        pass


def dpg_get_value(tag):
    value = None
    if dpg.does_item_exist(tag):
        value = dpg.get_value(tag)
    return value

def fit_image(img, dimensions):
    img_dim = np.flip(img.shape[:-1])
    scale = 1
    if (dimensions[0] <= dimensions[1]):
        scale = dimensions[0]/img_dim[0]
    else: scale = dimensions[1]/img_dim[1]
    img_dim[0]*=scale
    img_dim[1]*=scale
    return cv.resize(img, img_dim)  
    
def main():
    dpg.create_context()
    
    _img = cv.imread('./test.jpg')
    win_dimensions = [_img.shape[0], _img.shape[1]]
    dpg.create_viewport(title='img gui', width=win_dimensions[1], height=win_dimensions[0])
    
    # _imgdata = flat_img(_img)
    with dpg.window(tag='image_window'):
        img_panel = ImagePanel(win_dimensions[1], win_dimensions[0])
        img_panel.set_image(_img)

    def resize_callback():
        scale = int(dpg_get_value('downscale_combo'))
        h, w = _img.shape[:2]
        new_h, new_w = h // scale, w // scale
        img_panel.change_size(new_w=new_w, new_h=new_h)
        
        new_img = cv.resize(_img, (new_w, new_h))
        # TODO: add autoscale for img panel inner resolution to avoid size mismatch error
        img_panel.set_image(new_img)
        print('TESTETSET')
    
    with dpg.window(tag="ctlwindow", label="", no_close=True, min_size=(200,250)):
        with dpg.collapsing_header(label="gaussian_blur", tag="gmenu", default_open=True):
            sizes = [1, 2, 4]
            dpg.add_combo(sizes, default_value=sizes[0], tag='downscale_combo')
            dpg.add_button(label='apply', tag='apply_resize_button', callback=resize_callback)

    pp1 = ViewTriPanel()
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    # dpg.set_viewport_resize_callback(viewport_resize_cb)
    dpg.start_dearpygui()
    dpg.destroy_context()
    
if __name__ == "__main__":
    main()