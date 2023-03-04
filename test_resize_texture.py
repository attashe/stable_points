# Code provided by shebbbb on Reddit, combining CV and Dear PyGui and resizing images.
# https://www.reddit.com/r/DearPyGui/comments/t0xy8b/updating_a_texture_with_new_dimensions/hysqxf9/?context=3

import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
 
def flat_img(mat):
    return np.true_divide(np.asfarray(np.ravel(np.flip(mat,2)), dtype='f'), 255.0)
 
#full size image
_img = cv.imread('./test.jpg')
_imgdata = flat_img(_img)
#temp copy
_timg = _img.copy()
 
win_dimensions = [600, 600]
gbool = False
hbool = False
 
def fit_image(img, dimensions):
    img_dim = np.flip(img.shape[:-1])
    scale = 1
    if (dimensions[0] <= dimensions[1]):
        scale = dimensions[0]/img_dim[0]
    else: scale = dimensions[1]/img_dim[1]
    img_dim[0]*=scale
    img_dim[1]*=scale
    return cv.resize(img, img_dim)  
 
# update texture with new dimension using cv2
# configure_item on add_image to scale causes crashes
def resize_window_img(wintag, textag, dimensions, mat):
    img = fit_image(mat, dimensions)
    imgdata = flat_img(img)
    # delete texture/image, re-add
    dpg.delete_item(wintag, children_only=True)
    dpg.delete_item(textag)
    with dpg.texture_registry(show=False):      
        dpg.add_raw_texture(img.shape[1], img.shape[0], imgdata, tag=textag, format=dpg.mvFormat_Float_rgb)
        dpg.add_image(textag, parent=wintag)
 
def update_preview(mat):
    img = fit_image(mat, win_dimensions)    
    imgdata = flat_img(img)
    dpg.set_value("tex_tag", imgdata)
 
def gaussian(img, k, s):
    k = int(k)
    k = k if (k%2 != 0) else k+1    
    return cv.GaussianBlur(img, (k,k), s, 0, cv.BORDER_DEFAULT)
 
def shift_hue(mat, val):
    hsv = cv.cvtColor(mat, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    shift_h = (h+int(val*180))%255
    shift_hsv = cv.merge([shift_h, s, v])
    return cv.cvtColor(shift_hsv, cv.COLOR_HSV2BGR)
 
def handle_edit(tag):
    global _img, _timg
    mat = _img.copy()
    if(gbool):
        mat = gaussian(mat, dpg.get_value("gbar_k"), dpg.get_value("gbar_s"))
    if(hbool):
        mat = shift_hue(mat, dpg.get_value("hbar")) 
 
    _timg = mat
    update_preview(mat) 
 
def afteredit_cb(sender, data):
    handle_edit(data)
 
def box_cb(sender, data):
    global gbool, hbool, dbool
    if(sender == "gbox"):
        gbool = data
    elif(sender == "hbox"):
        hbool = data
    elif(sender == "dbox"):
        dbool = data
    handle_edit(sender)
 
def viewport_resize_cb(sender, data):
    win_dimensions[0] = data[2:][0]
    win_dimensions[1] = data[2:][1]
    resize_window_img("img_window", "tex_tag", win_dimensions, _timg)
 
dpg.create_context()
dpg.create_viewport(title='img gui', width=win_dimensions[0], height=win_dimensions[1])
 
with dpg.item_handler_registry(tag="float handler") as handler:
    dpg.add_item_deactivated_after_edit_handler(callback=afteredit_cb)
 
with dpg.texture_registry(show=False):  
    dpg.add_raw_texture(_img.shape[1], _img.shape[0], _imgdata, tag="tex_tag", format=dpg.mvFormat_Float_rgb)
 
with dpg.window(tag="img_window"):
    dpg.add_image("tex_tag")
    dpg.set_primary_window("img_window", True)
 
with dpg.window(tag="ctlwindow", label="", no_close=True, min_size=(200,250)):
    with dpg.collapsing_header(label="gaussian_blur", tag="gmenu", default_open=True):
        dpg.add_checkbox(label="on", tag="gbox", callback=box_cb)
        dpg.add_slider_float(label="ksize", tag="gbar_k", default_value=0.,  max_value=21)
        dpg.add_slider_float(label="sigma", tag="gbar_s", default_value=0.,  max_value=6)
    with dpg.collapsing_header(label="hue", tag="hmenu", default_open=True):
        dpg.add_checkbox(label="on", tag="hbox", callback=box_cb)
        dpg.add_slider_float(label="shift", tag="hbar", default_value=0., max_value=1)
 
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.bind_item_handler_registry("gbar_k", "float handler")
dpg.bind_item_handler_registry("gbar_s", "float handler")
dpg.bind_item_handler_registry("hbar", "float handler")
dpg.set_viewport_resize_callback(viewport_resize_cb)
dpg.start_dearpygui()
dpg.destroy_context()