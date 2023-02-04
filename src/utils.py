#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from typing import Union
from pathlib import Path


def convert_cv_to_dpg(image, width, height):
    resize_image = cv2.resize(image, (width, height))

    data = np.flip(resize_image, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')

    texture_data = np.true_divide(data, 255.0)

    return texture_data


def check_camera_connection(max_device_count=4, is_debug=False):
    device_no_list = []

    for device_no in range(0, max_device_count):
        if is_debug:
            print('Check Device No:' + str(device_no).zfill(2), end='')

        cap = cv2.VideoCapture(device_no)
        ret, _ = cap.read()
        if ret:
            device_no_list.append(device_no)
            if is_debug:
                print(' -> Find')
        else:
            if is_debug:
                print(' -> None')

    return device_no_list


def dpg_set_value(tag, value):
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, value)


def dpg_get_value(tag):
    value = None
    if dpg.does_item_exist(tag):
        value = dpg.get_value(tag)
    return value


def open_image(img_file: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def create_pointcloud(image):
    h, w = image.shape[:2]
    
    # Change the order of the points from (y, x) to be (x, y)
    # points = np.array(np.meshgrid(np.arange(w), np.arange(h))).swapaxes(0, 2).swapaxes(0, 1).reshape(-1, 2)
    points = np.array(np.meshgrid(np.arange(h), np.arange(w))).swapaxes(0, 2).reshape(-1, 2)
    points = points[:, [1, 0]]
    z_values = np.zeros((h * w, 1))
    points = np.concatenate((points, z_values), axis=1)
    colors = image.reshape(-1, 3)
    
    return points, colors


def on_selection(sender, unused, user_data):

    if user_data[1]:
        print("User selected 'Ok'")
    else:
        print("User selected 'Cancel'")

    # delete window
    dpg.delete_item(user_data[0])


def show_info(title, message, selection_callback=on_selection):

    # guarantee these commands happen in the same frame
    with dpg.mutex():

        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        with dpg.window(label=title, modal=True, no_close=True) as modal_id:
            dpg.add_text(message)
            dpg.add_button(label="Ok", width=75, user_data=(modal_id, True), callback=selection_callback)
            dpg.add_same_line()
            dpg.add_button(label="Cancel", width=75, user_data=(modal_id, False), callback=selection_callback)

    # guarantee these commands happen in another frame
    dpg.split_frame()
    width = dpg.get_item_width(modal_id)
    height = dpg.get_item_height(modal_id)
    dpg.set_item_pos(modal_id, [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2])
