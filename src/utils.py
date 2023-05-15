#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from pathlib import Path
from typing import Union
from collections import namedtuple

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image, ImageOps
from sklearn.neighbors import KDTree
from loguru import logger

from skimage import exposure
from blendmodes.blend import blendLayers, BlendType

from context import Context


def setup_color_correction(image):
    logger.info("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    logger.info("Applying color correction.")
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image


def clear_pointcloud(points, threshold, radius):
    # Create a KD tree from the points
    tree = KDTree(points)

    # Query the tree to find the neighbors for each point
    num_neighbors = tree.query_radius(points, r=radius, count_only=True)

    # Find points with fewer neighbors than threshold
    indices = np.where(num_neighbors < threshold)[0]
    keep_indices = np.setdiff1d(np.arange(len(points)), indices)

    # Return the points with enough neighbors
    return points[keep_indices], keep_indices


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


def resize_padding_pil(img, size):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    img = img.resize(new_size, Image.Resampling.LANCZOS)

    delta_w = size - new_size[0]
    delta_h = size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_img = ImageOps.expand(img, padding)
    
    return new_img


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


def convert_from_uvd_numpy(points, depth, focal_length):
    # d = 1
    focal_length = Context.upscale
    
    d = (depth[:, 0] + Context.near) * Context.depthscale
    # d = depth[:, 0]
    # logger.info(f'depth: min: {d.min()} max: {d.max()}')
    cx = Context.render.camera.center[0]
    cy = Context.render.camera.center[1]
    x_over_z = - (cx - points[:, 0]) / (focal_length * Context.render.image_width)
    y_over_z = - (cy - points[:, 1]) / (focal_length * Context.render.image_width)
    logger.debug(f'cx: {cx}, cy: {cy}, focal_length: {focal_length}')
    logger.debug(f'canvas_width: {Context.render.canvas_width}, canvas_height: {Context.render.canvas_height}')
    logger.debug(f'x_over_z: {x_over_z.max()}, y_over_z: {y_over_z.max()}')
    logger.debug(f'x_over_z.shape: {x_over_z.shape}, y_over_z.shape: {y_over_z.shape}, d: {d}')
    # stop
    z = d / np.sqrt(1. + x_over_z*x_over_z + y_over_z*y_over_z)
    x = x_over_z * z
    y = y_over_z * z
    
    z = z #* Context.render.image_height / 2
    
    points = np.stack((x, y, z), axis=1)
    return points


def save_pil_image(path: Path, pil_image: Image.Image):
    path.parent.mkdir(exist_ok=True, parents=True)
    pil_image.save(str(path))
    
# Automatic1111 Tile grid code
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/22bcc7be428c94e9408f589966c2040187245d81/modules/images.py#L60
Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image
