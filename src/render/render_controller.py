import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger

from context import Context


def update_render(image_path, depth_path):
    logger.info("Loading image")
    image = open_image(image_path)

    h, w = image.shape[:2]
    image = cv2.resize(image, (w * Context.upscale, h * Context.upscale))
    
    up_image_width = image.shape[1]
    up_image_height = image.shape[0]
    logger.info(f'Image size: {up_image_width}x{up_image_height} was loaded')
    
    points, colors = create_pointcloud(image)
    points -= np.array([up_image_width / 2, up_image_height / 2, 0])
    points += np.random.rand(*points.shape) * 0.00001
    
    if not os.path.exists(depth_path):
        # generate_depthmap(image, depth_path)
        generate_depthmap_adabins(image, depth_path)
    
    logger.info("Loading depthmap")
    depth = open_image(depth_path)
    depth = cv2.resize(depth, (up_image_width, up_image_height))

    depth = depth[..., 0].astype(np.float32)
    data_depth = (depth - depth.min()) / (depth.max() - depth.min()) * Context.depthscale
    points_depth = data_depth.reshape(-1, 1)
    
    print(f'img.shape: {image.shape}')
    print(f'depth.shape: {depth.shape}')
    
    points[:, 2] = points_depth[:, 0]

    h, w = image.shape[:2]
    image = cv2.resize(image, (Context.image_width, Context.image_height))
    depth = cv2.resize(depth, (Context.image_width, Context.image_height))
    
    render = Render(Context.image_width, Context.image_height, Context.canvas_width, Context.canvas_height, focal_length=Context.focal_length, device="cuda")
    render.points = points
    render.colors = colors
    
    render.camera.set_position(0, 0, - up_image_height)
    render.camera.radius = up_image_height
    render.camera.alpha = 0
    render.camera.beta = - np.pi / 2
    
    Context.render = render
    
    image, depth = Context.render.render()
    
    Context.render_image_idx += 1
    
    return image, depth, points, colors

def restart_render():
    (Path(Context.log_folder) / "render").mkdir(exist_ok=True)
    render_save_path = Path(Context.log_folder) / "render" / ('render_' + str(Context.render_image_idx).zfill(5) + '.png')
    depth_save_path = Path(Context.log_folder) / "render" / ('depth_' + str(Context.render_image_idx).zfill(5) + '.png')
    
    if Context.inpainted_image is not None:
        Image.fromarray(Context.inpainted_image).save(str(render_save_path))
    elif Context.rendered_image is not None:
        logger.info('Inpainted image is None, step back to rendered image')
        Image.fromarray(Context.rendered_image).save(str(render_save_path))
    else:
        logger.warning("Inpainting and Rendered image is None")
        show_info('Save Error', 'Inpainting and Rendered image is None')

    # init_render(str(render_save_path), str(depth_save_path))
    update_render(str(render_save_path), str(depth_save_path))
    
    update_render_view()
    

def restart_render_with_current_image_callback(sender, app_data):
    restart_render()