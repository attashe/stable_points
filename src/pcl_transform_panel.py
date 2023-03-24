import numpy as np
import dearpygui.dearpygui as dpg

from dataclasses import dataclass
from loguru import logger

from context import Context
from render_panel import update_render_view


def transform_pointcloud_with_vortex(pointcloud, xc, yc, zc, vortex_strength, chaos=0, rotate_axis=1):
    # Shift point cloud to center at (xc, yc, zc)
    centered_pointcloud = pointcloud - np.array([xc, yc, zc])
    
    # Calculate radius and angle of each point in point cloud
    if rotate_axis == 1:
        pass
    elif rotate_axis == 2:
        centered_pointcloud = np.roll(centered_pointcloud, 1, axis=1)
    elif rotate_axis == 0:
        centered_pointcloud = np.roll(centered_pointcloud, 2, axis=1)

    radius = np.linalg.norm(centered_pointcloud[:, :2], axis=1)
    angle = np.arctan2(centered_pointcloud[:, 1], centered_pointcloud[:, 0])
    
    # Apply vortex transformation to each point
    transformed_radius = radius + vortex_strength * centered_pointcloud[:, 2]
    transformed_angle = angle + vortex_strength * radius
    
    # Add chaos to the transformation
    if chaos != 0:
        chaos_x = np.random.uniform(-chaos, chaos, size=pointcloud.shape[0])
        chaos_y = np.random.uniform(-chaos, chaos, size=pointcloud.shape[0])
        transformed_radius += chaos_x
        transformed_angle += chaos_y
    
    transformed_z = centered_pointcloud[:, 2]
    
    # Convert back to Cartesian coordinates
    transformed_pointcloud = np.zeros_like(pointcloud)
    transformed_pointcloud[:, 0] = transformed_radius * np.cos(transformed_angle)
    transformed_pointcloud[:, 1] = transformed_radius * np.sin(transformed_angle)
    transformed_pointcloud[:, 2] = transformed_z
    
    if rotate_axis == 1:
        pass
    elif rotate_axis == 2:
        transformed_pointcloud = np.roll(transformed_pointcloud, -1, axis=1)
    elif rotate_axis == 0:
        transformed_pointcloud = np.roll(transformed_pointcloud, -2, axis=1)
    
    # Shift back to original position
    transformed_pointcloud += np.array([xc, yc, zc])
    
    return transformed_pointcloud


import torch

def transform_pointcloud_with_vortex_torch(pointcloud, xc, yc, zc, vortex_strength, chaos=0, rotate_axis=1, device='cpu'):
    # Convert point cloud to PyTorch tensor
    pointcloud = torch.tensor(pointcloud, device=device).float()
    center = torch.tensor([xc, yc, zc], device=pointcloud.device).float()
    # Shift point cloud to center at (xc, yc, zc)
    centered_pointcloud = pointcloud - center
    
    # Rotate point cloud if necessary
    if rotate_axis == 1:
        pass
    elif rotate_axis == 2:
        centered_pointcloud = torch.roll(centered_pointcloud, shifts=1, dims=1)
    elif rotate_axis == 0:
        centered_pointcloud = torch.roll(centered_pointcloud, shifts=2, dims=1)
    
    # Calculate radius and angle of each point in point cloud
    radius = torch.norm(centered_pointcloud[:, :2], dim=1)
    angle = torch.atan2(centered_pointcloud[:, 1], centered_pointcloud[:, 0])
    
    # Apply vortex transformation to each point
    transformed_radius = radius + vortex_strength * centered_pointcloud[:, 2]
    transformed_angle = angle + vortex_strength * radius
    
    # Add chaos to the transformation
    if chaos != 0:
        chaos_x = torch.rand(pointcloud.shape[0], device=pointcloud.device) * (2*chaos) - chaos
        chaos_y = torch.rand(pointcloud.shape[0], device=pointcloud.device) * (2*chaos) - chaos
        transformed_radius += chaos_x
        transformed_angle += chaos_y
    
    transformed_z = centered_pointcloud[:, 2]
    
    # Convert back to Cartesian coordinates
    transformed_pointcloud = torch.zeros_like(pointcloud)
    transformed_pointcloud[:, 0] = transformed_radius * torch.cos(transformed_angle)
    transformed_pointcloud[:, 1] = transformed_radius * torch.sin(transformed_angle)
    transformed_pointcloud[:, 2] = transformed_z
    
    # Rotate point cloud back to its original orientation
    if rotate_axis == 1:
        pass
    elif rotate_axis == 2:
        transformed_pointcloud = torch.roll(transformed_pointcloud, shifts=-1, dims=1)
    elif rotate_axis == 0:
        transformed_pointcloud = torch.roll(transformed_pointcloud, shifts=-2, dims=1)
    
    # Shift back to original position
    transformed_pointcloud += center
    
    # Convert back to NumPy array
    transformed_pointcloud = transformed_pointcloud.cpu().numpy()
    
    return transformed_pointcloud


def gravity_slider_callback(sender):
    Context.gravity_force = dpg.get_value(sender)
    
def vortex_slider_callback(sender):
    Context.vortex_strength = dpg.get_value(sender)
    
def chaos_slider_callback(sender):
    Context.chaos_coef = dpg.get_value(sender)


class PCLTransformPanel:
    
    def __init__(self) -> None:
        Context.gravity_force = 0.1
        Context.vortex_strength = 1.0
        Context.chaos_coef = 0.0
        
        self.use_torch = True
        
        with dpg.collapsing_header(label='Pointcloud transformation'):
            dpg.add_slider_float(label='Gravity force', default_value=Context.gravity_force, min_value=-1.0, max_value=1.0,
                                 tag='gravity_force_slider', callback=gravity_slider_callback)
            dpg.add_button(label='Gravity transform', callback=self.gravity_transform)
            
            dpg.add_slider_float(label='Vortex strength', default_value=Context.vortex_strength, min_value=0.0, max_value=5.0,
                                 tag='vortex_strength_slider', callback=vortex_slider_callback)
            dpg.add_slider_float(label='Chaos', default_value=Context.chaos_coef, min_value=0.0, max_value=1.0,
                                 tag='chaos_slider', callback=chaos_slider_callback)
            dpg.add_combo(['x', 'y', 'z'], default_value='y', label='Rotation axis', tag='axis_selector')
            dpg.add_button(label='Vortex transform', callback=self.vortex_transform)
            
    def gravity_transform(self, sender):
        pointcloud = Context.render.points.copy()
        center = pointcloud.mean(axis=0)
        logger.debug(f'{center=}')
        gravity_coef = 1.0 - Context.gravity_force + 1e-5
        
        logger.debug(f'{Context.render.points.sum()}')
        logger.debug(f'{gravity_coef=}')
        
        centered_pointcloud = pointcloud - center
        centered_pointcloud *= gravity_coef
        pointcloud = centered_pointcloud + center
        
        Context.render.points = pointcloud
        logger.debug(f'{Context.render.points.sum()}')
        update_render_view()
    
    def vortex_transform(self, sender):
        pointcloud = Context.render.points.copy()
        center = pointcloud.mean(axis=0)
        logger.debug(f'{center=}')
        vortex_strength = Context.vortex_strength / 100
        chaos_coef = Context.chaos_coef
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis = axis_map[dpg.get_value('axis_selector')]
        
        logger.debug(f'{Context.render.points.sum()}')
        logger.debug(f'{vortex_strength=}')
        logger.debug(f'{chaos_coef=}')
        
        # numpy version
        if self.use_torch:
            Context.render.points = transform_pointcloud_with_vortex_torch(pointcloud, center[0], center[1], center[2],
                                                                           vortex_strength=vortex_strength, chaos=chaos_coef,
                                                                           rotate_axis=axis, device='cuda')
        else:
            Context.render.points = transform_pointcloud_with_vortex(pointcloud, center[0], center[1], center[2],
                                                                    vortex_strength=vortex_strength, chaos=chaos_coef,
                                                                    rotate_axis=axis)
        logger.debug(f'{Context.render.points.sum()}')
        update_render_view()
    


def main():
    pass


if __name__ == "__main__":
    main()