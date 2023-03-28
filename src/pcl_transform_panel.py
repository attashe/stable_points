import torch
import numpy as np
import dearpygui.dearpygui as dpg

from dataclasses import dataclass
from loguru import logger

from context import Context
from render_panel import update_render_view


# TODO: Add chaos_fraction parameter
def transform_pointcloud_with_vortex(pointcloud, xc, yc, zc, vortex_strength, chaos=0, rotate_axis=1, chaos_fraction=1.0):
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
        
        chaos_nonzero = np.random.rand(chaos_x) > chaos_fraction
        
        transformed_radius += chaos_x * chaos_nonzero
        transformed_angle += chaos_y * chaos_nonzero
    
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

def transform_pointcloud_with_vortex_torch(pointcloud, xc, yc, zc, vortex_strength,
                                           chaos=0, rotate_axis=1, chaos_fraction=1.0, device='cpu'):
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
    transformed_radius = radius  # + vortex_strength * centered_pointcloud[:, 2]
    transformed_angle = angle + vortex_strength  # * radius
    
    # Add chaos to the transformation
    if chaos != 0:
        chaos_x = torch.rand(pointcloud.shape[0], device=pointcloud.device) * (2*chaos) - chaos
        chaos_y = torch.rand(pointcloud.shape[0], device=pointcloud.device) * (2*chaos) - chaos
        
        chaos_nonzero = torch.rand_like(chaos_x) < chaos_fraction
        
        transformed_radius += chaos_x * chaos_nonzero
        transformed_angle += chaos_y * chaos_nonzero
    
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
    Context.chaos_strength = dpg.get_value(sender)

def chaos_frac_slider_callback(sender):
    Context.chaos_fraction = dpg.get_value(sender)


class PCLTransformPanel:
    
    def __init__(self) -> None:
        Context.gravity_force = 0.1
        Context.vortex_strength = 1.0
        Context.chaos_strength = 0.0
        Context.chaos_fraction = 1.0
        
        self.use_torch = True
        
        with dpg.collapsing_header(label='Pointcloud transformation'):
            dpg.add_text(label='Gravity transform')
            dpg.add_slider_float(label='Gravity force', default_value=Context.gravity_force, min_value=-1.0, max_value=1.0,
                                 tag='gravity_force_slider', callback=gravity_slider_callback)
            dpg.add_button(label='Run', callback=self.gravity_transform)
            
            dpg.add_text(label='Vortex transform')
            dpg.add_slider_float(label='Vortex strength', default_value=Context.vortex_strength, min_value=0.0, max_value=5.0,
                                 tag='vortex_strength_slider', callback=vortex_slider_callback)
            dpg.add_slider_float(label='Chaos force', default_value=Context.chaos_strength, min_value=0.0, max_value=10.0,
                                 tag='chaos_slider', callback=chaos_slider_callback)
            dpg.add_slider_float(label='Chaos %', default_value=Context.chaos_fraction, min_value=0.0, max_value=1.0,
                                 tag='chaos_frac_slider', callback=chaos_frac_slider_callback)
            dpg.add_combo(['x', 'y', 'z'], default_value='y', label='Rotation axis', tag='axis_selector')
            dpg.add_button(label='Run', callback=self.vortex_transform)
            
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
        chaos_strength = Context.chaos_strength
        chaos_fraction = Context.chaos_fraction
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis = axis_map[dpg.get_value('axis_selector')]
        
        logger.debug(f'{Context.render.points.sum()}')
        logger.debug(f'{vortex_strength=}')
        logger.debug(f'{chaos_strength=}')
        
        # numpy version
        if self.use_torch:
            Context.render.points = transform_pointcloud_with_vortex_torch(pointcloud, center[0], center[1], center[2],
                                                                           vortex_strength=vortex_strength, chaos=chaos_strength,
                                                                           rotate_axis=axis, chaos_fraction=chaos_fraction,
                                                                           device='cuda')
        else:
            Context.render.points = transform_pointcloud_with_vortex(pointcloud, center[0], center[1], center[2],
                                                                    vortex_strength=vortex_strength, chaos=chaos_strength,
                                                                    rotate_axis=axis)
        logger.debug(f'{Context.render.points.sum()}')
        update_render_view()
    


def main():
    pass


if __name__ == "__main__":
    main()