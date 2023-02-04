import cv2
import numpy as np
import torch

from math import floor
from numba import njit

from .camera import PerspectiveCamera, ArcBallCamera


class Render:
    
    def __init__(self, image_width, image_height, canvas_width, canvas_height, focal_length, points=None, colors=None, device=None, camera=None):
        self.image_width = image_width
        self.image_height = image_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.focal_length = focal_length
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if camera is None:
            self.camera = ArcBallCamera(image_width, image_height, canvas_width, canvas_height, focal_length)
        
        # self.world_to_camera = self.compute_camera_matrix()
        
        self.image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.depth = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        
        self.points = None
        self.colors = None
        
        self.image_torch = torch.zeros((self.image_height, self.image_width, 3), dtype=torch.float32, device=self.device)
        self.depth_torch = torch.zeros((self.image_height, self.image_width), dtype=torch.float32, device=self.device)
        
    def update_camera_matrix(self):
        self.world_to_camera = self.compute_camera_matrix()
        
    def compute_camera_matrix(self):
        return self.camera.compute_camera_matrix()
    
    def set_image(self, image):
        np.copyto(self.image, image)
        
    def set_depth(self, depth):
        np.copyto(self.depth, depth)

    def set_points(self, points):
        self.points = points
    
    def render(self):
        self.world_to_camera = self.compute_camera_matrix()
        # Calculate pointcloud to camera with pytorch matrix multiplication
        points = torch.from_numpy(self.points).float().to(self.device)
        colors = torch.from_numpy(self.colors).byte().to(self.device)
        worldToCamera = torch.from_numpy(self.world_to_camera).float().to(self.device)
        points = torch.cat((points, torch.ones((len(points), 1), device=self.device)), dim=1)
        pCamera = torch.mm(points, worldToCamera.t())
        
        # Sort points and colors by distance to camera
        dist = torch.norm(pCamera[:, :3], dim=1)
        dist, idx = torch.sort(dist, descending=True)
        pCamera = pCamera[idx]
        colors = colors[idx]

        pScreen = torch.zeros((len(points), 2), dtype=torch.float32, device=self.device)
        pScreen[:, 0] = pCamera[:, 0] / -pCamera[:, 2]
        pScreen[:, 1] = pCamera[:, 1] / -pCamera[:, 2]

        pNDC = pScreen
        pNDC[:, 0] = (pScreen[:, 0] + self.canvas_width * 0.5) / self.canvas_width
        pNDC[:, 1] = (pScreen[:, 1] + self.canvas_height * 0.5) / self.canvas_height

        pRaster_x = (pNDC[:, 0] * self.image_width).long()
        pRaster_y = ((1 - pNDC[:, 1]) * self.image_height).long()

        # Get valid pixels
        valid = torch.logical_and(torch.logical_and(pRaster_x >= 0, pRaster_x < self.image_width),
                                    torch.logical_and(pRaster_y >= 0, pRaster_y < self.image_height))
        
        # Get depth
        cx, cy, cz = worldToCamera[0, 3], worldToCamera[1, 3], worldToCamera[2, 3]
        depth = torch.zeros((self.image_height, self.image_width), dtype=torch.float32, device=self.device)
        depth[pRaster_y[valid], pRaster_x[valid]] = (pCamera[valid, 0] - cx) * (pCamera[valid, 0] - cx) + \
                                                    (pCamera[valid, 1] - cy) * (pCamera[valid, 1] - cy) + \
                                                    (pCamera[valid, 2] - cz) * (pCamera[valid, 2] - cz)

        # Get color
        img = torch.zeros((self.image_height, self.image_width, 3), dtype=torch.uint8, device=self.device)
        img[pRaster_y[valid], pRaster_x[valid]] = colors[valid]

        return img.cpu().numpy(), depth.cpu().numpy()
