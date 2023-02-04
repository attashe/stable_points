import numpy as np

from loguru import logger


def get_x_rotation_matrix(x):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(x), -np.sin(x), 0],
                     [0, np.sin(x), np.cos(x), 0],
                     [0, 0, 0, 1]])

def get_y_rotation_matrix(y):
    return np.array([[np.cos(y), 0, np.sin(y), 0],
                     [0, 1, 0, 0],
                     [-np.sin(y), 0, np.cos(y), 0],
                     [0, 0, 0, 1]])

def get_z_rotation_matrix(z):
    return np.array([[np.cos(z), -np.sin(z), 0, 0],
                     [np.sin(z), np.cos(z), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    
class PerspectiveCamera:
    
    def __init__(self, image_width, image_height, canvas_width, canvas_height, focal_length):
        self.image_width = image_width
        self.image_height = image_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.focal_length = focal_length
        
        self.x_translate = 0.0
        self.y_translate = 0.0
        self.z_translate = 0.0
        self.x_rotate = 0.0
        self.y_rotate = 0.0
        self.z_rotate = 0.0
        
        # self.world_to_camera = self.compute_camera_matrix()
        
        self.image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.depth = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        
        self.points = None
        self.colors = None
        
    def set_position(self, x_translate, y_translate, z_translate):
        self.x_translate = x_translate
        self.y_translate = y_translate
        self.z_translate = z_translate
        
        self.world_to_camera = self.compute_camera_matrix()
        
    def set_rotation(self, x_rotate, y_rotate, z_rotate):
        self.x_rotate = x_rotate
        self.y_rotate = y_rotate
        self.z_rotate = z_rotate
        
        self.world_to_camera = self.compute_camera_matrix()
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.world_to_camera = self.compute_camera_matrix()

    def compute_camera_matrix(self):
        cameraToWorld = np.eye(4)

        # Get the rotation matrices
        R_x = get_x_rotation_matrix(self.x_rotate)
        R_y = get_y_rotation_matrix(self.y_rotate)
        R_z = get_z_rotation_matrix(self.z_rotate)

        # Translation matrix (camera to world)
        # TODO: Fix this matrix to standard form with far and near plane clipping
        # and with the correct focal length
        # T = np.array([
        #     [1, 0, 0, self.x_translate],
        #     [0, 1, 0, self.y_translate],
        #     [0, 0, 1 / self.focal_length, self.z_translate],
        #     [0, 0, 0, 1]])
        T = np.array([
            [self.focal_length, 0, 0, self.x_translate],
            [0, self.focal_length, 0, self.y_translate],
            [0, 0, 1, self.z_translate],
            [0, 0, 0, 1]])

        # First rotate, then translate
        cameraToWorld = T @ R_z @ R_y @ R_x

        # Invert the matrix to get world to camera matrix
        worldToCamera = np.linalg.inv(cameraToWorld)  # cameraToWorld.inverse(); 

        logger.debug(f'CameraToWorld: {cameraToWorld}')
        logger.debug(f'World to Camera: {worldToCamera}')

        return worldToCamera
    
    
class ArcBallCamera(PerspectiveCamera):
    
    def __init__(self, image_width, image_height, canvas_width, canvas_height, focal_length, alpha=0.0, beta=0.0, theta=0.0, radius=1.0, center=(0.0, 0.0, 0.0)):
        
        super().__init__(image_width, image_height, canvas_width, canvas_height, focal_length)
        
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.radius = radius
        
        self.center = np.array(center, dtype=np.float32)
        
        self.update_position_rotation()
        self.world_to_camera = self.compute_camera_matrix()
        
    def update_camera(self, alpha, beta, theta, radius):
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.radius = radius
        
        self.update_position_rotation()
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.update_position_rotation()
        
    def update_position_rotation(self):
        self.x_translate = self.center[0] + self.radius * np.sin(self.alpha) * np.sin(self.beta)
        self.y_translate = self.center[1] + self.radius * np.cos(self.beta)
        self.z_translate = self.center[2] + self.radius * np.cos(self.alpha) * np.sin(self.beta)
        
        self.x_rotate = (self.beta - np.pi / 2)
        self.y_rotate = self.alpha
        self.z_rotate = self.theta
        
        # self.world_to_camera = self.compute_camera_matrix()
        
    def compute_camera_matrix(self):
        self.update_position_rotation()
        logger.debug(f'camera [a, b, t, r] = [{self.alpha}, {self.beta}, {self.theta}, {self.radius}]')
        logger.debug(f'translate: [{self.x_translate}, {self.y_translate}, {self.z_translate}]')
        logger.debug(f'rotate: [{self.x_rotate}, {self.y_rotate}, {self.z_rotate}]')
        self.world_to_camera = super().compute_camera_matrix()
        
        return self.world_to_camera