
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#defining SFM reconstruction class 

class SfMReconstruction:
    def __init__(self, K):
        self.K = K  #intrinsic matrix 
        self.cameras = {}
        self.points_3d = []  #array of 3D points 
        self.point_colors = []   #optional - stors RGB colors for visualization of the 3D point cloud
        self.observations = {}   #map of (point_idx, cam_idx) -> 2D projection
        self.image_features = {} #stores keypoints / descriptors
        self.point_to_feature = {}  #optional - maps 3D point to which feature it came from
        
        self.feature_to_point = defaultdict(dict)  #optional - reverse maps feature to which 3D point it came from
    
    
    def add_camera(self, idx, R, t):  #builds projection matrix P = K[R|t], saves R, t, and P -> essential for triangulation

        P = self.K @ np.hstack((R, t))
        self.cameras[idx] = {'R': R, 't': t, 'P': P}
        
    def add_point(self, point_3d, color=None):  #appends a new 3D point, adds color for visualization
        idx = len(self.points_3d)       
        self.points_3d.append(point_3d)
        self.point_colors.append(color if color else [128, 128, 128])  
        self.point_to_feature[idx] = {}  #creates empty dict for feature mapping 
        return idx
    
    def add_observation(self, point_idx, cam_idx, pt_2d, feature_idx=None):
        self.observations[(point_idx, cam_idx)] = pt_2d

        if feature_idx is not None:
            self.point_to_feature[point_idx][cam_idx] = feature_idx
            self.feature_to_point[cam_idx][feature_idx] = point_idx
        
    #to help with constructing the point cloud

    def get_points_array(self):  
        if len(self.points_3d) == 0:
            return np.array([]).reshape(0, 3)
        return np.array(self.points_3d)
    
    def get_colors_array(self): 
        if len(self.point_colors) == 0:
            return np.array([]).reshape(0, 3)
        return np.array(self.point_colors)
