import cv2
import pandas as py
import numpy as np 
import matplotlib.pyplot as plt

#HELPER_FN 1: to convert and load images to grayscale
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

#HELPER_FN 2: to construct instrinsic matrix 
def construct_intrinsic_matrix(image_path):
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    fx = fy = float(W)  #assuming focal length is equal to the image width
    cx, cy = W / 2.0, H / 2.0  #cx, cy -> principal / midpoints

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    return K, (W, H)

#HELPER_FN 3: computes keypoints and descriptors using SIFT
def detect_and_describe(gray, n_features=8000): 

    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, desc = sift.detectAndCompute(gray, None)
    return kps, desc

#HELPER_FN 4: returns good feature matches using brute force matcher + lowe's ratio test
def match_features(desc1, desc2, ratio=0.85):  

    if desc1 is None or desc2 is None:  #matching impossible
        return []
    
    bf = cv2.BFMatcher()
    matches_knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m_n in matches_knn:
        if len(m_n) == 2:
            m, n = m_n
            #impleementing lowe's ratio test -> 
            if m.distance < ratio * n.distance:
                good.append(m)
    return good

#HELPER_FN 5: triangulation -> returns 3d coords of pts 1 (2d points in img 1), and pts 2(2d points in img 2)
def triangulate_points(P1, P2, pts1, pts2):

    if len(pts1) == 0:
        return np.array([])
    
    pts1_t = pts1.T
    pts2_t = pts2.T

    points4D = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t) #homogenous coords
    points_3d = (points4D[:3] / points4D[3]).T  #conv back into euclidean coords
    
    return points_3d

