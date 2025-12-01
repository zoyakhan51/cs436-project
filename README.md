# CS436 Project: 3D object reconstruction using 2d Images

## Overview

This repository reconstructs a sparse 3D point cloud from multiple images (n = 69) using feature matching, essential matrix estimation, triangulation, and incremental PnP-based pose estimation. The current implementation is divided into two phases according to our course requirements:

### Phase 1 — Two-View Reconstruction
- Load and pre-process images  
- Construct intrinsic matrix K  
- Detect SIFT keypoints and descriptors  
- Match features using Lowe’s ratio test  
- Estimate the Essential Matrix using RANSAC  
- Recover relative pose (R, t)  
- Triangulate 3D points for the correct pose  
- Export the sparse 3D point cloud  

### Phase 2 — Incremental SfM
- Extract features for each new image  
- Match new features to existing 3D points  
- Estimate camera pose using PnP + RANSAC  
- Triangulate new point tracks  
- Expand the global 3D map  
- (Optional) Run Bundle Adjustment for refinement  


## Code Organization

The repository is organized into modular Python files:

- **SFM.py** — main SfM reconstruction class (camera storage, 3D points, observations, triangulation helpers).  
  *Module:* `SFM`

- **helper_fns_p1.py** — image loading, SIFT feature extraction, feature matching, triangulation utilities.  
  *Module:* `helper_fns_p1`

- **phase_1.ipynb** — notebook demonstrating the complete two-view reconstruction pipeline.

- **phase_2_final.ipynb** — notebook implementing incremental SfM for multiple images.

## Set up

### Install Required Packages
Ensure you have Python 3.8+ installed. Then install the dependencies. After installing the dependencies, simply run the files for each phase:

```bash
pip install opencv-python numpy matplotlib


