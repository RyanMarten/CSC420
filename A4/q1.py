import numpy as np 
import matplotlib.pyplot as plt
import math
import open3d as o3d
import os.path
from os import path

"""# Part I: Projection

## Q1) 3-D Point Cloud
"""

# read in the images and the depth maps
rgb_im_1 = plt.imread("Part1/1/rgbImage.jpg")
rgb_im_2 = plt.imread("Part1/2/rgbImage.jpg")
rgb_im_3 = plt.imread("Part1/3/rgbImage.jpg")
depth_im_1 = plt.imread("Part1/1/depthImage.png")
depth_im_2 = plt.imread("Part1/2/depthImage.png")
depth_im_3 = plt.imread("Part1/3/depthImage.png")

# Read in the instrinsic and extrinsic parameters
ext_im_1 = np.loadtxt("Part1/1/extrinsic.txt")
ext_im_2 = np.loadtxt("Part1/2/extrinsic.txt")
ext_im_3 = np.loadtxt("Part1/3/extrinsic.txt")

int_im_1 = np.loadtxt("Part1/1/intrinsics.txt")
int_im_2 = np.loadtxt("Part1/2/intrinsics.txt")
int_im_3 = np.loadtxt("Part1/3/intrinsics.txt")

print("Extrinsic Camera Parameters: \n", ext_im_1)
print("\nInstrinsic Camera Parameters: \n", int_im_2)

# changing things up 
## THIS IS SO I CAN SEE THE RESULTS ON THE OTHER IMAGES!!!!
rgb_im_1 = rgb_im_3
depth_im_1 = depth_im_3
ext_im_1 = ext_im_3
int_im_1 = int_im_3

def compute_point_cloud_camera(rgb_im, depth_im, x, y, k):
  """ Computes camera coordinates of pixels from the depth map provided 

  Parameters: 
    - rgb_im (RGB image)
    - depth_im (Depth map image)
    - k (camera calibration matrix)
    - x, y (pixel location)

  Returns: (X,Y,Z,R,G,B)
  """
  q = np.array([x, y, 1]) * depth_im[y,x]
  Q = np.matmul(np.linalg.inv(k), q)
  
  x_n, y_n, z_n = Q
  r_n, g_n, b_n = rgb_im[y,x]
  return x_n, y_n, z_n, r_n, g_n, b_n

xyz = []
rgb = []

testing = True

if not path.exists("im1_camera.xyzrgb") or testing:
  print("Creating Camera XYZRGB file")
  with open("im1_camera.xyzrgb", "w") as f: 
    for i in range(rgb_im_1.shape[0]):
      for j in range(rgb_im_1.shape[1]):
        x,y,z,r,g,b = compute_point_cloud_camera(rgb_im_1, depth_im_1, j, i, k=int_im_1)
        r /= 255
        g /= 255
        b /= 255
        xyz.append([x,y,z])
        rgb.append([r,g,b])
        f.write(f"{x} {y} {z} {r} {g} {b}\n")

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("im1_camera.xyzrgb", format="xyzrgb")
print(pcd)
print(np.asarray(pcd.points))

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

# Downsample Visualization (Just for fun)
# downpcd = pcd.voxel_down_sample(voxel_size=0.005)
# o3d.visualization.draw_geometries([downpcd])

ext_im_1_real = np.zeros((4,4))
ext_im_1_real[:3, :] = ext_im_1
ext_im_1_real[3,3] = 1
print("Extrinsic Camera Parameters: \n", ext_im_1_real)

def compute_point_cloud_world(rgb_im, depth_im, x, y, k, ext):
    """ Computes world coordinates of pixels from the depth map provided 

    Parameters: 
      - rgb_im (RGB image)
      - depth_im (Depth map image)
      - k (camera calibration matrix)
      - ext (extrinsic parameters)
      - x, y (pixel location)

    Returns: (X,Y,Z,R,G,B)
    """
    # Inverse camera projection from image plane to camera coodinates
    q = np.array([x, y, 1]) * depth_im[y, x]
    c = np.matmul(np.linalg.inv(k), q)

    # Conversion from camera coordinates to world coordinates
    c_h = np.zeros((4,1))
    c_h[3] = 1
    c_h[:3] = np.reshape(c, (3,1))

    W =  np.matmul(np.linalg.inv(ext), c_h)

    x_n = W[0][0]
    y_n = W[1][0]
    z_n = W[2][0]
    h = W[3][0]

    # h is always 1
    # x_n /= h
    # y_n /= h
    # z_n /= h
    r_n, g_n, b_n = rgb_im[y, x]
    return x_n, y_n, z_n, r_n, g_n, b_n

xyz = []
rgb = []

testing = True

if not path.exists("im1_world.xyzrgb") or testing:
  print("Creating World XYZRGB file")
  with open("im1_world.xyzrgb", "w") as f: 
    for i in range(rgb_im_1.shape[0]):
      for j in range(rgb_im_1.shape[1]):
        x,y,z,r,g,b = compute_point_cloud_world(rgb_im_1, depth_im_1, j, i, k=int_im_1, ext=ext_im_1_real)
        r /= 255
        g /= 255
        b /= 255
        xyz.append([x,y,z])
        rgb.append([r,g,b])
        f.write(f"{x} {y} {z} {r} {g} {b}\n")

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("im1_world.xyzrgb", format="xyzrgb")
print(pcd)
print(np.asarray(pcd.points))

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])