import numpy as np 
import matplotlib.pyplot as plt
import math
import open3d as o3d
import os.path
from os import path

"""# Part I: Projection

## Q2) Rotation in Three Axis
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


print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("im1_camera.xyzrgb", format="xyzrgb")
print(pcd)
print(np.asarray(pcd.points).shape)

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def camera_coordinate_rotation(xyzrgb_filename, rgb_im, radians, axis):
    """ Rotates Camera Coordinate 3D PointCloud for given radians around specific axis
    and projects it to the image plane

    Parameters: 
    - radians (rotation for each step in radians)
    - axis    ('x', 'y', 'z')

    Returns: 
    - depth (image array of project depth values)
    - im    (image array of projected RGB values)
    """
    depth = np.zeros(rgb_im.shape[:2])
    im = np.zeros_like(rgb_im)

def rotation_around_axis(x, y, z, omega, t, axis):
    """ Rotates 3D Scene Point omega*t radians around specific axis
    [X', Y', Z'] = R[X, Y, Z], Where R is the rotation matrix

    Parameters: 
    - omega (rotation for each step in radians)
    - t (time steps)
    - axis (x=0, y=1, z=3)

    Returns: rotated points (X',Y',Z')
    """


def projection_to_image_plane():
    pass

o3d.visualization.draw_geometries([pcd])