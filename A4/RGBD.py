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
depth_im_1 = plt.imread("Part1/1/depthImage.png")

# Read in the instrinsic and extrinsic parameters
ext_im_1 = np.loadtxt("Part1/1/extrinsic.txt")
int_im_1 = np.loadtxt("Part1/1/intrinsics.txt")

color_raw = o3d.io.read_image("Part1/1/rgbImage.jpg")
depth_raw = o3d.io.read_image("Part1/1/depthImage.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
    color_raw, depth_raw)
print(rgbd_image)

plt.subplot(1, 2, 1)
plt.title('grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('depth image')
plt.imshow(rgbd_image.depth)
plt.show()

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])