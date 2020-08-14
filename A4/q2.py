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


def compute_point_cloud_camera(rgb_im, depth_im, x, y, k):
    """ Computes camera coordinates of pixels from the depth map provided 

    Parameters: 
        - rgb_im (RGB image)
        - depth_im (Depth map image)
        - k (camera calibration matrix)
        - x, y (pixel location)

    Returns: (X,Y,Z,R,G,B)
    """
    q = np.array([x, y, 1]) * depth_im[y, x]
    Q = np.matmul(np.linalg.inv(k), q)

    x_n, y_n, z_n = Q
    r_n, g_n, b_n = rgb_im[y, x]
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

print("Rendering original camera coords")
pcd = o3d.io.read_point_cloud("im1_camera.xyzrgb", format="xyzrgb")
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

def projection_to_image_plane(xyz, rgb, rgb_im, k):
    im = np.zeros_like(rgb_im)

    for i in range(len(xyz)):
        point = xyz[i]
        color = rgb[i]
        projected_point = np.matmul(k, point)
        if (projected_point[2] != 0):
          projected_point /= projected_point[2]

        if 0 <= projected_point[1] < im.shape[0]-1 and 0 <= projected_point[0] < im.shape[1]-1:
            im[int(round(projected_point[1])), int(round(projected_point[0]))] = color

    return im

# projected = projection_to_image_plane(xyz, rgb, rgb_im_1, int_im_1)
# plt.imshow(projected)
# plt.show()

def rotation_around_axis(x, y, z, theta, axis):
    """ Rotates 3D Scene Point theta radians around specific axis
    [X', Y', Z'] = R[X, Y, Z], Where R is the rotation matrix

    Parameters: 
    - theta (rotation in radians)
    - axis (x=0, y=1, z=3)

    Returns: rotated points (X',Y',Z')
    """
    p = np.array([x,y,z])
    cos = np.cos(theta)
    sin = np.sin(theta)

    if axis=="x":
      R = np.array([[1, 0, 0],
                   [0, cos, -1*sin],
                    [0, sin, cos]])

    elif axis=="y":
      R = np.array([[cos, 0, sin],
                   [0, 1, 0],
                    [-sin, 0, cos]])

    elif axis=="z":
      R = np.array([[cos, sin, 0],
                   [sin, cos, 0],
                    [0, 0, 1]])
    
    rotated = np.matmul(R, p)
    return rotated

xyz_rxax= []

for i in range(len(xyz)):
    x, y, z = xyz[i]
    xyz_rxax.append(rotation_around_axis(x, y, z, theta=0.3, axis="x"))

# rotated_and_projected = projection_to_image_plane(xyz_r, rgb, rgb_im_1, int_im_1)
# plt.imshow(rotated_and_projected)
# plt.show()

if not path.exists("im1_rotated.xyzrgb") or testing:
    print("Creating XYZRGB file")
    with open("im1_rotated.xyzrgb", "w") as f: 
        for i in range(len(xyz_rxax)): 
            x, y, z = xyz_rxax[i]
            r = rgb[i][0] 
            g = rgb[i][1]
            b = rgb[i][2] 
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


print("Loading rotated point cloud")
pcd = o3d.io.read_point_cloud("im1_rotated.xyzrgb", format="xyzrgb")
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

'''
def projection_to_image_plane(pcd, rgb_im, k):
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)

    im = np.zeros_like(rgb_im)

    print(k)

    for i in range(pcd_points.shape[0]):
        
        point = pcd_points[i]
        color = pcd_colors[1]
        projected_point = np.matmul(k, point)
        # projected_point /= projected_point[2]
    
        if 0 <= projected_point[0] < im.shape[0] and 0 <= projected_point[1] < im.shape[1]:
            # print(int(projected_point[0]), int(projected_point[1]))
            # print(color * 255)
            im[int(projected_point[0]), int(projected_point[1])] = color * 255
    
    return im

projected = projection_to_image_plane(pcd, rgb_im_1, int_im_1)
plt.imshow(projected)
plt.show()

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

'''
