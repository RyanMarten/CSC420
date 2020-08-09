import numpy as np 
import matplotlib.pyplot as plt
import math
import open3d as o3d

"""# Part I: Projection

## Q1) 3-D Point Cloud
"""

# Non interactive Mode

rgb_im_1 = plt.imread("Part1/1/rgbImage.jpg")
rgb_im_2 = plt.imread("Part1/2/rgbImage.jpg")
rgb_im_3 = plt.imread("Part1/3/rgbImage.jpg")
depth_im_1 = plt.imread("Part1/1/depthImage.png")
depth_im_2 = plt.imread("Part1/2/depthImage.png")
depth_im_3 = plt.imread("Part1/3/depthImage.png")

"""
fig, ax = plt.subplots(2, 3, figsize=(12,6))
ax[0][0].imshow(rgb_im_1)
ax[0][1].imshow(rgb_im_2)
ax[0][2].imshow(rgb_im_3)
ax[1][0].imshow(depth_im_1)
ax[1][1].imshow(depth_im_2)
ax[1][2].imshow(depth_im_3)
fig.tight_layout()
ax[0][1].set_title("input images and depths", fontsize=15)
plt.ioff()
plt.show()
"""

# Read in the instrinsic and extrinsic parameters
ext_im_1 = np.loadtxt("Part1/1/extrinsic.txt")
ext_im_2 = np.loadtxt("Part1/2/extrinsic.txt")
ext_im_3 = np.loadtxt("Part1/3/extrinsic.txt")

int_im_1 = np.loadtxt("Part1/1/intrinsics.txt")
int_im_2 = np.loadtxt("Part1/2/intrinsics.txt")
int_im_3 = np.loadtxt("Part1/3/intrinsics.txt")

print("Extrinsic Camera Parameters: \n", ext_im_1)
print("\nInstrinsic Camera Parameters: \n", int_im_2)

# working with image 1 first
k = int_im_1
x = 20
y = 100
rgb_im = rgb_im_1
depth_im = depth_im_1

def compute_point_cloud(rgb_im, depth_im, x, y, k):
  """ Computes world coordinates of pixels from the depth map provided 

  Parameters: 
    - rgb_im (RGB image)
    - depth_im (Depth map image)
    - k (camera calibration matrix)
    - x, y (pixel location)

  Returns: (X,Y,Z,R,G,B)
  """
  q = np.array([x, y, 1]) * depth_im[x,y]
  Q = np.matmul(np.linalg.inv(k), q)
  
  x_n, y_n, z_n = Q
  r_n, g_n, b_n = rgb_im[x,y]
  return x_n, y_n, z_n, r_n, g_n, b_n

print(compute_point_cloud(rgb_im, depth_im, x, y, k))

rgbd_im_1 = np.zeros((rgb_im.shape[0], rgb_im.shape[1], 4))

xyz = []
rgb = []

"""
# THIS IS CORRECT!
with open("test.xyzrgb", "w") as f: 
  for i in range(rgb_im.shape[0]):
    for j in range(rgb_im.shape[1]):
      x,y,z,r,g,b = compute_point_cloud(rgb_im_1, depth_im_1, i, j, k)
      r /= 255
      g /= 255
      b /= 255
      xyz.append([x,y,z])
      rgb.append([r,g,b])
      f.write(f"{x} {y} {z} {r} {g} {b}\n")
"""

# save the output of point_cloud into "pointCloud.txt"
# formatted: X_n,Y_n,Z_n,R_n,G_n,B_n
# where n is the number of 3D points with 3 coords and 3 color channel values
"""
with open("pointCloud.txt", "w") as f: 
  for i in range(rgb_im.shape[0]):
    for j in range(rgb_im.shape[1]):
      x,y,z,r,g,b = compute_point_cloud(rgb_im_1, depth_im_1, i, j, k)
      f.write(f"{x},{y},{z},{r},{g},{b}\n")
"""

'''

!head test.xyzrgb

# Why are there duplicates?????

print(rgb_im_1.shape[0] * rgb_im_1.shape[1])

# visualize the point cloud
# http://www.open3d.org/docs/release/tutorial/Basic/pointcloud.html
'''

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("test.xyzrgb", format="xyzrgb")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

'''

# visualizer = JVisualizer()
# visualizer.add_geometry(pcd)
# visualizer.show()

# The issue is this is a notebook I think 
# http://www.open3d.org/docs/release/tutorial/Basic/jupyter.html
# The visualizer is broken even on their own docs... I will need to run seperately
# in a python file locally.

# try visualizing another way 
# https://towardsdatascience.com/discover-3d-point-cloud-processing-with-python-6112d9ee38e7
# https://www.flyvast.com/flyvast/app/page-snapshot-viewer.html#/333/ec8d9a6c-de38-7249-e6fc-026c4ff67ef7
from mpl_toolkits import mplot3d

xyz = np.array(xyz)
rgb = np.array(rgb)

ax = plt.axes(projection='3d')
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = rgb/255, s=0.01)
plt.show()





plt.imshow(depth_im_1, cmap="viridis")
plt.colorbar()

print(depth_im_1.min(), depth_im_2.max())

print(depth_im_1[20][100])

# https://gist.github.com/CMCDragonkai/dd420c0800cba33142505eff5a7d2589
# Credit to Roger Qiu for this 3D surface plot function
def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)

surface_plot(depth_im_1, cmap="viridis")
"""
'''