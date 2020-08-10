import numpy as np
import matplotlib.pyplot as plt

# Read in the images 
fig, ax = plt.subplots(2, 2, figsize=(12,6))

pair1_1 = plt.imread("Part2/first_pair/p11.jpg")
pair1_2 = plt.imread("Part2/first_pair/p12.jpg")
pair2_1 = plt.imread("Part2/second_pair/p21.jpg")
pair2_2 = plt.imread("Part2/second_pair/p22.jpg")

ax[0][0].set_title("First Stereo Pair", fontsize=15)
ax[0][0].imshow(pair1_1)
ax[1][0].imshow(pair1_2)
ax[0][1].set_title("Second Stereo Pair", fontsize=15)
ax[0][1].imshow(pair2_1)
ax[1][1].imshow(pair2_2)
fig.tight_layout()
plt.show()


# Hand recording coorrespondences
plt.imshow(pair1_1)
plt.show()

plt.imshow(pair1_2)
plt.show()

plt.imshow(pair2_1)
plt.show()

plt.imshow(pair2_2)
plt.show()

_ = print("Size of Images:", pair1_1.shape) # all are the same size