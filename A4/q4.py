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

# Hand Recorded Correspondences

# top left, top right, bottom left, bottom right
p11_monitor = [[349, 157], [727, 159], [353, 393], [735, 383]]
p11_pencil = [[234, 393], [327, 386], [236, 492], [330, 484]]
p11_computer = [773, 172]
p11_keyboard = [[431, 526], [809, 513]]

p12_monitor = [[382, 175], [741, 181], [385, 405], [747, 398]]
p12_pencil = [[276, 408], [364, 402], [276, 506], [366, 496]]
p12_computer = [777, 191]
p12_keyboard = [[485, 539], [840, 825]]

p21_monitor = [[368, 81], [786, 64], [380, 347], [783, 308]]
p21_pencil = [[322, 367], [422, 350], [324, 469], [None, None]]
p21_computer = [819, 55]
p21_keyboard = [[510, 557], [None, None]]
p21_phone = [[332, 513], [398, 508], [332, 576], [405, 572]]

p22_monitor = [[397, 90], [846, 79], [402, 362], [840, 337]]
p22_pencil = [[332, 379], [441, 365], [335, 481], [None, None]]
p22_computer = [897, 69]
p22_keyboard = [[494, 587], [None, None]]
p22_phone = [[327, 530], [391, 530], [306, 592], [376, 597]]

# View for Hand recording correspondences
plt.imshow(pair1_1)
for quad in [p11_monitor, p11_pencil]:
    for i in range(4):
        plt.plot(quad[i][0],quad[i][1], "r.", markersize=10)
plt.show()

plt.imshow(pair1_2)
for quad in [p12_monitor, p12_pencil]:
    for i in range(4):
        plt.plot(quad[i][0],quad[i][1], "r.", markersize=10)
plt.show()

plt.imshow(pair2_1)
for quad in [p21_monitor, p21_phone]:
    for i in range(4):
        plt.plot(quad[i][0],quad[i][1], "r.", markersize=10)
plt.show()

plt.imshow(pair2_2)
for quad in [p22_monitor, p22_phone]:
    for i in range(4):
        plt.plot(quad[i][0],quad[i][1], "r.", markersize=10)
plt.show()

_ = print("Size of Images:", pair1_1.shape) # all are the same size