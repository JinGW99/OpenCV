import cv2
from matplotlib import pyplot as plt

image_file = "image/iu.jpg"

original = cv2.imread(image_file, cv2.IMREAD_COLOR)

gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

unchange = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

color = ('b', 'g', 'r')

for i,col in enumerate(color):
    hist = cv2.calcHist([original], [i], None, [256], [0, 256])
    plt.figure(1)
    plt.plot(hist, color = col)


plt.show()

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure(2)
plt.plot(hist)
plt.show()

gray = cv2.equalizeHist(gray)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure(2)
plt.plot(hist)
plt.show()