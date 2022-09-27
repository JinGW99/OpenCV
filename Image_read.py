import cv2

image_file = "image/iu.jpg"

original = cv2.imread(image_file, cv2.IMREAD_COLOR)

gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

unchange = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

cv2.imshow("IMREAD_COLOR", original)
cv2.imshow("IMREAD_GRAYSCALE", gray)
cv2.imshow("IMREAD_UNCHANGEED", unchange)

cv2.waitKey(0)