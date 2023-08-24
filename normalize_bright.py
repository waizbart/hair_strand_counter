import cv2
import numpy as np

# Load the image
image = cv2.imread('hair2.png', cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply gamma correction
gamma = 0.5
gamma_corrected = np.power(gray_image / 255.0, gamma)
gamma_corrected = np.uint8(gamma_corrected * 255)

corrected_image = cv2.resize(gamma_corrected, (0, 0), fx=0.5, fy=0.5)

cv2.imshow('corrected', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
