import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

image = cv.imread('hair1.png', cv.IMREAD_COLOR)

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

gamma = 0.5
gamma_corrected = np.power(gray_image / 255.0, gamma)
gamma_corrected = np.uint8(gamma_corrected * 255)

ret, thresh = cv.threshold(gamma_corrected, 190, 255, cv.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
veryEroded = cv.dilate(thresh, kernel, iterations=3)
contornos, _ = cv.findContours(veryEroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_contour = max(contornos, key=cv.contourArea)

x, y, w, h = cv.boundingRect(largest_contour)

rectangular_mask = np.zeros_like(image)
cv.drawContours(rectangular_mask, [largest_contour], 0, (255, 255, 255), -1)

cropped = cv.bitwise_and(image, rectangular_mask)
cropped = cropped[y:y+h, x:x+w]

def objective_function(params, initial_image):
    x, y, w, h = params
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    
    roi = initial_image[y:y+h, x:x+w]
    black_pixels = np.count_nonzero(roi == 0)
    print(black_pixels)
    return -w * h * black_pixels

image_width = cropped.shape[1]
image_height = cropped.shape[0]

initial_guess = [0, 0, image_width, image_height]
bounds = [(0, image_width), (0, image_height), (0, image_width), (0, image_height)]
result = minimize(objective_function, initial_guess, args=(cropped), bounds=bounds, method='L-BFGS-B')

# Extrair os parâmetros do retângulo
x, y, w, h = result.x
print(f"Retângulo: x={x}, y={y}, w={w}, h={h}")
# Desenhar o retângulo delimitador no contorno original
cv.rectangle(cropped, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

resized = cv.resize(cropped, (0, 0), fx=0.5, fy=0.5)

plt.imshow(resized)
plt.show()
