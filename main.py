import cv2 as cv
import numpy as np

# Todo: pela área do contorno, determinar a quantidade de fios de cabelo

image = cv.imread('hair1.png', cv.IMREAD_COLOR)

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

gamma = 0.5
gamma_corrected = np.power(gray_image / 255.0, gamma)
gamma_corrected = np.uint8(gamma_corrected * 255)

ret,thresh = cv.threshold(gamma_corrected,190,255,cv.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)

veryEroded = cv.dilate(thresh, kernel, iterations=3)
contornos, _ = cv.findContours(veryEroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
maior_contorno = max(contornos, key=cv.contourArea)
momentos = cv.moments(maior_contorno)
centro_x = int(momentos['m10'] / momentos['m00'])
centro_y = int(momentos['m01'] / momentos['m00'])

kernel = np.ones((3,3),np.uint8)
eroded = cv.erode(thresh, kernel, iterations=1)
eroded = cv.cvtColor(eroded, cv.COLOR_GRAY2BGR)

resized_image = cv.resize(eroded, (0, 0), fx=0.5, fy=0.5)
cv.imshow('corrected', resized_image)
cv.waitKey(0)

square_size = 400

cropped = eroded[centro_y-square_size:centro_y+square_size, centro_x-square_size:centro_x+square_size]
bordas = cv.Canny(cropped, threshold1=100, threshold2=200)

cv.imshow('Bordas', bordas)
cv.waitKey(0)

contornos, _ = cv.findContours(bordas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
quantidade_fios = len(contornos)
print(f"Quantidade de fios de cabelo: {quantidade_fios}")
cv.drawContours(cropped, contornos, -1, (0, 255, 0), 2)

cv.imshow('corrected', cropped)
cv.waitKey(0)

cv.destroyAllWindows()
