import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class HairCounter():
    def __init__(self, image):
        self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.brightness_center = Point()

    def gama_correction(self, gamma):
        gamma_corrected = np.power(self.image / 255.0, gamma)
        gamma_corrected = np.uint8(gamma_corrected * 255)
        self.image = gamma_corrected
        
    def threshold(self):
        thresh = cv.threshold(self.image, 150, 255, cv.THRESH_TOZERO)[1]
        self.image = thresh
        
    def get_brightness_center(self):
        kernel = np.ones((5, 5), np.uint8)
        veryEroded = cv.dilate(self.image, kernel, iterations=3)
        contornos, _ = cv.findContours(veryEroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        maior_contorno = max(contornos, key=cv.contourArea)
        momentos = cv.moments(maior_contorno)
        self.brightness_center.x = int(momentos['m10'] / momentos['m00'])
        self.brightness_center.y = int(momentos['m01'] / momentos['m00'])        
        self.largest_contour = maior_contorno
        
    def erode(self):
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv.erode(self.image, kernel, iterations=1)
        eroded = cv.cvtColor(eroded, cv.COLOR_GRAY2BGR)
        self.image = eroded
        
    def get_contours(self):
        bordas = cv.Canny(self.image, threshold1=100, threshold2=200)
        contornos, _ = cv.findContours(bordas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contornos
        
    def count_hair(self):
        quantidade_fios = 0
        for _, contorno in enumerate(self.contours):
            area = cv.contourArea(contorno)
            if area > 0:
                momentos = cv.moments(contorno)
                centro_x = int(momentos['m10'] / momentos['m00'])
                centro_y = int(momentos['m01'] / momentos['m00'])

                cv.drawContours(self.image, [contorno], -1, (0, 255, 0), 2)
                cv.putText(self.image, f"{area:.2f}", (centro_x, centro_y), cv.FONT_ITALIC, 0.2, (0, 0, 255), 1)
                
                if area > 200:
                    quantidade_fios += 2
                else:
                    quantidade_fios += 1
                
        self.hair_number = quantidade_fios
        
    def resize(self, factor):
        self.image = cv.resize(self.image, (0, 0), fx=factor, fy=factor)
        
    def show(self):
        plt.imshow(self.image)
        plt.show()
        
    def find_bigger_square(self):
        mask = np.zeros_like(self.image)
        cv.drawContours(mask, [self.largest_contour], 0, (255, 255, 255), thickness=cv.FILLED)
        filled_contour = cv.addWeighted(self.image, 1, mask, 0.5, 0)
        center_x, center_y = self.brightness_center.x, self.brightness_center.y
        
        for i in range(0, filled_contour.shape[0]):
            x = center_x - i
            y = center_y - i
            w = 2 * i
            h = 2 * i
            roi = filled_contour[y:y+h, x:x+w]
            black_pixels = np.count_nonzero(roi == 0)
            
            if black_pixels > 0:
                cv.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.image = self.image[y:y+h, x:x+w]
                self.used_area = {
                    'x1': x,
                    'y1': y,
                    'x2': x + w,
                    'y2': y + h
                }
                break
            
        self.used_area = {
            'x1': 0,
            'y1': 0,
            'x2': self.image.shape[1],
            'y2': self.image.shape[0]
        }
    
    def run(self):
        self.gama_correction(0.5)
        self.threshold()
        self.get_brightness_center()
        self.erode()
        self.find_bigger_square()
        self.get_contours()
        self.count_hair()   
        
class Point():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        
 