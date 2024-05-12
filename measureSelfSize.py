import cv2
import numpy as np

image = cv2.imread('teams.png')
height, width, _ = image.shape
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(contour)
        x_perc = x/width
        y_perc = y/height
        print(f'perc of x: {x_perc}')
        print(f'perc of y: {y_perc}')