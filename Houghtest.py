import cv2
import numpy as np

img = cv2.imread('images/frame-0.png')

lowerl1 = np.array([0, 230, 0])
upperl1 = np.array([155, 255, 155])

lowerl2 = np.array([230, 0, 0])
upperl2 = np.array([255, 155, 155])

maskl2 = cv2.inRange(img, lowerl2, upperl2)

imgl2 = 1.0 * maskl2

maskl1 = cv2.inRange(img, lowerl1, upperl1)
imgl1 = 1.0 * maskl1

maxLineGap=20
minLineLength=5
lines = cv2.HoughLinesP(maskl1,1,np.pi/180,100,minLineLength,maxLineGap)

print lines
for x1,y1,x2,y2 in lines[0]:
       cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

lines = cv2.HoughLinesP(maskl2,1,np.pi/180,100,minLineLength,maxLineGap)
"""
print lines
for x1,y1,x2,y2 in lines[0]:
       cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
"""

cv2.imshow('gray',img)
cv2.waitKey()