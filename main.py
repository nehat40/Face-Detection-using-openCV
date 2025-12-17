import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


img = cv.imread("pasar-tiempo-con-los-mejores-amigos-45609558.webp")  # or frame from VideoCapture

# Correct grayscale conversion
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,4)
print(faces)

for x,y,w,h in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)

cv.imshow("Original", img)
cv.waitKey(0)
cv.destroyAllWindows()

