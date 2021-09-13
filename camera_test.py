import numpy as np
import cv2
import os
import time

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
print("[INFO] Detector set!")

last_time = time.time()
print("[INFO] Time set!")
print("[INFO] Setting Video Capture device...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("[INFO] Video Capture device set!")
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('frame', frame)
    print('Frame time: {} seconds'.format(time.time()-last_time))

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        print("[INFO] Camera test stopped!")
        break
cap.release()
cv2.destroyAllWindows()
