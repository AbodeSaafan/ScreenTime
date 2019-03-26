# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Test function
def getIthFrame(cap, i):
    for x in range(i-1):
        cap.read()
        
    return cap.read()


# This loads the video file
cap = cv2.VideoCapture('../lib/Sample_1.mp4')

# This grabs a frame from the capture
ret, frame = getIthFrame(cap, 225)


# This converts the frame into rgb
visFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

# THis displays it
plt.figure()
plt.axis('off')
plt.imshow(visFrame)