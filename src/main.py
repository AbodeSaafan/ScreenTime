# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import VideoReader
import FaceDetector

vr = VideoReader.VideoReader('../lib/Sample_1.mp4', sampleRate = 1)
fd = FaceDetector.FaceDetector()

# Skipping a minute in for testing
for i in range(40):
    frame = fd.detectFace(vr.getNextFrame())
    
# This displays it
plt.figure()
plt.axis('off')
plt.imshow(frame)
plt.show()
