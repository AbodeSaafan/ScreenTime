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
    vr.getNextFrame()

fd.loadFrame(vr.getNextFrame())

faces = fd.detectFaces()

visFaces = fd.extractFaces(faces)

    
# This displays it
plt.figure()
plt.axis('off')
plt.imshow(visFaces[0])
plt.show()

plt.figure()
plt.imshow(fd.showFaces(faces))