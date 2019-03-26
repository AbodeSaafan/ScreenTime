# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import VideoReader


vr = VideoReader.VideoReader('../lib/Sample_1.mp4', sampleRate = 1)

# Skipping a minute in for testing
for i in range(10):
    vr.getNextFrame()

# This displays it
plt.figure()
plt.axis('off')
plt.imshow(vr.getNextFrame())
