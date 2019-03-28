# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import VideoReader
import FaceDetector
import FaceCluster

vr = VideoReader.VideoReader('../lib/Sample_1.mp4', sampleRate = 1)
fd = FaceDetector.FaceDetector()
fc = FaceCluster.FaceCluster()

## Skipping a minute in for testing
#for i in range(42):
#    vr.getNextFrame()
#
#fd.loadFrame(vr.getNextFrame())
#
#faces = fd.detectFaces()
#
#visFaces = fd.extractFaces(faces)
#
#fc.addFaces(visFaces)
    
# This displays it
#plt.figure()
#plt.axis('off')
#plt.imshow(visFaces[0])
#plt.show()
#
#plt.figure()
#plt.imshow(fd.showFaces(faces))




## THIS CODE IS USED FOR TESTING FACE DETECTOR ##
## IT WILL OUTPUT A VIDEO WITH DETECTED FRAMES ##

## KEEP THE FILE NAME THE SAME SO THAT IT IS GIT IGNORED ##
out = cv2.VideoWriter('test_temp.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (1280,720))

frame = vr.getNextFrame()

while(len(frame) > 0 ):
    fd.loadFrame(frame)
    faces = fd.detectFaces()
    newFrame = fd.showFaces(faces)
    
    out.write(newFrame)
    
    frame = vr.getNextFrame()

out.release()

################################################