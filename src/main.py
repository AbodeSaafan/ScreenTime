# Imports
from VideoReader import VideoReader
from FaceDetector import FaceDetector
from FaceCluster import FaceCluster
from ScreenTimeGui import ScreenTimeGui
from ClassifyImage import ClassifyImage
import numpy as np

## THIS CODE IS USED FOR TESTING FACE DETECTOR ##
## IT WILL OUTPUT A VIDEO WITH DETECTED FRAMES ##

## KEEP THE FILE NAME THE SAME SO THAT IT IS GIT IGNORED ##
#out = cv2.VideoWriter('test_temp.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (1280,720))
#
#frame = vr.getNextFrame()
#
#while(len(frame) > 0 ):
#    fd.loadFrame(frame)
#    faces = fd.detectFaces()
#    newFrame = fd.showFaces(faces)
#    
#    out.write(newFrame)
#    
#    frame = vr.getNextFrame()
#
#out.release()

################################################
#%%
def computeFunction(vidPath, g):   
    vr = VideoReader(vidPath, sampleRate = 5)
    fd = FaceDetector()
    fc = FaceCluster()

    frame = vr.getNextFrame()
    
    # Number of frames we are processing
    g.setProgressMax(vr.totalSample)
    
    while(len(frame) > 0):
        # Face detection on each selected frame
        fd.loadFrame(frame)
        faces = fd.detectFaces()
        eye = fd.detectEye()
#        profile = fd.detectProfileFaces()
        
        fc.addFaces(fd.extractFaces(faces, eye))
        
        frame = vr.getNextFrame()
        g.progress()
    
    # Switches GUI to processing cluster and classification state    
    g.processingState()
    
    # Cluster faces
    fc.startCluster()
    g.fc = fc
    
    # Screentime % of each person/cluster
    g.clusterShares = fc.getScreenTimeShare()

    # Classify the clusters
    gc =  ClassifyImage("../lib/preTrained/50epoch20batch_model")
    genderClusters = []
    
    for cId in range(fc.numOfClusters):
        clusterImgs = np.array(fc.actualFaces)[np.where(fc.cluster.labels == cId)]
        genderClusters.append(gc.classifyCluster(clusterImgs, 0.5))
    
    g.genderClusters = genderClusters
    
    # Since male = 1 and female = 0 we can do a dot product to get screentime by
    # gender
    g.maleGenderShare = np.dot(np.array(genderClusters), np.array(g.clusterShares))
    g.femaleGenderShare = 1 - g.maleGenderShare
    
    g.enableResultsButton()
    

if __name__ == "__main__":
    gui = ScreenTimeGui(computeFunction)


#%%