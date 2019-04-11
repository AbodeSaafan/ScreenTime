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
    vr = VideoReader(vidPath, sampleRate = 1)
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
    
    # Classify the clusters
    gc =  ClassifyImage("5epoch10batch_model")
    
    for cId in range(fc.numOfClusters - 1):
        clusterImgs = np.array(fc.actualFaces)[np.where(fc.cluster.labels == cId)]
        gc.classifyCluster(clusterImgs, 0.5)
    
    
    
    
    g.enableResultsButton()
    #g.showClusterResults(fc)
    
    #fc.showClusterResults()
    
    #x = fc.getScreenTimeShare()
#    fc.getClusterImages(0)
    
   
if __name__ == "__main__":
    gui = ScreenTimeGui(computeFunction)
#    gc = GenderClassifier.GenderClassifier()
#    gc.initializeCNN()
#    gc.fitCNN()

#
#    filenames = glob.glob("../data/test/female/*.jpg")
#    filenames.sort()
#    images = [image.load_img(img, target_size = (250,250)) for img in filenames]
#    
#    ci = ClassifyImage()
#    ci.testingAccuracy('2epoch10batch_model', "../data/test/male/*.jpg",'female')
#    model = ci.loadModelandWeights("50epoch20batch_model")
##    results = []
#    for i in range(len(images)):
#        results.append(ci.classifyImage(model, images[i]))
#        
#    
#    print(results.count('female'))
    
    

#%%