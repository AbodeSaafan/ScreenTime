# Imports

#TODO only import needed stuff
import VideoReader
import FaceDetector
import FaceCluster
import ScreenTimeGui
import matplotlib.pyplot as plt
import Dbscan


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


#frame = vr.getNextFrame()
#
#while(len(frame) > 0 ):
#    fd.loadFrame(frame)
#    faces = fd.detectFaces()
#    eye = fd.detectEye()
#    profile = fd.detectProfileFaces()
#    
#    fc.addFaces(fd.extractFaces(faces, eye, profile))
#    
#    frame = vr.getNextFrame()
#
#fc.startCluster()
##fc.showClusterResults()
##x = fc.getScreenTimeShare()
#c1 =  fc.getClusterImages(0)



#%%
def computeFunction(vidPath, g):   
    vr = VideoReader.VideoReader(vidPath, sampleRate = 1)
    fd = FaceDetector.FaceDetector()
    fc = FaceCluster.FaceCluster()

    frame = vr.getNextFrame()
    
    # Number of frames we are processing
    g.setProgressMax(vr.totalSample)
    
    while(len(frame) > 0):
        fd.loadFrame(frame)
        faces = fd.detectFaces()
        eye = fd.detectEye()
        profile = fd.detectProfileFaces()
        
        fc.addFaces(fd.extractFaces(faces, eye, profile))
        
        frame = vr.getNextFrame()
        g.progress()
        
    g.clusterState()
    fc.startCluster()
    
    g.fc = fc
    g.enableClusterButton()
    #g.showClusterResults(fc)
    
    #fc.showClusterResults()
    
    #x = fc.getScreenTimeShare()
#    fc.getClusterImages(0)
    
   
if __name__ == "__main__":
    #gui = ScreenTimeGui.ScreenTimeGui(computeFunction)
    vr = VideoReader.VideoReader("../lib/sample_1.mp4", sampleRate = 1)
    fd = FaceDetector.FaceDetector()
    fc = FaceCluster.FaceCluster()
    
   
    frame = vr.getNextFrame()   
    
#    i = 0
#    while(len(frame) > 0 and i < 10):
#        i+= 1
#        fd.loadFrame(frame)
#        faces = fd.detectFaces()
#        eye = fd.detectEye()
#        profile = fd.detectProfileFaces()
#        
#        fc.addFaces(fd.extractFaces(faces, eye, profile))
#        
#        frame = vr.getNextFrame()
        
    #db.fit(fc.faceVectors)
    import numpy as np
    testVecs = [np.array([1,5]),
                np.array([5,5]),
                np.array([6,7]),
                np.array([1,1]),
                np.array([10,3]),
                np.array([3,3]),
                np.array([0,4]),
                np.array([2,8]),
                np.array([1,9]),
                np.array([6,5]),
                np.array([8,7]),
                np.array([7,7])
           ] 
    from sklearn.cluster import DBSCAN
    clu = DBSCAN(metric="euclidean", eps=2, min_samples=2).fit(testVecs)
    print(clu.labels_)
    print(np.array(testVecs)[np.where(clu.labels_ == 1)])
    db = Dbscan.Dbscan(2, 2)
    
    clusters = db.fit(testVecs)
#%%