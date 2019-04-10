# Imports

#TODO only import needed stuff
import VideoReader
import FaceDetector
import FaceCluster
import ScreenTimeGui
import GenderClassifier
import ClassifyImage
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image
import glob
from keras.preprocessing import image





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
#        profile = fd.detectProfileFaces()
        
        fc.addFaces(fd.extractFaces(faces, eye))
        
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
#    gui = ScreenTimeGui.ScreenTimeGui(computeFunction)
#    gc = GenderClassifier.GenderClassifier()
#    gc.initializeCNN()
#    gc.fitCNN()

#
#    filenames = glob.glob("../data/test/female/*.jpg")
#    filenames.sort()
#    images = [image.load_img(img, target_size = (250,250)) for img in filenames]
#    
    ci = ClassifyImage.ClassifyImage()
    ci.testingAccuracy('10epoch20batch_model', "../data/test/male/*.jpg", )
#    model = ci.loadModelandWeights("50epoch20batch_model")
#    results = []
#    for i in range(len(images)):
#        results.append(ci.classifyImage(model, images[i]))
#        
#    
#    print(results.count('female'))
    
    

#%%

