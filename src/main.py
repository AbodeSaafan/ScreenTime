from VideoReader import VideoReader
from FaceDetector import FaceDetector
from FaceCluster import FaceCluster
from ScreenTimeGui import ScreenTimeGui
from ClassifyImage import ClassifyImage
import numpy as np
import argparse


'''
This compute function is passed into the GUI class and is called when a 
user selects a video
'''
def computeFunction(vidPath, g):   
    vr = VideoReader(vidPath, sampleRate = g.args.sampleRate)
    fd = FaceDetector(g.args.faceDetectionMinN)
    fc = FaceCluster(g.args.faceVectorScale, 
                     g.args.clusterEpsi,
                     g.args.clusterMinSamples)

    frame = vr.getNextFrame()
    
    # Number of frames we are processing
    g.setProgressMax(vr.totalSample)
    
    while(len(frame) > 0):
        # Face detection on each selected frame
        fd.loadFrame(frame)
        faces = fd.detectFaces()
        eye = fd.detectEye()
        #profile = fd.detectProfileFaces() Used for detecting profile faces
        
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

    # Classify the clusters one at a time
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
    

'''
This function takes care of the arguments based on the command line to launch
the program. They are not super user-friendly since they are meant to be 
changed by an admin/dev only. Refer to report.
'''
def processArguments():
    # Processing arguments for the program
    parser = argparse.ArgumentParser(description="ScreenTime application")
    
    # Since we don't know the frame rate of the video we have a loose restriction
    # on the frame rate. But it should atleast be 1 or above    
    parser.add_argument("--sampleRate",
                        metavar="1 <= sampleRate <= video frame rate",
                        type=int,
                        nargs="?",
                        default=5,
                        choices=range(1, 200),
                        help="How often per second we sample the video")
    
    # 60000 is just so that we only have a lower limit, ideally these arguments
    # will never get up that high
    parser.add_argument("--clusterMinSamples",
                        metavar="1 <= clusterMinSamples",
                        type=int,
                        nargs="?",
                        default=5,
                        choices=range(1, 60000),
                        help="MinSamples argument for DBSCAN clustering")
        
    parser.add_argument("--faceDetectionMinN",
                        metavar="1 <= faceDetectionMinN",
                        type=int,
                        nargs="?",
                        default=4,
                        choices=range(1, 60000),
                        help="Min neighbours for face detection")
    
    parser.add_argument("--clusterEpsi",
                        metavar="0 <= clusterEpsi",
                        type=float,
                        nargs="?",
                        default=0.4,
                        help="Epsilon argument for DBSCAN clustering")
    
    parser.add_argument("--faceVectorScale",
                        metavar="0 < faceVectorScale <= 1",
                        type=float,
                        nargs="?",
                        default=1.0/255,
                        help="Scale argument for vector conversion")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = processArguments()
    
    gui = ScreenTimeGui(computeFunction, args)