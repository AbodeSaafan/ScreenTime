#https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
#https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf
#https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/
#https://github.com/cmusatyalab/openface
import cv2
from Dbscan import Dbscan
import matplotlib.pyplot as plt
import numpy as np


'''
This class is used to create the cluster of faces 
'''
class FaceCluster():
    def __init__(self):
        # Do stuff here like init vars we need
        self.embedder = cv2.dnn.readNetFromTorch("../lib/nn4.small2.v1.t7")
        self.faceVectors = []
        self.actualFaces = []
        
    def addFaces(self, faces):
        # Add to cluster
        for face in faces:
            # Creating a blob to pass into the network
            # TODO read more about this though 
            faceBlob = cv2.dnn.blobFromImage(cv2.resize(face, (96,96)), 
                                             1.0/255, # Scale factor
                                             (96, 96)) # Spatial size

            self.embedder.setInput(faceBlob)
            
            self.actualFaces.append(face)
            self.faceVectors.append(self.embedder.forward()[0])
            
    def getFaceVectors(self):
        return self.faceVectors
    
    def startCluster(self):
        # Create instance of cluster and cluster the faces
        self.cluster = Dbscan(.4, 5)
        self.cluster.fit(self.faceVectors)
        self.numOfClusters = max(self.cluster.labels) + 1
        
    def showClusterResults(self):
        # Loop through each cluster
        for c in range(-1, self.numOfClusters + 1):
            matches = np.where(self.cluster.labels == c)[0]
            
            plt.figure()
            for i in range(len(matches)):
                if(i > 24):
                    break
                plt.subplot(5,5, i+1)
                plt.axis('off')
                plt.imshow(cv2.cvtColor(self.actualFaces[matches[i]], cv2.COLOR_BGR2RGB))
    
    def getClusterImages(self, c):
        clusterPics = []
        matches = np.where(self.cluster.labels == c)[0]
            
        plt.figure()
        for i in range(len(matches)):
            clusterPics.append(cv2.cvtColor(self.actualFaces[matches[i]], cv2.COLOR_BGR2RGB))
        return clusterPics
    
    def getScreenTimeShare(self):
        clusterShare = []
        
        # All the positively identified faces
        posFaces = len(np.where(self.cluster.labels != -1)[0])
        
        # Loop through each cluster
        for c in range(0, self.numOfClusters):
            matches = np.where(self.cluster.labels == c)[0]
            
            clusterShare.append(float(len(matches))/posFaces)
        return clusterShare