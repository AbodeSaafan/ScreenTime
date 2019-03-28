#https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
#https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf
#https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/
#https://github.com/cmusatyalab/openface
import cv2

'''
This class is used to create the cluster of faces 
'''
class FaceCluster():
    def __init__(self):
        # Do stuff here like init vars we need
        self.embedder = cv2.dnn.readNetFromTorch("../lib/nn4.small2.v1.t7")
        self.faceVectors = []
        
    def addFaces(self, faces):
        # Add to cluster
        for face in faces:
            # Creating a blob to pass into the network
            # TODO read more about this though 
            faceBlob = cv2.dnn.blobFromImage(face, 
                                             1.0 / 255, # Scale factor
                                             (96, 96)) # Spatial size

            self.embedder.setInput(faceBlob)
            self.faceVectors.append(self.embedder.forward())
            
    def getFaceVectors(self):
        return self.faceVectors