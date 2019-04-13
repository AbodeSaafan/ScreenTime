import cv2

class FaceDetector():
    def __init__(self, minN):
        self.minN = minN
        
        #cascades are classifiers to detect facial features
        #can train a custom cascade or use cv2's
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.rightEye =  cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
        self.profileFace =  cv2.CascadeClassifier('haarcascade_profileface.xml')

    def loadFrame(self, frame):
        self.frame = frame
        
    # Get an image of just the face from the frame 
    # Meant to be called in detect faces
    def extractFaces(self, faces, eyes):
        visualFaces = []
        
        for (x,y,w,h) in faces:
            for (a,b,c,d) in eyes:
                if (a > x and b > y and a+c < x+w and b+d < y+h):
                    visualFaces.append(self.frame[y:y+h, x:x+w, :])
        
        
#        for (s,t,u,v) in profile:
#            visualFaces.append(self.frame[t:t+v, s:s+u, :])

        return visualFaces
    
    
    def detectFaces(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
#        optimal params for the friends clip
        faces = self.faceCascade.detectMultiScale(gray,
                                                  scaleFactor=1.1,
                                                  minNeighbors= self.minN)
        
        return faces
    
#   might keep profile face detection instead of this, but this eliminates noise for some cases
    def detectEye(self):
#        Only detecting one eye because some faces have a profile angle
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        sharingan = self.rightEye.detectMultiScale(gray)
        
        return sharingan
    
#   not a redundant cascade, frontal does not capture the same objects as profil
#    def detectProfileFaces(self):
#        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
#        profile = self.profileFace.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#        
#        return profile
    
    def showFaces(self, faces):
        frame = self.frame
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
            
        return frame
    
    