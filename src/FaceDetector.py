import cv2

class FaceDetector():
    def __init__(self):
        
#        cascades are classifiers to detect facial features
#        can train a custom cascade or use cv2's
#        TODO: find efficient cascade approach
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Loading current frame that we are working on instead of passing it 
    def loadFrame(self, frame):
        self.frame = frame
        
    # Get an image of just the face from the frame 
    # Meant to be called in detect faces
    def extractFace(self, frame, face):
        return
        
    # Should just return a bunch of faces (array of faces?)
    def detectFaces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        TODO: find optimal scale and neighbours parameter        
        faces = self.face_cascade.detectMultiScale(gray, 1.4, 4)
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       
        # Should return faces
        return frame
    

    # Calls detect face or maybe not? and then returns frame that shows it
    # Eats the input from detectFaces ? 
    def showFaces(self, frame, faces):
        return 
    
    