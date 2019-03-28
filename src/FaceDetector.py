import cv2

class FaceDetector():
    def __init__(self):
        
#        cascades are classifiers to detect facial features
#        can train a custom cascade or use cv2's
#        TODO: find efficient cascade approach
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Load frame that we are working with
    # TODO: Re-eval this method later
    def loadFrame(self, frame):
        self.frame = frame
        
    # Get an image of just the face from the frame 
    # Meant to be called in detect faces
    def extractFaces(self, faces):
        visualFaces = []
        
        for (x,y,w,h) in faces:
            visualFaces.append(self.frame[y:y+h, x:x+w, :])
        
        return visualFaces
    
    # Should just return a bunch of faces (array of faces?)
    def detectFaces(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
#        TODO: find optimal scale and neighbours parameter        
        faces = self.face_cascade.detectMultiScale(gray, 1.4, 4)
        
        return faces
    

    # Calls detect face or maybe not? and then returns frame that shows it
    # Eats the input from detectFaces ? 
    def showFaces(self, faces):
        frame = self.frame
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
            
        return frame
    
    