import cv2

class FaceDetector():
    def __init__(self):
        
#        cascades are classifiers to detect facial features
#        can train a custom cascade or use cv2's
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.right_eye =  cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

    # Load frame that we are working with
    # TODO: Re-eval this method later
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
        
        return visualFaces
    
    
    def detectFaces(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.4, 4)
        
        return faces
    
    def detectEye(self):
#        Only detecting one eye because some faces have a profile angle
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        sharingan = self.right_eye.detectMultiScale(gray)
        
        return sharingan
    
    def showFaces(self, faces):
        frame = self.frame
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
            
        return frame
    
    