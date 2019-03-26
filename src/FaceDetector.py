import cv2

class FaceDetector():
    def __init__(self):
        
#        cascades are classifiers to detect facial features
#        can train a custom cascade or use cv2's
#        TODO: find efficient cascade approach
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
    def detectFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        TODO: find optimal scale and neighbours parameter        
        faces = self.face_cascade.detectMultiScale(gray, 1.4, 4)
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       
        return frame
