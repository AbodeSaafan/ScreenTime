import cv2

'''
This class interfaces with the video file and keeps track of
which frames we want to grab based on sample rate and frame rate
'''
class VideoReader:
    def __init__(self, videoPath, sampleRate):
        # TODO: Check that video path is valid
        self.cap = cv2.VideoCapture(videoPath)
        
        self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        
        if (sampleRate < 1 or sampleRate > self.frameRate):
            raise ValueError("Invalid sample rate")
        self.sampleRate = sampleRate
        
        
    # Gives you the next frame based on sample rate
    def getNextFrame(self):
        # Skip to next frame we need
        for i in range(int(self.frameRate/self.sampleRate) - 1):
            self.cap.read()
        
        ret, frame = self.cap.read()
        
        if(ret):
            return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        return False