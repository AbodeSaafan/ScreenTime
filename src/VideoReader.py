from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
from os import path

'''
This class interfaces with the video file and keeps track of
which frames we want to grab based on sample rate and frame rate
'''
class VideoReader:
    def __init__(self, videoPath, sampleRate):
        if(not(path.isfile(videoPath))):
            raise ValueError("Invalid video file")
        self.cap = VideoCapture(videoPath)
        
        self.frameRate = self.cap.get(CAP_PROP_FPS)
        
        # Calculating how many frames we will look at
        totalFrames = self.cap.get(CAP_PROP_FRAME_COUNT)
        self.totalSample = int((totalFrames / self.frameRate) / sampleRate)
        
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
            return frame
        
        else:
            return []