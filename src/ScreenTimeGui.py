from Tkinter import Tk, Label, Button, mainloop
from ttk import Progressbar
import tkFileDialog
from os import path
from threading import Thread
from multiprocessing import Process, Manager, Queue

class ScreenTimeGui:
    InitialLoad = 0
    mainFont = ("Helvetica", 18)

    def __init__(self, computeFunc):
        # Compute function that we will use later
        self.__compute = computeFunc
        
        # Master window
        self.__master = Tk()
        self.__master.title("ScreenTime")
        
        # Text for user instructions 
        self.__instructionsText = Label(self.__master,
                                      font=ScreenTimeGui.mainFont, 
                                      text="Please select the video you want to run ScreenTime on")
        self.__instructionsText.grid(row=0, column=0, padx=(10, 40))
        
        # Button for selecting video
        self.__browseButton = Button(self.__master, 
                                   command=self.__selectFile,
                                   font=ScreenTimeGui.mainFont,
                                   text="browse")
        self.__browseButton.grid(row=0, column=1, padx=(10, 10))
        
        # Progress bar setup for later
        self.__progBar = Progressbar(self.__master,
                                   orient="horizontal",
                                   length=300,
                                   mode="determinate")
        mainloop()
            
    def __selectFile(self):
        # Opening file
        vidFile = tkFileDialog.askopenfile(parent=self.__master,
                                             mode='rb',
                                             title='Select the video')
        if(vidFile != None):
            vidPath = path.abspath(vidFile.name)
            self.__startComputation(vidPath)
            
    def __startComputation(self, vidPath):
        # Change the UI to reflect the status
        self.__browseButton.grid_remove()
        
        self.__instructionsText.config(text="Setting up ...")
        
        self.__progBar.grid(row=1, column=0)
        
        #self.p1 = Process(target=self.__compute, args = (vidPath, self))
        #self.p1.start()
        #self.__progBar.start(80)
        #self.after(20, self.progress)
        
        # Call the compute function
        th = Thread(target =  self.__compute, args = (vidPath, self))
        th.start()
        
        
    def setProgressMax(self, maxVal):
        # Sets up the maximum amount for the progress bar and starts it
        self.__progBar["maximum"] = maxVal + 1
        self.__instructionsText.config(text="Face detection on video ... (%d/%d)" \
                                       % (0,maxVal))
        
    def progress(self):
        # Updates the progress bar by incrementing 1 
        newVal = self.__progBar["value"] + 1
        
        self.__instructionsText.config(text="Face detection on video... (%d/%d)" \
                                       % (newVal,self.__progBar["maximum"]))
        self.__progBar.step()
        
    def clusterState(self):
        self.__instructionsText.config(text="Clustering the faces ...")
        self.__progBar.config(mode="indeterminate")
        self.__progBar.start()