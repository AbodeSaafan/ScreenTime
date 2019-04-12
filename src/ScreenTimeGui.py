from tkinter import Tk, Label, Button, Canvas, Scale, mainloop, ttk, filedialog
from PIL import Image, ImageTk
from os import path
from threading import Thread
import numpy as np
from FaceCluster import FaceCluster

class ScreenTimeGui:
    mainFont = ("Fixedsys", 18)
    subFont = ("Fixedsys", 14)

    def __init__(self, computeFunc):
        # Initalizing vars
        self.fc = FaceCluster()
        self.genderClusters = []
        self.clusterShares = []
        self.maleGenderShare = 0
        self.femaleGenderShare = 0
        
        # Compute function that we will use later
        self.__compute = computeFunc
        
        # Master window
        self.__master = Tk()
        self.__master.resizable(False, False)
        self.__master.title("ScreenTime")
        
        # Text for user instructions 
        self.__instructionsText = Label(self.__master,
                                      font=ScreenTimeGui.mainFont, 
                                      text="Please select the video you want to run ScreenTime on")
        self.__instructionsText.grid(row=0, column=0, padx=(40, 40))
        
        # Button for selecting video
        self.__browseButton = Button(self.__master, 
                                   command=self.__selectFile,
                                   font=ScreenTimeGui.mainFont,
                                   text="browse")
        self.__browseButton.grid(row=0, column=1, padx=(10, 0))
        
        # Progress bar setup for later
        self.__progBar = ttk.Progressbar(self.__master,
                                   orient="horizontal",
                                   length=500,
                                   mode="determinate")
        
        mainloop()
            
    def __selectFile(self):
        # Opening file
        vidFile = filedialog.askopenfile(parent=self.__master,
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

        # Call the compute function
        self.th = Thread(target =  self.__compute, args = (vidPath, self))
        self.th.start()
        
        
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
        
    def processingState(self):
        # Changes the GUI to show that clustering and classification 
        # is being processed
        self.__instructionsText.config(text="Clustering and classifying faces ...")
        self.__progBar.config(mode="indeterminate")
        self.__progBar.start()
        
        self.__seeResultsButton = Button(self.__master,
                                   command=self.showClusterResults,
                                   font=ScreenTimeGui.mainFont,
                                   text="Show results")
        
    def enableResultsButton(self):
        # Make progress bar full and stop it
        self.__progBar.stop()
        self.__progBar.config(mode="determinate")
        self.__progBar["maximum"] = self.__progBar["value"] + 1.001
        self.__progBar.step()
        
        
        self.__instructionsText.config(text="Results are ready")
        self.__seeResultsButton.grid(row=2, column=0)
        
    def __backButtonPressed(self):
        # On cluster back button press
        if(self.__selectedClusterNum - 1 >= 0):
            self.__selectCluster(self.__selectedClusterNum - 1)
        
    def __forwardButtonPressed(self):
        # On cluster forward button press
        if(self.__selectedClusterNum + 1 < self.fc.numOfClusters):
            self.__selectCluster(self.__selectedClusterNum + 1)
        
    def __selectCluster(self, c):
        if(self.genderClusters[c] == 1):
            genderText = "Predicted gender: Male" 
        else:
            genderText = "Predicted gender: Female" 
        
        percent = self.clusterShares[c]
        percentText = "Person's screentime percentage:  {:.2%}\n".format(percent)
        
        self.__infoText.config(text = percentText + genderText)
        
        # This selects the cluster c and updates some class vars
        self.__selectedClusterNum = c
        self.__clusterInfoText.config(text="Cluster " + str(c+1))
        self.__selectedCluster = self.fc.getClusterImages(c)
        self.__clusterImagesSlider.config(to=len(self.__selectedCluster))
        self.__clusterImagesSlider.set(1)
        self.__showClusterImage(1)
        
        
    def __showClusterImage(self, i):
        # This creates the image and updates the UI to show it
        self.__currentImg = ImageTk.PhotoImage(master = self.__master, 
                                 image=Image.fromarray(self.__selectedCluster[int(i) - 1]))
        
        self.__clusterImageCanvas.itemconfig(self.__imageOnCanvas,
                                             image = self.__currentImg)
        self.__clusterImageCanvas.update()
    
    def showClusterResults(self):
        self.__seeResultsButton.grid_forget()
        
        # Change UI to get ready for cluster results
        self.__instructionsText.grid_forget()
        self.__progBar.grid_forget()
        
        # Header of window
        self.__backClusterButton = Button(self.__master,
                                          command=self.__backButtonPressed,
                                          font=ScreenTimeGui.mainFont,
                                          text="<")
        self.__backClusterButton.grid(row=0, column=0, padx=(0, 60))
        
        self.__clusterInfoText = Label(self.__master,
                                       font=ScreenTimeGui.mainFont,
                                       text="Cluster 1")
        self.__clusterInfoText.grid(row=0, column=2, padx=(20,20))
        
        self.__forwardClusterButton = Button(self.__master,
                                             command=self.__forwardButtonPressed,
                                             font=ScreenTimeGui.mainFont,
                                             text=">")
        self.__forwardClusterButton.grid(row=0, column=4, padx=(60, 0))
        
        # Pictures in cluster with slider
        self.__clusterImageCanvas = Canvas(self.__master,
                                           width=200,
                                           height=200)
        self.__clusterImageCanvas.grid(row=2, column=2)
        
        img = ImageTk.PhotoImage(master = self.__master, 
                                 image=Image.fromarray(np.zeros((20,20))))
        self.__imageOnCanvas = self.__clusterImageCanvas.create_image(0,
                                                                      0,
                                                                      image = img,
                                                                      anchor="nw")
        
        self.__clusterImagesSlider = Scale(self.__master,
                                           from_=1,
                                           to=1,
                                           orient="horizontal",
                                           command=self.__showClusterImage)
        self.__clusterImagesSlider.grid(row=3, column=2, padx=(80,80))
        
        # UI elements for the data for each cluster (share and gender)
        self.__infoText = Label(self.__master,
                                font=ScreenTimeGui.subFont,
                                text="")
        self.__infoText.grid(row=4, column=2, padx=(60,60))
        
        # The info display for the entire program (gender)
        maleSplit = sum(self.genderClusters)/(len(self.genderClusters))
        self.__genderSplit = Label(self.__master,
                                   font=ScreenTimeGui.subFont,
                                   borderwidth=2,
                                   relief="groove",
                                   text="Actor ratio\nMale: {:.0%}\nFemale: {:.0%}".format(
                                   maleSplit,
                                   1.0-maleSplit))
        
        self.__genderSplit.grid(row=5, column=1, padx=(60,60))
        
        self.__genderScreentime = Label(self.__master,
                                        font=ScreenTimeGui.subFont,
                                        borderwidth=2,
                                        relief="groove",
                                        text=
                                        "Gender screentime\nMale: {:.0%}\nFemale: {:.0%}".format(
                                            self.maleGenderShare,
                                            self.femaleGenderShare))
        self.__genderScreentime.grid(row=5, column=3, padx=(60,60))
        
        
        # Select first cluster and image to initalize UI
        self.__selectCluster(0)
        self.__showClusterImage(1)