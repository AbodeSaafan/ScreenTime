from Tkinter import * 
import tkFileDialog

class ScreenTimeGui:
    InitialLoad = 0
    
    def __init__(self):
        # Master window
        self.master = Tk()
        self.master.title("Screen Time")
        
        self.instructionsText = Text(self.master)
        self.instructionsText.insert(INSERT, 
                                     "Please select the video you want to run ScreenTime on")
        
        self.browseButton = Button(self.master, command=self.__selectFile, text="browse")
        self.browseButton.pack()
    
    def display(self):
        mainloop()
        
    def __selectFile(self):
        self.file = tkFileDialog.askopenfile(parent=self.master,mode='rb',title='Select the video')
