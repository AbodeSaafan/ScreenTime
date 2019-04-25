# ScreenTime
This project was created for CSC420(Introduction to Image Understanding) at the University of Toronto Mississauga for Dr. Sanja Fidler.


## Introduction
The objective of our system is to learn the on-screen time of actors in a video clip. From the screen-time of each actor, we can calculate a percentage of how much an actor appeared. Furthermore, we are able to predict the gender of each actor. This gender classication allows us to discover the ratio of female to male actors. The system is ultimately able to give a percentage of on-screen
time for each actor and each gender.

The design of this system is explained in more detail in the report attached below. The system uses a combination of Python's openCV library and Keras/TensorFlow. The Tkinter library was also used for a user-friendly GUI.

## Running it
From an anaconda shell (prompt), navigate to the src folder and simply run main.py

`python main.py`

To adjust or fine tune some arguments, you first use the help to get the variable names

`python main.py -h`

For conveinence, the output for the command's help is below: 

~~~
usage: main.py [-h] [--sampleRate [1 <= sampleRate <= video frame rate]]
               [--clusterMinSamples [1 <= clusterMinSamples]]
               [--faceDetectionMinN [1 <= faceDetectionMinN]]
               [--clusterEpsi [0 <= clusterEpsi]]
               [--faceVectorScale [0 < faceVectorScale <= 1]]

ScreenTime application

optional arguments:
  -h, --help            show this help message and exit
  --sampleRate [1 <= sampleRate <= video frame rate]
                        How often per second we sample the video
  --clusterMinSamples [1 <= clusterMinSamples]
                        MinSamples argument for DBSCAN clustering
  --faceDetectionMinN [1 <= faceDetectionMinN]
                        Min neighbours for face detection
  --clusterEpsi [0 <= clusterEpsi]
                        Epsilon argument for DBSCAN clustering
  --faceVectorScale [0 < faceVectorScale <= 1]
                        Scale argument for vector conversion
~~~

Then you can run it however you would like:

`python main.py --sampleRate 10 --clusterEpsi 0.5`


## Report
A report that describes this project as well as our research during the project is available [here](https://github.com/AbodeSaafan/ScreenTime/blob/master/report.pdf). 
