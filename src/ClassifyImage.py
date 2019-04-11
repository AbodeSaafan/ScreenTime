from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import glob
from random import sample
from cv2 import resize


class ClassifyImage():
    
    def __init__(self, modelName):
        self.__modelName = modelName
        self.__model = self.__loadModelandWeights()
    
    def __loadModelandWeights(self):
        # load json and create model
        JSONfile = open(self.__modelName+".json", "r")
        loadedModelJSON = JSONfile.read()
        JSONfile.close()
        loadedModel = model_from_json(loadedModelJSON)
        # load weights into new model
        loadedModel.load_weights(self.__modelName+".h5")
        loadedModel.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        # Loaded model from disk
        return loadedModel
        
    def classifyImage(self, img):
        #TODO comments
        newImg = image.img_to_array(img)
        newImg = np.expand_dims(newImg, axis=0)
        result = self.__model.predict(newImg)

        return int(result[0][0])
    
    def classifyCluster(self, clusterImages, ratio):
        # We pick n random images where n is based on ratio 
        n = int(len(clusterImages) * ratio)
        randIndexes = sample(range(clusterImages) - 1, n)

        # Classify each random image
        results = []        
        for i in range(len(randIndexes)):
            randomImage = clusterImages[randIndexes[i]]
            r = self.classifyImage(resize(randomImage, (250,250)))
            results.append(r)
        
        print(r)
        
    def testingAccuracy(self, modelName, dataPath, gender='male'):
        filenames = glob.glob(dataPath)
        filenames.sort()
        images = [image.load_img(img, target_size = (250,250)) for img in filenames]
        
#        ci = ClassifyImage.ClassifyImage()
        model = self.loadModelandWeights(modelName)
        results = []
        for i in range(len(images)):
            results.append(self.classifyImage(model, images[i]))
            
        
        print('{} out of {}'.format(results.count(gender), len(filenames)))

        